#!/usr/bin/env python3
"""
offline_runner.py — Single-YAML, fully-offline validation pipeline

Reads config.yaml, computes Model A+B from a local prices CSV (no APIs),
builds Model C weights history from a local universe CSV, then runs the
backtester — all in one command. Outputs are timestamped so runs never overwrite.

Usage:
  python validation/offline_runner.py --config config.yaml
"""
from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# --- repo roots & imports ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so we can import repo modules

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

import pandas as pd
import numpy as np

# Reuse your existing modules
from model_a_direction.model_a_v_1 import (
    run_model_a, LocalCSVClient as ALocalClient, LocalCSVSourceConfig as ALocalCfg,
    ModelAConfig as AConfig, FetchConfig as AFetch
)
from model_b_strength.model_b_v1 import (
    run_model_b, LocalCSVClient as BLocalClient, LocalCSVSourceConfig as BLocalCfg,
    ModelBConfig as BConfig, FetchConfig as BFetch
)
from validation.build_validation_inputs import (
    CParams, build_weights_history
)
from validation.run_backtest import main as run_backtest_main

# --------------------- helpers ---------------------

def _as_datestr(x):
    from datetime import date, datetime
    if isinstance(x, (datetime, date)):
        return x.strftime('%Y-%m-%d')
    return str(x) if x is not None else None
    
def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _coerce_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s: return None
    return pd.to_datetime(s).tz_localize(None)

def _compose_regime(a_state: str, b_state: str) -> str:
    a = (a_state or "").strip().capitalize()
    b = (b_state or "").strip().capitalize()
    if a == "Up" and b == "Strong":
        return "Uptrend"
    if a == "Down" and b == "Strong":
        return "Downtrend"
    return "Range"

def _symbol_map_from_yaml(d: Dict[str, str]) -> Dict[str,str]:
    # keys should be coin ids: 'bitcoin','ethereum','solana'
    out = {}
    for k,v in (d or {}).items():
        out[str(k).lower()] = str(v)
    return out

# --------------------- core ---------------------

def build_exposures_from_ab_data(a_df: pd.DataFrame,
                                 b_df: pd.DataFrame,
                                 scalars: Dict[str, float],
                                 start: Optional[str],
                                 end: Optional[str]) -> pd.DataFrame:
    """Compose daily exposures from already-computed A and B dataframes."""
    A = a_df.copy(); B = b_df.copy()
    # Normalize
    A.index = pd.to_datetime(A.index, utc=True).tz_convert(None).normalize()
    B.index = pd.to_datetime(B.index, utc=True).tz_convert(None).normalize()

    # Columns
    a_col = "state" if "state" in A.columns else ("direction" if "direction" in A.columns else None)
    b_col = "strength_state" if "strength_state" in B.columns else ("state" if "state" in B.columns else None)
    if not a_col or not b_col:
        raise SystemExit("Model A or B outputs missing state columns.")

    df = A[[a_col]].rename(columns={a_col: "A_state"}).join(
         B[[b_col]].rename(columns={b_col: "B_state"}),
         how="inner"
    ).sort_index()

    if start: df = df[df.index >= pd.to_datetime(start)]
    if end:   df = df[df.index <= pd.to_datetime(end)]

    regime = df.apply(lambda r: _compose_regime(r["A_state"], r["B_state"]), axis=1)
    expo   = regime.map({
        "Uptrend":   float(scalars.get("uptrend",   1.0)),
        "Range":     float(scalars.get("range",     0.25)),
        "Downtrend": float(scalars.get("downtrend", 0.0))
    }).astype(float)

    out = pd.DataFrame({"exposure_scalar": expo, "regime": regime}, index=df.index)
    return out

def run_offline(cfg_path: Path) -> int:
    if yaml is None:
        raise SystemExit("PyYAML is required. Please: pip install pyyaml")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # --- config fields ---
    meta = cfg.get("meta", {})
    data = cfg.get("data", {})
    universe = data.get("universe_prices")
    baro     = data.get("barometer_prices", universe)  # allow reuse of same CSV
    assets_map = _symbol_map_from_yaml(data.get("symbol_map", {"bitcoin":"XBTEUR","ethereum":"ETHEUR","solana":"SOLEUR"}))

    back = cfg.get("backtest", {})
    start = back.get("start"); end = back.get("end")
    prices_wide_csv = data.get("prices_wide", universe)  # wide panel CSV for backtester (date,asset,close)
    benches = list(data.get("benchmarks", []))

    # Model params
    a_params = cfg.get("model_a", {})
    b_params = cfg.get("model_b", {})
    c_params = cfg.get("model_c", {})
    alloc    = cfg.get("allocator", {})

    # Paths/out
    out_root_runs   = Path(back.get("outdir_root", "validation/runs"))
    out_root_inputs = Path(cfg.get("outputs", {}).get("inputs_outdir_root", "validation/inputs"))
    write_inputs = bool(cfg.get("outputs", {}).get("write_inputs", True))

    # --- timestamped folders ---
    ts = _ts_utc()
    run_tag = str(meta.get("run_tag") or "offline")
    run_dir = out_root_runs / f"{ts}_{run_tag}"
    inp_dir = out_root_inputs / f"{ts}_{run_tag}"
    _ensure_dir(run_dir); _ensure_dir(inp_dir)

    # --- A: run from local CSV (barometer) ---
    a_client = ALocalClient(ALocalCfg(csv_path=baro, symbol_map=assets_map))
    a_cfg = AConfig(
        lookbacks=tuple(a_params.get("lookbacks", [120,180,300])),
        skip_days=int(a_params.get("skip_days", 7)),
        enter_up_z=float(a_params.get("enter_up_z", +0.25)),
        exit_up_z=float(a_params.get("exit_up_z", +0.10)),
        enter_down_z=float(a_params.get("enter_down_z", -0.25)),
        exit_down_z=float(a_params.get("exit_down_z", -0.10)),
        dwell_days=int(a_params.get("dwell_days", 5)),
    )
    a_res = run_model_a(api_key=None, barometer_kind=str(a_params.get("barometer_kind", "btc_eth_sol_333")),
                        fetch_cfg=AFetch(days=540), model_cfg=a_cfg, client=a_client)
    a_df = a_res.data  # has 'state'/'direction', 'D', etc.

    # --- B: run from local CSV (same barometer) ---
    b_client = BLocalClient(BLocalCfg(csv_path=baro, symbol_map=assets_map))
    b_cfg = BConfig(
        N=int(b_params.get("er_window", 20)),
        enter_strong=float(b_params.get("enter_strong", 0.30)),
        exit_strong=float(b_params.get("exit_strong", 0.25)),
        enter_weak=float(b_params.get("enter_weak", 0.20)),
        exit_weak=float(b_params.get("exit_weak", 0.25)),
        dwell_days=int(b_params.get("dwell_days", 3)),
    )
    b_res = run_model_b(api_key=None, barometer_kind=str(b_params.get("barometer_kind", a_params.get("barometer_kind","btc_eth_sol_333"))),
                        fetch_cfg=BFetch(days=540), model_cfg=b_cfg, client=b_client)
    b_df = b_res.data  # has 'strength_state'

    # --- exposures: A+B -> regime -> scalar ---
    scalars = {
        "uptrend":   float(alloc.get("exp_uptrend",   1.0)),
        "range":     float(alloc.get("exp_range",     1.0)),
        "downtrend": float(alloc.get("exp_downtrend", 0.0)),
    }
    exposures = build_exposures_from_ab_data(a_df, b_df, scalars, start, end)
    if write_inputs:
        (inp_dir / "exposures.csv").write_text(
            exposures.reset_index().rename(columns={"index":"date"}).to_csv(index=False), encoding="utf-8"
        )

    # --- Model C weights history (from local universe CSV) ---
    cp = CParams(
        N=int(c_params.get("N", 15)),
        cap=float(c_params.get("cap", 0.25)),
        floor_eur=float(c_params.get("floor_eur", 250.0)),
        portfolio_eur=float(c_params.get("portfolio_eur", 10000.0)),
        skip_days=int(c_params.get("skip_days", 3)),
        rebalance=str(c_params.get("rebalance", "4W-SUN")),
    )
    weights_hist = build_weights_history(Path(universe), cp, start, end)
    if write_inputs:
        weights_hist.to_csv(inp_dir / "weights_history.csv", index=False)

    # --- Backtest (using your runner) ---
    bt_argv = [
        "--exposures", str(inp_dir / "exposures.csv"),
        "--weights",   str(inp_dir / "weights_history.csv"),
        "--prices",    str(prices_wide_csv),
        "--signal-lag", str(int(back.get("signal_lag", 1))),
        "--tc-bps",     str(float(back.get("tc_bps", 10))),
        "--outdir",     str(run_dir),
    ]
    if start: bt_argv += ["--start", start]
    if end:   bt_argv += ["--end", end]
    for b in benches or []:
        bt_argv += ["--bench", str(b)]

    rc = run_backtest_main(bt_argv)

    # Manifest for reproducibility
    manifest = {
        "timestamp_utc": ts,
        "config_path": str(cfg_path.resolve()),
        "inputs": {
            "barometer_prices": str(Path(baro).resolve()),
            "universe_prices":  str(Path(universe).resolve()),
            "prices_wide":      str(Path(prices_wide_csv).resolve()),
        },
        "scalars": scalars,
        "window": {"start": start, "end": end},
        "folders": {"inputs": str(inp_dir.resolve()), "run": str(run_dir.resolve())},
        "benches": benches,
        "shapes": {
            "exposures_rows": int(len(exposures)),
            "weights_rows": int(len(weights_hist)),
            "rebalances": int(weights_hist["date"].nunique()) if not weights_hist.empty else 0
        }
    }
    (run_dir / "offline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Offline run saved to {run_dir}")
    return rc

# --------------------- CLI ---------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Offline validation runner (single YAML)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args(argv)
    return run_offline(Path(args.config))

if __name__ == "__main__":
    raise SystemExit(main())
