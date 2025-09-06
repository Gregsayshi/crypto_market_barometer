"""
run_ensemble.py — Orchestrate Models A, B, C and emit a final allocation
-------------------------------------------------------------------------

Repo layout (assumed):
  model_a_direction/model_a_v_1.py
  model_b_strength/model_b_v1.py
  model_c_momentum/model_c_v1.py
  model_ensemble/run_ensemble.py  <-- this file

What this does (MVP):
1) Runs Model A (direction) and Model B (trend strength) on the chosen barometer.
2) Runs Model C (cross-sectional momentum) over Kraken's EUR universe.
3) Combines A+B into regimes (Uptrend / Range / Downtrend), maps those to an exposure scalar,
   and scales Model C weights to produce a final long-only allocation for the sleeve.
4) Writes all outputs under a timestamped run folder, e.g. model_ensemble/outputs/20250903_181530Z/.

Notes:
- This uses subprocess to call the existing scripts, passing explicit output paths so we know where files are.
- CoinGecko API key for A/B is optional; pass --cg-key-file if you have coingecko.key at repo root (or elsewhere).
- The exposure policy (scalar per regime) is configurable via CLI flags.

"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import yaml
from copy import deepcopy

import pandas as pd

# --------------------------- Defaults & paths ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_A_PATH = REPO_ROOT / "model_a_direction" / "model_a_v_1.py"
MODEL_B_PATH = REPO_ROOT / "model_b_strength" / "model_b_v1.py"
MODEL_C_PATH = REPO_ROOT / "model_c_momentum" / "model_c_v1.py"
OUTPUTS_ROOT = REPO_ROOT / "model_ensemble" / "outputs"
LOGS_ROOT = REPO_ROOT / "model_ensemble" / "logs"

# --------------------------- Helpers ---------------------------
def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_config(path: str | None) -> dict:
    cfg = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # simple ${ENV} / ${ENV:-default} expansion
        import os, re
        def expand(val: str) -> str:
            if not isinstance(val, str): return val
            # ${VAR:-default}
            val = re.sub(r"\$\{([^}:]+):-([^}]+)\}", lambda m: os.getenv(m.group(1), m.group(2)), val)
            # ${VAR}
            return os.path.expandvars(val)
        def walk(x):
            if isinstance(x, dict): return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list): return [walk(v) for v in x]
            return expand(x)
        cfg = walk(raw)
    return cfg

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("\n»", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")

@dataclass
class ExposurePolicy:
    uptrend: float = 1.0
    range_: float = 0.5
    downtrend: float = 0.0


# --------------------------- Orchestration ---------------------------

def run_models(
    barometer_kind: str,
    days_ab: int,
    cg_key_file: Optional[Path],
    max_pairs: int,
    since_days_c: int,
    N: int,
    exposure: ExposurePolicy,
    portfolio_eur: float,
    outdir: Path,
) -> Dict[str, str]:
    """Run A, B, C and return a dict of file paths."""
    _ensure_dirs(outdir, LOGS_ROOT)

    ts = _ts()
    # A outputs
    a_out = outdir / f"modelA_output_{barometer_kind}_{ts}.csv"
    # B outputs
    b_out = outdir / f"modelB_output_{barometer_kind}_{ts}.csv"
    # C outputs
    c_w_out = outdir / f"modelC_weights_EUR_{ts}.csv"
    c_d_out = outdir / f"modelC_diagnostics_EUR_{ts}.csv"

    # 1) Model A
    a_cmd = [sys.executable, str(MODEL_A_PATH), "--barometer-kind", barometer_kind, "--days", str(days_ab), "--out", str(a_out)]
    if cg_key_file and cg_key_file.exists():
        a_cmd += ["--api-key-file", str(cg_key_file)]
    _run(a_cmd)

    # 2) Model B
    b_cmd = [sys.executable, str(MODEL_B_PATH), "--barometer-kind", barometer_kind, "--days", str(days_ab), "--out", str(b_out)]
    if cg_key_file and cg_key_file.exists():
        b_cmd += ["--api-key-file", str(cg_key_file)]
    _run(b_cmd)

    # 3) Model C
    c_cmd = [
        sys.executable, str(MODEL_C_PATH),
        "--max-pairs", str(max_pairs),
        "--since-days", str(since_days_c),
        "--N", str(N),
        "--notional-eur", str(portfolio_eur),
        "--out-weights", str(c_w_out),
        "--out-diag", str(c_d_out),
    ]
    _run(c_cmd)

    return {
        "A": str(a_out),
        "B": str(b_out),
        "C_WEIGHTS": str(c_w_out),
        "C_DIAG": str(c_d_out),
    }


def _latest_state_a(a_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(a_csv, index_col=0, parse_dates=True)
    row = df.iloc[-1]
    state = str(row.get("state", row.get("direction", "")))
    date = str(df.index[-1].date())
    return {"date": date, "state": state}


def _latest_state_b(b_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(b_csv, index_col=0, parse_dates=True)
    row = df.iloc[-1]
    st = str(row.get("strength_state", ""))
    date = str(df.index[-1].date())
    return {"date": date, "strength_state": st}


def _compose_regime(a_state: str, b_state: str) -> str:
    a = (a_state or "").strip().capitalize()
    b = (b_state or "").strip().capitalize()
    if a == "Up" and b == "Strong":
        return "Uptrend"
    if a == "Down" and b == "Strong":
        return "Downtrend"
    return "Range"


def build_final_allocation(paths: Dict[str,str], exposure: ExposurePolicy, outdir: Path) -> Path:
    a_csv = Path(paths["A"]) ; b_csv = Path(paths["B"]) ; w_csv = Path(paths["C_WEIGHTS"]) ; d_csv = Path(paths["C_DIAG"]) 

    a = _latest_state_a(a_csv)
    b = _latest_state_b(b_csv)
    regime = _compose_regime(a_state=a["state"], b_state=b["strength_state"])
    if regime == "Uptrend":
        scalar = exposure.uptrend
    elif regime == "Downtrend":
        scalar = exposure.downtrend
    else:
        scalar = exposure.range_

    # Read Model C weights (prefer weights CSV; fall back to diagnostics if needed)
    if w_csv.exists():
        w = pd.read_csv(w_csv)
        w = w.rename(columns={"pair_key":"pair_key", "wsname":"wsname", "weight":"weight_modelC"})
    else:
        d = pd.read_csv(d_csv)
        d = d.rename(columns={"weight":"weight_modelC"})
        w = d.loc[d["weight_modelC"]>0, ["pair_key","wsname","weight_modelC"]]

    w["exposure_scalar"] = scalar
    w["final_weight"] = w["weight_modelC"] * w["exposure_scalar"]

    # Summary header row (repeat per row for simplicity)
    w["modelA_state"] = a["state"]
    w["modelB_strength"] = b["strength_state"]
    w["regime"] = regime
    w["modelA_date"] = a["date"]
    w["modelB_date"] = b["date"]

    ts = _ts()
    out_final = outdir / f"final_allocation_{regime}_{ts}.csv"
    w.to_csv(out_final, index=False)
    
    # Write a small JSON summary too
    summary = {
        "timestamp_utc": ts,
        "modelA": a,
        "modelB": b,
        "regime": regime,
        "exposure_scalar": scalar,
        "rows": int(len(w)),
        "weights_sum_modelC": float(w["weight_modelC"].sum()),
        "weights_sum_final": float(w["final_weight"].sum()),
        "paths": paths,
    }
    with open(outdir / f"run_summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRegime: {regime} | scalar={scalar} | A({a['date']})={a['state']} | B({b['date']})={b['strength_state']}")
    print(f"Final allocation saved → {out_final}")
    return out_final


# --------------------------- CLI ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run Models A+B+C and emit final allocation")
    # YAML config
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")

    # Regular CLI (can override YAML)
    ap.add_argument("--barometer-kind", type=str, default=None,
                    help="bitcoin|ethereum|solana|btc_eth_5050|btc_eth_sol_333")
    ap.add_argument("--days-ab", type=int, default=None, help="A/B: days to request (demo cap 365)")
    ap.add_argument("--cg-key-file", type=str, default=None, help="Path to coingecko.key (optional)")

    ap.add_argument("--max-pairs", type=int, default=None, help="C: limit number of EUR pairs to consider")
    ap.add_argument("--since-days-c", type=int, default=None, help="C: days of OHLC history to request")
    ap.add_argument("--N", type=int, default=None, help="C: target breadth")
    ap.add_argument("--portfolio-eur", type=float, default=None, help="C: portfolio notional for € floor calc")

    ap.add_argument("--exp-uptrend", type=float, default=None, help="Allocator scalar for Uptrend")
    ap.add_argument("--exp-range",   type=float, default=None, help="Allocator scalar for Range")
    ap.add_argument("--exp-downtrend", type=float, default=None, help="Allocator scalar for Downtrend")

    ap.add_argument("--outdir", type=str, default=None, help="Output directory (else uses config/meta or default)")

    args = ap.parse_args()

    # --- Load YAML and merge with defaults ---
    yaml_cfg = load_config(args.config)  # uses the helper from earlier
    defaults = {
        "meta": {"output_root": str(OUTPUTS_ROOT), "logs_root": str(LOGS_ROOT)},
        "barometer": {"kind": "btc_eth_sol_333"},
        "data": {
            "coingecko": {"api_tier": "demo", "vs_currency": "usd", "days": 365, "api_key_file": None},
            "kraken":    {"max_pairs": 120, "since_days": 400},
        },
        "model_c": {
            "N_target": 10,
            "N_min": 8, "N_max": 12,
            "notional_floor_eur": 25.0,
            "portfolio_notional_eur": 1000.0,
            "weight_cap": 0.25,
        },
        "allocator": {"exposure_scalars": {"uptrend": 1.0, "range": 0.25, "downtrend": 0.0}},
    }
    cfg = deep_update(defaults, yaml_cfg or {})

    # --- CLI overrides YAML (only for flags actually provided) ---
    if args.barometer_kind:        cfg["barometer"]["kind"] = args.barometer_kind
    if args.days_ab is not None:   cfg["data"]["coingecko"]["days"] = args.days_ab
    if args.cg_key_file:           cfg["data"]["coingecko"]["api_key_file"] = args.cg_key_file

    if args.max_pairs is not None:   cfg["data"]["kraken"]["max_pairs"] = args.max_pairs
    if args.since_days_c is not None: cfg["data"]["kraken"]["since_days"] = args.since_days_c
    if args.N is not None:            cfg["model_c"]["N_target"] = args.N
    if args.portfolio_eur is not None: cfg["model_c"]["portfolio_notional_eur"] = args.portfolio_eur

    scalars = cfg["allocator"]["exposure_scalars"]
    if args.exp_uptrend is not None:  scalars["uptrend"] = args.exp_uptrend
    if args.exp_range is not None:    scalars["range"]   = args.exp_range
    if args.exp_downtrend is not None:scalars["downtrend"] = args.exp_downtrend

    # --- Materialize values from cfg ---
    barometer_kind = cfg["barometer"]["kind"]
    days_ab        = int(cfg["data"]["coingecko"]["days"])
    cg_key_file    = cfg["data"]["coingecko"].get("api_key_file")

    max_pairs      = int(cfg["data"]["kraken"]["max_pairs"])
    since_days_c   = int(cfg["data"]["kraken"]["since_days"])
    N              = int(cfg["model_c"]["N_target"])
    portfolio_eur  = float(cfg["model_c"]["portfolio_notional_eur"])

    exposure = ExposurePolicy(
        uptrend=float(scalars["uptrend"]),
        range_=float(scalars["range"]),
        downtrend=float(scalars["downtrend"]),
    )

    # --- Output directory (args > YAML meta > default) ---
    ts_root = _ts()
    outdir = Path(args.outdir) if args.outdir else Path(cfg.get("meta", {}).get("output_root", str(OUTPUTS_ROOT))) / ts_root
    _ensure_dirs(outdir)

    # --- Run models and build final allocation ---
    paths = run_models(
        barometer_kind=barometer_kind,
        days_ab=days_ab,
        cg_key_file=Path(cg_key_file) if cg_key_file else None,
        max_pairs=max_pairs,
        since_days_c=since_days_c,
        N=N,
        exposure=exposure,
        portfolio_eur=portfolio_eur,
        outdir=outdir,
    )

    build_final_allocation(paths, exposure, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())