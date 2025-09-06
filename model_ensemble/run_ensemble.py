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

import pandas as pd

# --------------------------- Defaults & paths ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_A_PATH = REPO_ROOT / "model_a_direction" / "model_a_v_1.py"
MODEL_B_PATH = REPO_ROOT / "model_b_strength" / "model_b_v1.py"
MODEL_C_PATH = REPO_ROOT / "model_c_momentum" / "model_c_v1.py"
OUTPUTS_ROOT = REPO_ROOT / "model_ensemble" / "outputs"
LOGS_ROOT = REPO_ROOT / "model_ensemble" / "logs"

# --------------------------- Helpers ---------------------------

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
    ap.add_argument("--barometer-kind", type=str, default="btc_eth_sol_333", help="Barometer for A & B: bitcoin|ethereum|solana|btc_eth_5050|btc_eth_sol_333")
    ap.add_argument("--days-ab", type=int, default=365, help="Days to request for A/B (demo cap 365)")
    ap.add_argument("--cg-key-file", type=str, default=None, help="Path to coingecko.key for A/B (optional)")

    ap.add_argument("--max-pairs", type=int, default=120, help="Model C: limit the number of EUR pairs to consider")
    ap.add_argument("--since-days-c", type=int, default=400, help="Model C: days of OHLC history to request")
    ap.add_argument("--N", type=int, default=10, help="Model C: target breadth")
    ap.add_argument("--portfolio-eur", type=float, default=1000.0, help="Model C: portfolio notional (for EUR floor calc)")

    ap.add_argument("--exp-uptrend", type=float, default=1.0, help="Exposure scalar for Uptrend")
    ap.add_argument("--exp-range", type=float, default=0.25, help="Exposure scalar for Range")
    ap.add_argument("--exp-downtrend", type=float, default=0.0, help="Exposure scalar for Downtrend")

    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: model_ensemble/outputs/<ts>)")

    args = ap.parse_args()

    ts_root = _ts()
    outdir = Path(args.outdir) if args.outdir else (OUTPUTS_ROOT / ts_root)
    _ensure_dirs(outdir)

    exposure = ExposurePolicy(uptrend=args.exp_uptrend, range_=args.exp_range, downtrend=args.exp_downtrend)

    paths = run_models(
        barometer_kind=args.barometer_kind,
        days_ab=args.days_ab,
        cg_key_file=Path(args.cg_key_file) if args.cg_key_file else None,
        max_pairs=args.max_pairs,
        since_days_c=args.since_days_c,
        N=args.N,
        exposure=exposure,
        portfolio_eur=args.portfolio_eur,
        outdir=outdir,
    )

    build_final_allocation(paths, exposure, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
