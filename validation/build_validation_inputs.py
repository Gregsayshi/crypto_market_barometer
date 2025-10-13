from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure repo root is on sys.path so we can import sibling packages
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Model A (robust aliasing) ---
try:
    from model_a_direction.model_a_v_1 import AParams, compute_exposure_A  # v1.1 naming
except ImportError:
    # Fallback if your file exposes `compute_exposure` instead of `compute_exposure_A`
    from model_a_direction.model_a_v_1 import AParams  # noqa
    try:
        from model_a_direction.model_a_v_1 import compute_exposure as compute_exposure_A  # alias
    except ImportError as e:
        raise ImportError("Cannot import Model A: expected compute_exposure_A or compute_exposure in model_a_v_1.py") from e

# --- Model B (robust aliasing) ---
try:
    from model_b_strength.model_b_v1 import BParams, compute_exposure_B  # v1.1 naming
except ImportError:
    from model_b_strength.model_b_v1 import BParams  # noqa
    try:
        from model_b_strength.model_b_v1 import compute_exposure as compute_exposure_B  # alias
    except ImportError as e:
        raise ImportError("Cannot import Model B: expected compute_exposure_B or compute_exposure in model_b_v1.py") from e

# --- Model C (same path) ---
from model_c_momentum.model_c_v1 import CParams, build_weights_history
def _read_long_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date","asset"])
    return df

def _series_from_long(df: pd.DataFrame, asset: str) -> pd.Series:
    s = df[df["asset"]==asset].set_index("date")["close"].sort_index()
    return s

def _combine_allocator(exp_A: pd.Series, exp_B: pd.Series, w_A: float, w_B: float,
                       caps: dict[str,float]) -> pd.Series:
    t = exp_A.index.union(exp_B.index)
    a = exp_A.reindex(t).ffill()
    b = exp_B.reindex(t).ffill()
    raw = (w_A * a + w_B * b).clip(0,1)

    # Map to regime caps via nearest regime of the *average* signal
    # For v1.1 keep simple: range cap applies when raw in (0,1), uptrend when >=0.75, downtrend when <=0.25
    cap_series = pd.Series(caps.get("range", 1.0), index=t)
    cap_series[raw >= 0.75] = caps.get("uptrend", 1.0)
    cap_series[raw <= 0.25] = caps.get("downtrend", 0.0)

    return (raw * cap_series).clip(0,1)

def build_inputs(
    barometer_prices_csv: str,
    universe_prices_csv: str,
    a_cfg: dict,
    b_cfg: dict,
    c_cfg: dict,
    allocator_cfg: dict,
    outdir: str
):
    outp = Path(outdir); outp.mkdir(parents=True, exist_ok=True)

    # --- A: direction (single series, e.g., BTC)
    baro = _read_long_prices(barometer_prices_csv)
    sym_map = a_cfg.get("symbol_map", {}) or {}
    btc = sym_map.get("bitcoin", "XBTEUR")
    ser_btc = _series_from_long(baro, btc)
    expA = compute_exposure_A(ser_btc, AParams(
        lookbacks=a_cfg.get("lookbacks",[120,180,300]),
        skip_days=a_cfg.get("skip_days",7),
        enter_up_z=a_cfg.get("enter_up_z",0.25), exit_up_z=a_cfg.get("exit_up_z",0.10),
        enter_down_z=a_cfg.get("enter_down_z",-0.25), exit_down_z=a_cfg.get("exit_down_z",-0.10),
        dwell_days=a_cfg.get("dwell_days",5),
        smooth_ema_days=a_cfg.get("smooth_ema_days",0),
    ))
    expA = expA.set_index("date")["exp_A"]

    # --- B: breadth/strength (e.g., BTC/ETH/SOL)
    eth = sym_map.get("ethereum", "ETHEUR")
    sol = sym_map.get("solana",   "SOLEUR")
    basket = {btc: ser_btc, eth: _series_from_long(baro, eth)}
    if sol in baro["asset"].unique():
        basket[sol] = _series_from_long(baro, sol)
    expB = compute_exposure_B(basket, BParams(
        er_window=b_cfg.get("er_window",20),
        enter_strong=b_cfg.get("enter_strong",0.30), exit_strong=b_cfg.get("exit_strong",0.25),
        enter_weak=b_cfg.get("enter_weak",0.20),    exit_weak=b_cfg.get("exit_weak",0.25),
        dwell_days=b_cfg.get("dwell_days",3),
        smooth_ema_days=b_cfg.get("smooth_ema_days",0),
    ))
    expB = expB.set_index("date")["exp_B"]

    # --- Allocator: combine A and B
    blend = allocator_cfg.get("blend", {}) or {}
    wA = float(blend.get("w_A", 0.5)); wB = float(blend.get("w_B", 0.5))
    caps = allocator_cfg.get("exp_caps", {"uptrend":1.0,"range":1.0,"downtrend":0.0})
    exposure = _combine_allocator(expA, expB, wA, wB, caps).rename("exposure_scalar").reset_index()
    exposure.to_csv(outp / "exposures.csv", index=False)

    # --- C: cross-section weights from *filtered* universe
    cparams = CParams(
        mode=c_cfg.get("mode","equal_weight"),
        N=c_cfg.get("N",10),
        cap=c_cfg.get("cap",0.25),
        floor_eur=c_cfg.get("floor_eur",250.0),
        portfolio_eur=c_cfg.get("portfolio_eur",10000.0),
        skip_days=c_cfg.get("skip_days",3),
        rebalance=c_cfg.get("rebalance","4W-SUN"),
        hysteresis=c_cfg.get("hysteresis",3),
        momo_lb=c_cfg.get("momo_lb",60),
    )
    weights = build_weights_history(universe_prices_csv, cparams,
                                    start=c_cfg.get("start"), end=c_cfg.get("end"))
    weights.to_csv(outp / "weights_history.csv", index=False)

def main():
    ap = argparse.ArgumentParser(description="Build exposures (A+B→allocator) and cross-section weights (C) from local CSVs.")
    ap.add_argument("--barometer-prices", required=True)
    ap.add_argument("--universe-prices", required=True)
    ap.add_argument("--outdir", required=True)

    # Optional JSON overrides (keep CLI simple; your offline_runner passes dicts)
    ap.add_argument("--a-json", default=None)
    ap.add_argument("--b-json", default=None)
    ap.add_argument("--c-json", default=None)
    ap.add_argument("--allocator-json", default=None)
    args = ap.parse_args()

    import json
    j = lambda s: json.loads(s) if s else {}

    build_inputs(
        args.barometer_prices,
        args.universe_prices,
        a_cfg=j(args.a_json),
        b_cfg=j(args.b_json),
        c_cfg=j(args.c_json),
        allocator_cfg=j(args.allocator_json),
        outdir=args.outdir,
    )

if __name__ == "__main__":
    main()
