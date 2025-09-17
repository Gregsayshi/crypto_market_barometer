#!/usr/bin/env python3
# validation/build_validation_inputs.py — Build exposures & weights history for run_backtest.py
#
# What it does
# ------------
# 1) Reads the full daily logs from Model A and Model B (their CSVs) and composes
#    a *daily* exposure series by mapping A+B → Regime → Exposure Scalar.
#    Output: CSV with columns [date, exposure_scalar, regime].
#
# 2) Builds a *weights history* for Model C using your local long CSV of prices
#    (date, asset, close). At each rebalance date (default: W-MON), it computes
#    cross-sectional momentum (R90 & R300 with a 7-day skip) using ONLY past data
#    up to that date, selects ~N assets, applies caps & an EUR floor, and writes
#    [date, asset, weight] rows (one block per rebalance date).
#
# Then you can run your existing validation/run_backtest.py with:
#   python validation/run_backtest.py #     --exposures validation/inputs/exposures.csv #     --weights   validation/inputs/weights_history.csv #     --prices    data/prices_eur.csv #     --bench XBTEUR --bench ETHEUR #     --start 2024-01-01 --end 2024-12-31
#
# Notes
# -----
# - All timestamps are normalized to *date* (UTC naive, daily bars).
# - This script is offline-only; it never calls external APIs.
# - The 'asset' values written to weights_history.csv must match the 'asset'
#   column in your prices CSV that you'll pass to run_backtest.py.
# - If your Kraken long CSV uses e.g. 'altname' like 'ADAEUR' or 'XBTEUR',
#   ensure your prices CSV uses the same 'asset' labels.
#
# Usage
# -----
# python validation/build_validation_inputs.py #   --model-a-csv model_ensemble/outputs/20250903_181530Z/modelA_output_btc_eth_sol_333_20250903_181530Z.csv #   --model-b-csv model_ensemble/outputs/20250903_181530Z/modelB_output_btc_eth_sol_333_20250903_181530Z.csv #   --kraken-price-csv data/kraken/kraken_eur_universe.csv #   --rebalance W-MON #   --N 10 --cap 0.25 --floor-eur 25 --portfolio-eur 1000 #   --exp-uptrend 1.0 --exp-range 0.25 --exp-downtrend 0.0 #   --start 2024-01-01 --end 2024-12-31 #   --outdir validation/inputs
#
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Exposure side (A+B) -----------------------------

def _compose_regime(a_state: str, b_state: str) -> str:
    a = (a_state or "").strip().capitalize()
    b = (b_state or "").strip().capitalize()
    if a == "Up" and b == "Strong":
        return "Uptrend"
    if a == "Down" and b == "Strong":
        return "Downtrend"
    return "Range"


@dataclass
class ExposureScalars:
    uptrend: float = 1.0
    range_: float = 0.25
    downtrend: float = 0.0


def build_exposures_from_ab(a_csv: Path, b_csv: Path, scalars: ExposureScalars,
                            start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Return daily exposures DataFrame with index=date; columns: exposure_scalar, regime."""
    a = pd.read_csv(a_csv, parse_dates=[0], index_col=0)
    b = pd.read_csv(b_csv, parse_dates=[0], index_col=0)

    # Normalize column names
    a_cols = {c.lower(): c for c in a.columns}
    b_cols = {c.lower(): c for c in b.columns}
    a_state_col = a_cols.get("state") or a_cols.get("direction")
    b_state_col = b_cols.get("strength_state") or b_cols.get("state")

    a = a[[a_state_col]].rename(columns={a_state_col: "A_state"})
    b = b[[b_state_col]].rename(columns={b_state_col: "B_state"})

    df = a.join(b, how="inner")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()

    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]

    regime = df.apply(lambda r: _compose_regime(r["A_state"], r["B_state"]), axis=1)
    scalar = regime.map({"Uptrend": scalars.uptrend, "Range": scalars.range_, "Downtrend": scalars.downtrend}).astype(float)

    out = pd.DataFrame({"exposure_scalar": scalar, "regime": regime}, index=df.index)
    out = out.sort_index()
    return out


# -------------------------- Weights history (Model C) --------------------------

@dataclass
class CParams:
    N: int = 10
    cap: float = 0.25
    floor_eur: float = 25.0
    portfolio_eur: float = 1000.0
    skip_days: int = 7
    lookbacks: Tuple[int,int] = (90, 300)
    rebalance: str = "W-MON"


def _load_long_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for c in ("date", "asset", "close"):
        if c not in cols:
            raise ValueError(f"{path} must have columns date, asset, close (missing '{c}')")
    df = df[[cols["date"], cols["asset"], cols["close"]]].copy()
    df.columns = ["date", "asset", "close"]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["asset"] = df["asset"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "asset", "close"]).sort_values(["asset","date"])
    return df


def _ts_mom_with_skip(close: pd.Series, K: int, skip: int) -> float:
    # Uses only history up to the current rebalance date (exclusive of skip window)
    if len(close) < (K + skip + 1): 
        return np.nan
    p2 = float(close.iloc[-(skip+1)])
    p1 = float(close.iloc[-(K+skip+1)])
    if p1 <= 0 or p2 <= 0:
        return np.nan
    return np.log(p2) - np.log(p1)


def _rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["rank90"]  = (-out["r90"]).rank(method="average", na_option="bottom")
    out["rank300"] = (-out["r300"]).rank(method="average", na_option="bottom")
    pen = 5.0
    rank300_eff = out["rank300"] + pen * out["r300"].isna().astype(float)
    out["score"] = 0.5 * out["rank90"] + 0.5 * rank300_eff
    out["pos"] = out["score"].rank(method="first")
    return out


def _equal_weights(names: List[str]) -> Dict[str, float]:
    if not names: return {}
    w = 1.0 / float(len(names))
    return {n: w for n in names}


def _apply_caps_and_floor(weights: Dict[str, float], cap: float, floor_eur: float, portfolio_eur: float) -> Dict[str, float]:
    if not weights:
        return {}
    raw = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(raw.values())
    if s <= 0:
        return {}
    base = {k: v / s for k, v in raw.items()}
    w = {k: min(v, cap) for k, v in base.items()}
    feasible = min(1.0, cap * len(w))
    # water-fill
    def _sum(d): return sum(d.values())
    deficit = feasible - _sum(w)
    while deficit > 1e-12:
        headroom = {k: cap - w[k] for k in w if w[k] < cap - 1e-12}
        if not headroom: break
        mass = sum(base[k] for k in headroom)
        if mass <= 0: break
        for k in list(headroom.keys()):
            add = deficit * (base[k] / mass)
            w[k] = min(cap, w[k] + add)
        new_def = feasible - _sum(w)
        if abs(new_def - deficit) < 1e-12: break
        deficit = new_def
    # EUR floor
    floor_w = floor_eur / max(portfolio_eur, 1e-9)
    kept = {k: v for k, v in w.items() if v >= floor_w}
    if not kept:
        k_top = max(w, key=w.get)
        return {k_top: w[k_top]}
    # scale to feasible
    sum_kept = _sum(kept)
    target = min(1.0, cap * len(kept))
    if sum_kept > 0 and sum_kept <= target + 1e-12:
        scale = target / sum_kept
        kept = {k: min(v * scale, cap) for k, v in kept.items()}
    return kept


def build_weights_history(kraken_price_csv: Path, params: CParams,
                          start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Returns long DataFrame with columns: date, asset, weight (rebalance-only rows).
    Assumes 'asset' values match your prices CSV for run_backtest.py.
    """
    df = _load_long_prices(kraken_price_csv)
    # Calendar for rebalances
    full_idx = pd.DatetimeIndex(sorted(df["date"].unique()))
    if start:
        full_idx = full_idx[full_idx >= pd.to_datetime(start)]
    if end:
        full_idx = full_idx[full_idx <= pd.to_datetime(end)]
    if full_idx.empty:
        raise SystemExit("No dates in requested window after filtering.")

    # Rebalance dates
    cal = pd.date_range(start=full_idx.min(), end=full_idx.max(), freq=params.rebalance)
    cal = cal.intersection(full_idx)

    out_rows: List[Dict[str, object]] = []

    assets = sorted(df["asset"].unique())
    panel = {a: df[df["asset"] == a].set_index("date")["close"].asfreq("D").ffill() for a in assets}

    for d in cal:
        # Per-asset signals using only data up to 'd' (inclusive)
        stats_rows = []
        for a in assets:
            s = panel[a].loc[:d].copy()
            r90  = _ts_mom_with_skip(s, 90, params.skip_days)
            r300 = _ts_mom_with_skip(s, 300, params.skip_days)
            if np.isnan(r90) and np.isnan(r300):
                continue
            stats_rows.append({"asset": a, "r90": r90, "r300": r300})
        if not stats_rows:
            continue
        stats = pd.DataFrame(stats_rows).set_index("asset")
        ranked = _rank_panel(stats).sort_values("score")
        selected = ranked.index.tolist()[:params.N]
        w0 = _equal_weights(selected)
        w  = _apply_caps_and_floor(w0, params.cap, params.floor_eur, params.portfolio_eur)
        for a, wgt in sorted(w.items(), key=lambda kv: (-kv[1], kv[0])):
            out_rows.append({"date": d.normalize(), "asset": a, "weight": float(wgt)})
    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=["date","asset","weight"])
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values(["date","asset"])
    return out


# ---------------------------------- CLI ----------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build historical exposures & weights for validation backtests")
    ap.add_argument("--model-a-csv", type=str, required=True, help="Path to Model A daily CSV (date index; has 'state'/'direction')")
    ap.add_argument("--model-b-csv", type=str, required=True, help="Path to Model B daily CSV (date index; has 'strength_state')")
    ap.add_argument("--kraken-price-csv", type=str, required=True, help="Long CSV with Kraken EUR prices: date,asset,close")
    ap.add_argument("--rebalance", type=str, default="W-MON", help="Rebalance frequency for weights history (e.g., W-MON)")
    ap.add_argument("--N", type=int, default=10, help="Target breadth for Model C")
    ap.add_argument("--cap", type=float, default=0.25, help="Per-asset weight cap")
    ap.add_argument("--floor-eur", type=float, default=25.0, help="Absolute EUR floor per position")
    ap.add_argument("--portfolio-eur", type=float, default=1000.0, help="Portfolio notional used for EUR floor")
    ap.add_argument("--skip-days", type=int, default=7, help="Skip window for momentum calculations")
    ap.add_argument("--exp-uptrend", type=float, default=1.0, help="Exposure scalar in Uptrend")
    ap.add_argument("--exp-range", type=float, default=0.25, help="Exposure scalar in Range")
    ap.add_argument("--exp-downtrend", type=float, default=0.0, help="Exposure scalar in Downtrend")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--outdir", type=str, default="validation/inputs", help="Output folder for CSVs")

    args = ap.parse_args(argv)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    scalars = ExposureScalars(uptrend=args.exp_uptrend, range_=args.exp_range, downtrend=args.exp_downtrend)
    exposures = build_exposures_from_ab(Path(args.model_a_csv), Path(args.model_b_csv), scalars, args.start, args.end)
    exposures_path = outdir / "exposures.csv"
    exposures.reset_index().rename(columns={"index":"date"}).to_csv(exposures_path, index=False)
    print(f"Wrote exposures → {exposures_path} ({len(exposures)} rows)")

    params = CParams(N=args.N, cap=args.cap, floor_eur=args.floor_eur, portfolio_eur=args.portfolio_eur,
                     skip_days=args.skip_days, rebalance=args.rebalance)
    weights_hist = build_weights_history(Path(args.kraken_price_csv), params, args.start, args.end)
    weights_path = outdir / "weights_history.csv"
    weights_hist.to_csv(weights_path, index=False)
    print(f"Wrote weights history → {weights_path} ({len(weights_hist)} rows, {len(weights_hist['date'].unique()) if not weights_hist.empty else 0} rebalances)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
