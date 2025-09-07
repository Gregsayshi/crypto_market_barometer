"""
Model-agnostic Validation & Backtesting Runner
==============================================

This script backtests any model that produces:
  1) Exposures:    date, exposure_scalar (0..1), regime     (CSV)  OR run_summary_*.json
  2) Weights:      date (rebalance-only), asset, weight     (CSV; long-only; sum ≈ 1 per date)
  3) Prices:       date, asset, close (EUR)                 (CSV; daily; short gaps ffilled)

It supports an auto-pick mode that grabs the latest ensemble run from
model_ensemble/outputs/<timestamp>/ (run_summary_*.json + modelC_diagnostics_*.csv / modelC_weights_*.csv).

Two run modes:
- Walk-forward (default): unbiased. Uses weights only from their dates onward.
- Retro-constant (--retro-constant): use the current snapshot across the entire window
  (look-ahead; useful for sanity checks & risk profiling).

Examples
--------
# Auto-pick latest ensemble; choose baselines by column name in --prices
python validation/run_backtest.py \
  --from-ensemble-latest \
  --prices data/prices_eur.csv \
  --bench XBTEUR --bench ETHEUR

# Manual inputs
python validation/run_backtest.py \
  --exposures model_ensemble/outputs/<ts>/run_summary_<ts>.json \
  --weights   model_ensemble/outputs/<ts>/modelC_diagnostics_EUR_<ts>.csv \
  --prices    data/prices_eur.csv \
  --bench XBTEUR

Notes
-----
- Everything is day-level: timestamps normalized to dates.
- If a single-day weights snapshot lands at/after the last price date, we shift/clamp
  so at least one next bar exists (or use --retro-constant to backfill across the window).
- Baselines: use --bench repeatedly to add any price column (e.g., XBTEUR, ETHEUR).
  Cash(EUR) and an equal-weight basket are always included.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- Utils / IO -------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_any_ts(ts_val: str | None) -> pd.Timestamp:
    """
    Parse ISO or ensemble-style 'YYYYMMDD_HHMMSSZ' into tz-aware UTC Timestamp.
    Fallback to now(UTC) if parsing fails.
    """
    if not ts_val:
        return pd.Timestamp.utcnow()
    try:
        return pd.to_datetime(ts_val, utc=True)
    except Exception:
        try:
            from datetime import datetime as _dt
            return pd.Timestamp(_dt.strptime(str(ts_val), "%Y%m%d_%H%M%SZ")).tz_localize("UTC")
        except Exception:
            return pd.Timestamp.utcnow()


# -------------------------------- Readers --------------------------------

def read_exposures(path: Path) -> pd.DataFrame:
    """
    Return DataFrame index=date (tz-naive), cols: exposure_scalar, regime.
    Accepts CSV or JSON (run_summary_*.json).
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        scalar = float(obj.get("exposure_scalar", 1.0))
        regime = str(obj.get("regime", "Unknown"))
        dt = _parse_any_ts(obj.get("timestamp_utc") or obj.get("timestamp"))
        idx = pd.DatetimeIndex([dt]).tz_convert(None).normalize()
        return pd.DataFrame({"exposure_scalar": [scalar], "regime": [regime]}, index=idx)

    # CSV
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("timestamp")
    if not date_col:
        raise ValueError("Exposures CSV must include a 'date' or 'timestamp' column")
    idx = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None).dt.normalize()
    out = pd.DataFrame(
        {
            "exposure_scalar": df[cols.get("exposure_scalar", "exposure_scalar")].astype(float).values,
            "regime": df[cols.get("regime", "regime")] if "regime" in cols else "",
        },
        index=idx,
    ).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def read_weights_history(path: Path) -> pd.DataFrame:
    """Return MultiIndex (date, asset) with 'weight' column; dates normalized to day."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for name in ("date", "asset", "weight"):
        if name not in cols:
            raise ValueError(f"Weights CSV missing '{name}' column")
    df = df[[cols["date"], cols["asset"], cols["weight"]]].copy()
    df.columns = ["date", "asset", "weight"]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    df["asset"] = df["asset"].astype(str)
    df["weight"] = df["weight"].astype(float)
    df = df.groupby(["date", "asset"], as_index=False)["weight"].sum()
    return df.set_index(["date", "asset"]).sort_index()


def read_prices(path: Path) -> pd.DataFrame:
    """Pivot to wide: index=date (D), columns=asset, values=close (float)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for name in ("date", "asset", "close"):
        if name not in cols:
            raise ValueError(f"Prices CSV missing '{name}' column")
    df = df[[cols["date"], cols["asset"], cols["close"]]].copy()
    df.columns = ["date", "asset", "close"]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["asset"] = df["asset"].astype(str)
    df["close"] = df["close"].astype(float)
    piv = df.pivot_table(index="date", columns="asset", values="close", aggfunc="last").sort_index()
    piv = piv.asfreq("D").ffill(limit=2)  # daily, fill short gaps
    return piv


# ---------------------------- Auto-pick latest run ----------------------------

def _find_latest_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    subs = [p for p in root.iterdir() if p.is_dir()]
    if not subs:
        return None
    subs.sort(key=lambda p: p.name, reverse=True)
    return subs[0]


def autopick_from_latest_ensemble(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Return (exposures_df, weights_mi, chosen_run_dir).
    weights_mi is a single-date snapshot built from diagnostics/weights CSV.
    """
    run_dir = _find_latest_subdir(root)
    if run_dir is None:
        raise FileNotFoundError(f"No subdirectories found in {root}")

    # exposures
    jsons = sorted(run_dir.glob("run_summary_*.json"))
    if not jsons:
        raise FileNotFoundError(f"No run_summary_*.json found in {run_dir}")
    exposures = read_exposures(jsons[-1])

    # weights (prefer diagnostics)
    diag = sorted(run_dir.glob("modelC_diagnostics_*.csv"))
    wts = sorted(run_dir.glob("modelC_weights_*.csv"))
    if not diag and not wts:
        raise FileNotFoundError(f"No modelC_diagnostics_* or modelC_weights_* found in {run_dir}")

    if diag:
        df = pd.read_csv(diag[-1])
        cols = {c.lower(): c for c in df.columns}
        asset_col = cols.get("pair_key") or cols.get("wsname") or cols.get("asset")
        if asset_col is None or "weight" not in cols:
            raise ValueError("Diagnostics CSV missing 'pair_key/wsname/asset' or 'weight'")
        d = df[[asset_col, cols["weight"]]].copy()
        d = d[d[cols["weight"]] > 0]
        ts_val = (
            df[cols.get("timestamp_utc", cols.get("timestamp", next(iter(df.index), None)))].iloc[0]
            if (cols.get("timestamp_utc") or cols.get("timestamp"))
            else run_dir.name
        )
        dt = _parse_any_ts(ts_val)
        date = dt.tz_convert(None).normalize()
        w_mi = d.rename(columns={asset_col: "asset", cols["weight"]: "weight"})
        w_mi["date"] = date
        w_mi = w_mi[["date", "asset", "weight"]].set_index(["date", "asset"]).sort_index()
    else:
        df = pd.read_csv(wts[-1])
        cols = {c.lower(): c for c in df.columns}
        asset_col = cols.get("pair_key") or cols.get("wsname") or cols.get("asset")
        if asset_col is None or "weight" not in cols:
            raise ValueError("Weights CSV missing 'asset/pair_key/wsname' or 'weight'")
        stamp = wts[-1].stem.split("_")[-1]
        dt = _parse_any_ts(stamp)
        date = dt.tz_convert(None).normalize()
        w_mi = df[[asset_col, cols["weight"]]].rename(columns={asset_col: "asset", cols["weight"]: "weight"})
        w_mi["date"] = date
        w_mi = w_mi[["date", "asset", "weight"]].set_index(["date", "asset"]).sort_index()

    return exposures, w_mi, run_dir


# --------------------------------- Engine ---------------------------------

@dataclass
class EngineConfig:
    rebalance: str = "W-MON"      # informational only
    signal_lag_days: int = 1
    tc_bps_one_way: float = 10.0
    retro_constant: bool = False
    start: Optional[str] = None
    end: Optional[str] = None


def align_calendar(
    weights_mi: pd.DataFrame,
    prices: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    retro_constant: bool = False,
) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    Walk-forward: start at max(first price, first weight)
    Retro-constant: start at first price (we'll backfill the snapshot backwards)
    """
    p_idx = prices.index.normalize()
    w_idx = weights_mi.index.get_level_values(0).normalize()

    d0 = p_idx.min() if retro_constant else max(p_idx.min(), w_idx.min())
    d1 = p_idx.max()

    if start:
        d0 = max(d0, pd.to_datetime(start))
    if end:
        d1 = min(d1, pd.to_datetime(end))

    cal = pd.date_range(start=d0, end=d1, freq="D")
    return cal, prices.loc[cal]


def expand_weights_daily(weights_mi: pd.DataFrame, calendar: pd.DatetimeIndex, retro_constant: bool = False) -> pd.DataFrame:
    by_date = weights_mi.reset_index().pivot(index="date", columns="asset", values="weight").sort_index()
    # normalize per rebalance day
    by_date = by_date.div(by_date.sum(axis=1).replace(0.0, np.nan), axis=0)

    by_date = by_date.reindex(calendar)
    by_date = by_date.ffill()
    if retro_constant:
        by_date = by_date.bfill()
    by_date = by_date.fillna(0.0)

    assets = sorted(by_date.columns.unique())
    W = pd.DataFrame(0.0, index=calendar, columns=assets)
    # safe assignment (indices now match)
    W.loc[calendar, assets] = by_date[assets].values
    return W


def apply_exposure_with_lag(
    exposure_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lag_days: int,
    retro_constant: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    E = exposure_df.sort_index().reindex(calendar).ffill()
    if retro_constant:
        E = E.bfill()  # single snapshot covers entire window
    scalar = E["exposure_scalar"].astype(float).shift(lag_days).fillna(0.0)
    regime = E["regime"].astype(str).shift(lag_days) if "regime" in E.columns else pd.Series("", index=calendar)
    return scalar, regime


def simulate(weights_mi: pd.DataFrame, prices: pd.DataFrame, exposure: pd.DataFrame, cfg: EngineConfig) -> Dict[str, pd.DataFrame]:
    # Normalize weights to day-level
    wm = weights_mi.reset_index().copy()
    wm["date"] = pd.to_datetime(wm["date"]).dt.normalize()
    weights_mi = wm.set_index(["date", "asset"]).sort_index()

    # Guard against snapshots at/after last price date
    pmax = prices.index.max().normalize()
    wmin = weights_mi.index.get_level_values(0).min()
    if wmin > pmax:
        print(f"[WARN] weights snapshot {wmin.date()} is after last price date {pmax.date()}, clamping to {pmax.date()}.")
        wm = weights_mi.reset_index()
        wm["date"] = pmax
        weights_mi = wm.set_index(["date", "asset"]).sort_index()
    elif wmin >= pmax and not cfg.retro_constant:
        new_d = pmax - pd.Timedelta(days=1)
        if new_d in prices.index:
            print(f"[WARN] weights snapshot {wmin.date()} has no next price day; shifting to {new_d.date()}.")
            wm = weights_mi.reset_index()
            wm["date"] = new_d
            weights_mi = wm.set_index(["date", "asset"]).sort_index()

    # Calendar
    calendar, P = align_calendar(
        weights_mi, prices, start=cfg.start, end=cfg.end, retro_constant=cfg.retro_constant
    )

    wm_clip = weights_mi.reset_index()
    wm_clip = wm_clip[
        (wm_clip["date"] >= calendar.min()) & (wm_clip["date"] <= calendar.max())
    ]
    weights_mi = wm_clip.set_index(["date", "asset"]).sort_index()


    R = P.pct_change().fillna(0.0)

    # Weights (daily) & exposure
    W = expand_weights_daily(weights_mi, calendar, retro_constant=cfg.retro_constant)
    exp_scalar, regime = apply_exposure_with_lag(exposure, calendar, cfg.signal_lag_days, retro_constant=cfg.retro_constant)

    # Drop assets with missing prices on a day; renormalize that day
    mask_ok = (~P.isna()).astype(float)
    W_eff = (W * mask_ok)
    row_sums = W_eff.sum(axis=1)
    W_eff = W_eff.div(row_sums.replace(0.0, np.nan), axis=0).fillna(0.0)

    # Apply gross exposure
    gross = exp_scalar.clip(lower=0.0)
    W_gross = W_eff.mul(gross, axis=0)

    # Portfolio returns (pre-TC)
    port_ret = (W_gross * R).sum(axis=1)

    # Turnover only on days weights change (rebalance rows forward-filled)
    W_unscaled = expand_weights_daily(weights_mi, calendar, retro_constant=cfg.retro_constant)  # pre-exposure
    W_shift = W_unscaled.shift(1).fillna(0.0)
    turnover = 0.5 * (W_unscaled - W_shift).abs().sum(axis=1)
    tc = (turnover > 1e-12).astype(float) * (turnover * (cfg.tc_bps_one_way / 1e4))

    ret_net = port_ret - tc

    return {
        "prices": P,
        "returns": R,
        "weights_daily": W_gross,
        "weights_unscaled": W_unscaled,
        "exposure_scalar": gross.rename("exposure_scalar").to_frame(),
        "regime": regime.rename("regime").to_frame(),
        "turnover": turnover.rename("turnover").to_frame(),
        "tc": tc.rename("tc").to_frame(),
        "portfolio": pd.DataFrame({"ret_pre_tc": port_ret, "ret_net": ret_net}),
    }


# ------------------------------ Metrics & Viz ------------------------------

def perf_metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {k: math.nan for k in ["CAGR","Ann.Vol","Sharpe","Sortino","MaxDD","Calmar","Wealth(x)","TUW","CVaR5%","CVaR1%"]}
    ann = 365.0
    eq = (1 + r).cumprod()
    wealth = float(eq.iloc[-1])
    years = max((r.index[-1] - r.index[0]).days / 365.0, 1e-9)
    cagr = wealth ** (1/years) - 1
    vol = float(r.std() * math.sqrt(ann))
    downside = r[r < 0]
    dvol = float(downside.std() * math.sqrt(ann)) if len(downside) > 1 else math.nan
    sharpe = float((r.mean() / (r.std() + 1e-12)) * math.sqrt(ann)) if r.std() > 0 else math.nan
    sortino = float((r.mean() / (dvol + 1e-12)) * math.sqrt(ann)) if (isinstance(dvol, float) and dvol > 0) else math.nan
    peak = eq.cummax(); dd = eq/peak - 1.0
    maxdd = float(dd.min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else math.nan
    q05 = float(np.nanpercentile(r, 5)); q01 = float(np.nanpercentile(r, 1))
    cvar05 = float(r[r <= q05].mean()) if np.isfinite(q05) else math.nan
    cvar01 = float(r[r <= q01].mean()) if np.isfinite(q01) else math.nan
    tuw = float((dd < 0).mean())
    return {"CAGR": cagr, "Ann.Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd, "Calmar": calmar,
            "Wealth(x)": wealth, "TUW": tuw, "CVaR5%": cvar05, "CVaR1%": cvar01}


def monthly_heatmap(ax, r: pd.Series):
    mret = (1 + r).resample("M").apply(lambda x: x.prod() - 1)
    if mret.empty:
        ax.text(0.5, 0.5, "No data", ha="center"); return
    df = mret.to_frame("ret"); df["year"] = df.index.year; df["month"] = df.index.strftime("%b")
    piv = df.pivot(index="year", columns="month", values="ret").reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title("Monthly Returns")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod(); peak = eq.cummax(); return eq/peak - 1.0


def plot_all(out: Dict[str, pd.DataFrame], baselines: Dict[str, pd.Series], outdir: Path) -> None:
    r_net = out["portfolio"]["ret_net"]

    eq = pd.DataFrame({"Strategy": (1 + r_net).cumprod()})
    for name, r in baselines.items():
        eq[name] = (1 + r.loc[eq.index].fillna(0.0)).cumprod()

    plt.figure(figsize=(11,5))
    for col in eq.columns: plt.plot(eq.index, eq[col], label=col)
    plt.title("Equity Curves"); plt.legend(); plt.tight_layout(); plt.savefig(outdir/"equity_curves.png"); plt.close()

    plt.figure(figsize=(11,4))
    plt.plot(r_net.index, drawdown_series(r_net), label="Strategy DD")
    for name, r in baselines.items(): plt.plot(r.index, drawdown_series(r), label=f"{name} DD")
    plt.title("Drawdowns"); plt.legend(); plt.tight_layout(); plt.savefig(outdir/"drawdowns.png"); plt.close()

    roll = r_net.rolling(90)
    vol90 = roll.std() * math.sqrt(365); sharpe90 = (roll.mean() / (roll.std() + 1e-12)) * math.sqrt(365)
    plt.figure(figsize=(11,3.5)); plt.plot(vol90.index, vol90); plt.title("Rolling Volatility (90d)"); plt.tight_layout(); plt.savefig(outdir/"rolling_vol.png"); plt.close()
    plt.figure(figsize=(11,3.5)); plt.plot(sharpe90.index, sharpe90); plt.title("Rolling Sharpe (90d)"); plt.tight_layout(); plt.savefig(outdir/"rolling_sharpe.png"); plt.close()

    fig, ax = plt.subplots(figsize=(9,5)); monthly_heatmap(ax, r_net); plt.tight_layout(); plt.savefig(outdir/"monthly_heatmap.png"); plt.close(fig)

    regime = out["regime"]["regime"].astype(str).reindex(eq.index).fillna("")
    reg_to_val = {"Uptrend": 1, "Range": 0, "Downtrend": -1}; vals = regime.map(lambda s: reg_to_val.get(str(s), 0))
    plt.figure(figsize=(11,5))
    plt.plot(eq.index, eq["Strategy"], label="Strategy")
    ymin, ymax = eq["Strategy"].min(), eq["Strategy"].max()
    for rname, v in [("Uptrend",1),("Range",0),("Downtrend",-1)]:
        mask = vals == v
        if mask.any(): plt.fill_between(eq.index, ymin, ymax, where=mask, alpha=0.08, step="pre", label=f"{rname} ribbon")
    plt.title("Strategy with Regime Ribbon"); plt.legend(); plt.tight_layout(); plt.savefig(outdir/"regime_ribbon.png"); plt.close()

    plt.figure(figsize=(11,3)); plt.plot(out["turnover"].index, out["turnover"]["turnover"]); plt.title("Turnover (per rebalance day)"); plt.tight_layout(); plt.savefig(outdir/"turnover.png"); plt.close()

    W = out["weights_daily"].copy()
    if not W.empty:
        top = W.mean().sort_values(ascending=False).head(10).index.tolist()
        W_top = W[top]
        plt.figure(figsize=(11,5)); bottom = np.zeros(len(W_top))
        for col in W_top.columns:
            plt.fill_between(W_top.index, bottom, bottom + W_top[col].values, step="pre"); bottom = bottom + W_top[col].values
        plt.title("Weights (Top 10 constituents)"); plt.tight_layout(); plt.savefig(outdir/"weights_stack.png"); plt.close()


# ------------------------------- Baselines --------------------------------

def build_baselines(prices: pd.DataFrame, bench_symbols: Optional[List[str]]) -> Dict[str, pd.Series]:
    baselines: Dict[str, pd.Series] = {}
    idx = prices.index
    baselines["Cash(EUR)"] = pd.Series(0.0, index=idx)
    if bench_symbols:
        for sym in bench_symbols:
            s = str(sym)
            if s in prices.columns:
                baselines[f"HODL:{s}"] = prices[s].pct_change().fillna(0.0)
            else:
                print(f"[WARN] baseline symbol '{s}' not found in prices; skipping.")
    eq_w = pd.DataFrame(1.0/len(prices.columns), index=prices.index, columns=prices.columns)
    baselines["EWP basket"] = (eq_w * prices.pct_change().fillna(0.0)).sum(axis=1)
    return baselines


# ----------------------------------- CLI -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Model-agnostic validation/backtesting runner")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--from-ensemble-latest", action="store_true", help="Auto-pick latest run from model_ensemble/outputs/")
    ap.add_argument("--ensemble-root", type=str, default="model_ensemble/outputs", help="Root folder for ensemble outputs")

    ap.add_argument("--exposures", help="CSV or JSON run_summary with date/exposure_scalar/regime")
    ap.add_argument("--weights", help="CSV with date, asset, weight (rebalance rows)")
    ap.add_argument("--prices", required=True, help="CSV with date, asset, close (EUR)")

    ap.add_argument("--rebalance", default="W-MON", help="Rebalance frequency label (for reference)")
    ap.add_argument("--signal-lag", type=int, default=1, help="Signal/weight lag in days")
    ap.add_argument("--tc-bps", type=float, default=10.0, help="One-way transaction cost in bps per turnover")

    ap.add_argument("--bench", action="append", default=None, help="Add any price column as a baseline; repeatable (e.g., --bench XBTEUR --bench ETHEUR)")

    # Window & mode
    ap.add_argument("--retro-constant", action="store_true",
                    help="Hold the current snapshot across the entire window (look-ahead; sanity checks).")
    ap.add_argument("--start", type=str, default=None, help="Backtest start date (YYYY-MM-DD), optional.")
    ap.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD), optional.")

    ap.add_argument("--outdir", type=str, default=None, help="Output directory root (default validation/runs/<ts>)")

    args = ap.parse_args(argv)

    # Inputs
    if args.from_ensemble_latest:
        exposures, weights_mi, chosen_run = autopick_from_latest_ensemble(Path(args.ensemble_root))
        print(f"Auto-picked latest ensemble run: {chosen_run}")
    else:
        if not args.exposures or not args.weights:
            raise SystemExit("Provide --exposures and --weights, or use --from-ensemble-latest")
        exposures = read_exposures(Path(args.exposures))
        weights_mi = read_weights_history(Path(args.weights))

    prices = read_prices(Path(args.prices))
    print("EXPOSURES:", exposures.index.min(), "→", exposures.index.max(), exposures.shape)
    print("WEIGHTS  :", weights_mi.index.get_level_values(0).min(), "→",
        weights_mi.index.get_level_values(0).max(), weights_mi.shape,
        "assets:", len(weights_mi.index.get_level_values(1).unique()))
    print("PRICES   :", prices.index.min(), "→", prices.index.max(), prices.shape,
        "assets:", len(prices.columns))    

    # Engine config
    ecfg = EngineConfig(
        rebalance=args.rebalance,
        signal_lag_days=args.signal_lag,
        tc_bps_one_way=args.tc_bps,
        retro_constant=args.retro_constant,
        start=args.start,
        end=args.end,
    )

    # Run simulation
    out = simulate(weights_mi, prices, exposures, ecfg)

    # Baselines & metrics
    baselines = build_baselines(prices, args.bench)
    strat = out["portfolio"]["ret_net"]

    rows = {"Strategy": perf_metrics(strat)}
    for name, r in baselines.items():
        rows[name] = perf_metrics(r)
    metrics_df = pd.DataFrame(rows).T

    # Write outputs
    ts = _ts()
    out_root = Path(args.outdir) if args.outdir else Path("validation") / "runs" / ts
    _ensure_dir(out_root)
    metrics_df.to_csv(out_root / "metrics_summary.csv")
    plot_all(out, baselines, out_root)

    manifest = {
        "timestamp_utc": ts,
        "mode": "retro-constant" if args.retro_constant else "walk-forward",
        "rebalance": args.rebalance,
        "signal_lag_days": args.signal_lag,
        "tc_bps_one_way": args.tc_bps,
        "window": {"start": args.start, "end": args.end},
        "inputs": {
            "mode": "auto" if args.from_ensemble_latest else "manual",
            "ensemble_root": str(Path(args.ensemble_root).resolve()) if args.from_ensemble_latest else None,
            "exposures": None if args.from_ensemble_latest else str(Path(args.exposures).resolve()),
            "weights": None if args.from_ensemble_latest else str(Path(args.weights).resolve()),
            "prices": str(Path(args.prices).resolve()),
        },
        "baselines_requested": args.bench,
        "metrics_summary_path": str((out_root / "metrics_summary.csv").resolve()),
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved metrics & charts to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
