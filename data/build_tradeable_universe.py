from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Params:
    window_days: int
    min_age_days: int
    min_coverage: float
    min_adv30_eur: float
    min_nonzero_vol: float
    max_close_vwap_dev: float
    max_zero_return: float
    vol_min: float
    vol_max: float
    exclude_pattern: Optional[str]
    whitelist: List[str]
    blacklist: List[str]
    as_of: Optional[str]


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def annualize_vol(daily_std: float) -> float:
    return float(daily_std * math.sqrt(365.0))


def compute_metrics(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Per-asset quality metrics on trailing window ending at as_of (or last date)."""
    if not {"date", "asset", "close"}.issubset(df.columns):
        raise SystemExit("Input CSV must have columns: date, asset, close[, vwap, volume]")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["asset", "date"])

    end_date = pd.to_datetime(params.as_of).tz_localize(None) if params.as_of else df["date"].max()
    start_date = end_date - timedelta(days=params.window_days)
    dfw = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    if "vwap" not in dfw.columns:
        dfw["vwap"] = np.nan
    if "volume" not in dfw.columns:
        dfw["volume"] = np.nan

    rows = []
    for asset, g in dfw.groupby("asset"):
        g = g.sort_values("date").copy()
        age_days = (g["date"].max() - g["date"].min()).days + 1
        obs = int(g["date"].nunique())
        coverage = obs / max(age_days, 1)

        vwap_filled = pd.to_numeric(g["vwap"], errors="coerce").fillna(g["close"])
        volu = pd.to_numeric(g["volume"], errors="coerce").fillna(0.0)
        notional = vwap_filled * volu
        adv30_series = notional.rolling(30, min_periods=10).mean()
        adv30_eur = float(adv30_series.iloc[-30:].mean()) if len(adv30_series) else float("nan")

        pct_nonzero_vol = float((volu > 0).mean())

        close = pd.to_numeric(g["close"], errors="coerce")
        rel_dev = (close - vwap_filled).abs() / close.replace(0, np.nan)
        med_abs_close_vwap_dev = float(rel_dev.median()) if len(rel_dev) else float("nan")

        ret = close.pct_change()
        pct_zero_return_days = float((ret.fillna(0.0).abs() < 1e-12).mean())
        ann_vol = float(annualize_vol(pd.to_numeric(ret, errors="coerce").dropna().std())) if ret.notna().sum() > 2 else float("nan")

        rows.append({
            "asset": asset,
            "first_date": str(g["date"].min().date()),
            "last_date": str(g["date"].max().date()),
            "age_days": int(age_days),
            "obs": int(obs),
            "coverage": float(coverage),
            "adv30_eur": float(adv30_eur) if not np.isnan(adv30_eur) else np.nan,
            "pct_nonzero_vol": float(pct_nonzero_vol),
            "med_abs_close_vwap_dev": float(med_abs_close_vwap_dev) if not np.isnan(med_abs_close_vwap_dev) else np.nan,
            "pct_zero_return_days": float(pct_zero_return_days),
            "ann_vol": float(ann_vol) if not np.isnan(ann_vol) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)


def apply_gates(met: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Apply objective gates; adds boolean columns and 'eligible'."""
    m = met.copy()
    m["gate_age"]       = m["age_days"]                >= params.min_age_days
    m["gate_cover"]     = m["coverage"]                >= params.min_coverage
    m["gate_liquidity"] = m["adv30_eur"]               >= params.min_adv30_eur
    m["gate_nzvol"]     = m["pct_nonzero_vol"]         >= params.min_nonzero_vol
    m["gate_dev"]       = m["med_abs_close_vwap_dev"]  <= params.max_close_vwap_dev
    m["gate_zero"]      = m["pct_zero_return_days"]    <= params.max_zero_return
    m["gate_volband"]   = (m["ann_vol"] >= params.vol_min) & (m["ann_vol"] <= params.vol_max)

    gates = ["gate_age","gate_cover","gate_liquidity","gate_nzvol","gate_dev","gate_zero","gate_volband"]
    m["eligible_rules"] = m[gates].all(axis=1)

    if params.exclude_pattern:
        rx = re.compile(params.exclude_pattern)
        m["excluded_by_pattern"] = m["asset"].apply(lambda a: bool(rx.search(a)))
    else:
        m["excluded_by_pattern"] = False

    m["eligible"] = m["eligible_rules"] & (~m["excluded_by_pattern"])

    if params.whitelist:
        m.loc[m["asset"].isin(params.whitelist), "eligible"] = True
    if params.blacklist:
        m.loc[m["asset"].isin(params.blacklist), "eligible"] = False
    return m


def add_quality_score(m: pd.DataFrame) -> pd.DataFrame:
    """Composite quality score; higher is better."""
    s = m.copy()

    def rdesc(series):
        x = series.rank(method="average", na_option="bottom", ascending=False)
        return x / x.max()

    def rasc(series):
        x = series.rank(method="average", na_option="bottom", ascending=True)
        return x / x.max()

    s["r_adv30"] = rdesc(s["adv30_eur"])
    s["r_cover"] = rdesc(s["coverage"])
    s["r_dev"]   = rasc(s["med_abs_close_vwap_dev"])
    s["r_zero"]  = rasc(s["pct_zero_return_days"])

    w_adv30, w_cov, w_dev, w_zero = 0.40, 0.25, 0.20, 0.15
    s["quality_score"] = (
        w_adv30 * s["r_adv30"] +
        w_cov   * s["r_cover"] +
        w_dev   * s["r_dev"] +
        w_zero  * s["r_zero"]
    )
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a tradeable universe from a long CSV using age/liquidity/sanity gates.")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (date,asset,close[,vwap,volume])")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--as-of", default=None, help="As-of date YYYY-MM-DD (default: last date in file)")
    ap.add_argument("--window-days", type=int, default=365, help="Trailing window for metrics (default: 365)")

    # Gates
    ap.add_argument("--min-age-days", type=int, default=180)
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--min-adv30-eur", type=float, default=50_000.0)
    ap.add_argument("--min-nonzero-vol", type=float, default=0.90)
    ap.add_argument("--max-close-vwap-dev", type=float, default=0.05)
    ap.add_argument("--max-zero-return", type=float, default=0.20)
    ap.add_argument("--vol-min", type=float, default=0.30, help="Annualized vol lower bound (e.g., 0.30 = 30%)")
    ap.add_argument("--vol-max", type=float, default=2.50, help="Annualized vol upper bound (e.g., 2.50 = 250%)")

    ap.add_argument("--exclude-pattern", type=str, default=r"^ZUSD|^USDT|.*3L$|.*3S$|^W[A-Z]+EUR",
                    help="Regex for assets to exclude (default: stablecoins/leveraged/wrapped)")
    ap.add_argument("--whitelist", nargs="*", default=["XBTEUR","ETHEUR","SOLEUR"], help="Always-include assets")
    ap.add_argument("--blacklist", nargs="*", default=[], help="Always-exclude assets")

    args = ap.parse_args(argv)

    params = Params(
        window_days=int(args.window_days),
        min_age_days=int(args.min_age_days),
        min_coverage=float(args.min_coverage),
        min_adv30_eur=float(args.min_adv30_eur),
        min_nonzero_vol=float(args.min_nonzero_vol),
        max_close_vwap_dev=float(args.max_close_vwap_dev),
        max_zero_return=float(args.max_zero_return),
        vol_min=float(args.vol_min),
        vol_max=float(args.vol_max),
        exclude_pattern=str(args.exclude_pattern) if args.exclude_pattern else None,
        whitelist=list(args.whitelist or []),
        blacklist=list(args.blacklist or []),
        as_of=args.as_of,
    )

    inp = Path(args.inp)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    met = compute_metrics(df, params)
    gated = apply_gates(met, params)
    scored = add_quality_score(gated).sort_values(
        ["eligible", "quality_score"], ascending=[False, False]
    ).reset_index(drop=True)

    metrics_path = outdir / "asset_metrics.csv"
    scored.to_csv(metrics_path, index=False)

    elig = scored[scored["eligible"]][["asset", "quality_score"]].copy()
    elig_path = outdir / "eligible_assets.csv"
    elig.to_csv(elig_path, index=False)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    filtered = df[df["asset"].isin(elig["asset"])].sort_values(["asset", "date"])
    uni_path = outdir / "universe_filtered.csv"
    filtered.to_csv(uni_path, index=False)

    manifest = {
        "ts_utc": ts_utc(),
        "input_csv": str(inp.resolve()),
        "as_of": params.as_of,
        "window_days": params.window_days,
        "gates": {
            "min_age_days": params.min_age_days,
            "min_coverage": params.min_coverage,
            "min_adv30_eur": params.min_adv30_eur,
            "min_nonzero_vol": params.min_nonzero_vol,
            "max_close_vwap_dev": params.max_close_vwap_dev,
            "max_zero_return": params.max_zero_return,
            "vol_min": params.vol_min,
            "vol_max": params.vol_max,
            "exclude_pattern": params.exclude_pattern,
        },
        "overrides": {"whitelist": params.whitelist, "blacklist": params.blacklist},
        "counts": {
            "assets_total": int(met["asset"].nunique()),
            "assets_eligible": int(elig["asset"].nunique()),
        },
        "outputs": {
            "asset_metrics_csv": str(metrics_path.resolve()),
            "eligible_assets_csv": str(elig_path.resolve()),
            "universe_filtered_csv": str(uni_path.resolve()),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Wrote:\n - {metrics_path}\n - {elig_path}\n - {uni_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())