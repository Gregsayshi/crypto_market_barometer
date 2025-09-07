"""
Build a EUR price panel from Kraken for validation/backtests
===========================================================

This script fetches **daily OHLC** (interval=1440) for a set of EUR-quoted pairs
from Kraken and writes a long CSV with columns: **date, asset, close**.

It is **model-agnostic**: it can auto-pick assets from the latest ensemble run
or read them from a weights/diagnostics CSV. The output is compatible with
`validation/run_backtest.py`'s `--prices` input.

Examples
--------
# 1) Auto-pick the latest ensemble, use its selected names (diagnostics), 400 days history
python validation/build_prices_kraken.py \
  --from-ensemble-latest \
  --ensemble-root model_ensemble/outputs \
  --since-days 400 \
  --out data/prices_eur.csv \
  --asset-col pair_key \
  --include XBTEUR  # add BTC/EUR baseline

# 2) From an explicit weights/diagnostics CSV
python validation/build_prices_kraken.py \
  --weights model_ensemble/outputs/<ts>/modelC_diagnostics_EUR_<ts>.csv \
  --asset-col pair_key \
  --since-days 400 \
  --out data/prices_eur.csv \
  --include XBTEUR --include ETHEUR  # add BTC/EUR and ETH/EUR baselines

Notes
-----
- Kraken OHLC endpoint expects the **altname** (e.g., ADAEUR, XBTEUR).
- Your weights may store either **pair_key/altname** (good) or **wsname** (e.g., ADA/EUR).
  We call `AssetPairs` to map wsname↔altname when needed.
- The **asset strings in the output** match the ones in your weights file so the
  backtester can join without extra mapping.
- Be polite to the API: throttle with `--sleep-sec`.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

KRAKEN_API = "https://api.kraken.com/0/public"

# ------------------------------- helpers -------------------------------

def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def get_asset_pairs() -> pd.DataFrame:
    url = f"{KRAKEN_API}/AssetPairs"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {data['error']}")
    items = data["result"]
    rows = []
    for key, obj in items.items():
        altname = obj.get("altname")  # e.g., ADAEUR
        wsname = obj.get("wsname")    # e.g., ADA/EUR
        quote = obj.get("quote")
        status = obj.get("status", "online")
        rows.append({"pair_key": key, "altname": altname, "wsname": wsname, "quote": quote, "status": status})
    return pd.DataFrame(rows)


def map_assets_to_altname(names: List[str], asset_col: str, pairs: pd.DataFrame) -> Dict[str, str]:
    """Return mapping from the original asset string to Kraken altname for OHLC calls."""
    pairs = pairs.copy()
    # Build lookups
    by_alt = {str(a): str(a) for a in pairs["altname"].dropna().astype(str)}
    by_ws  = {str(w): str(a) for w, a in pairs[["wsname","altname"]].dropna().astype(str).values}
    by_key = {str(k): str(a) for k, a in pairs[["pair_key","altname"]].dropna().astype(str).values}

    out: Dict[str, str] = {}
    for n in names:
        s = str(n)
        alt = None
        if asset_col.lower() == "wsname":
            alt = by_ws.get(s)
        elif asset_col.lower() in ("pair_key","altname","asset"):
            alt = by_alt.get(s) or by_key.get(s) or by_ws.get(s)
        else:
            # heuristic: try all maps
            alt = by_alt.get(s) or by_key.get(s) or by_ws.get(s)
        if not alt:
            raise KeyError(f"Cannot map asset '{s}' to Kraken altname (pair)")
        out[s] = alt
    return out


def fetch_ohlc_altname(altname: str, start_unix: int, interval: int = 1440) -> pd.DataFrame:
    """Fetch daily OHLC from Kraken for a single **altname** (e.g., 'XBTEUR').

    Kraken returns a dict whose key is often an **internal pair key** (e.g., 'XXBTZEUR'),
    not necessarily the provided altname. We therefore scan the result and take the
    **first list value** (the OHLC array) and ignore the exact key name. Pagination is
    handled via the 'last' cursor.
    """
    url = f"{KRAKEN_API}/OHLC"
    params = {"pair": altname, "interval": interval, "since": start_unix}
    frames: List[pd.DataFrame] = []
    last = start_unix
    while True:
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken OHLC error for {altname}: {data['error']}")
        res = data.get("result", {})
        # The result key is the 'altname' (or sometimes internal key); find first list
        arr = None
        for k, v in res.items():
            if isinstance(v, list):
                arr = v; break
        if arr is None:
            break
        df = pd.DataFrame(arr, columns=["time","open","high","low","close","vwap","volume","count"])
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True).dt.tz_convert(None)
        df["close"] = df["close"].astype(float)
        frames.append(df[["time","close"]])
        # Pagination via 'last'
        last_new = res.get("last")
        if not last_new or last_new == last:
            break
        last = int(last_new)
        params["since"] = last
        # Safety: stop if we reached today
        if last > int(datetime.now(timezone.utc).timestamp()):
            break
    if not frames:
        return pd.DataFrame(columns=["time","close"]).astype({"time":"datetime64[ns]","close":"float"})
    out = pd.concat(frames, axis=0).drop_duplicates("time").sort_values("time")
    # daily freq & ffill short gaps
    out = out.set_index("time").asfreq("D").ffill(limit=2).reset_index()
    return out

# ------------------------------- sources -------------------------------

def from_latest_ensemble(ensemble_root: Path, asset_col: str) -> Tuple[List[str], pd.DataFrame]:
    # Find latest timestamped subdir
    subs = [p for p in ensemble_root.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No subdirectories in {ensemble_root}")
    subs.sort(key=lambda p: p.name, reverse=True)
    latest = subs[0]
    # Prefer diagnostics for asset list
    diag = sorted(latest.glob("modelC_diagnostics_*.csv"))
    wts = sorted(latest.glob("modelC_weights_*.csv"))
    if not diag and not wts:
        raise FileNotFoundError(f"No modelC diagnostics/weights files in {latest}")
    if diag:
        df = pd.read_csv(diag[-1])
    else:
        df = pd.read_csv(wts[-1])
    cols = {c.lower(): c for c in df.columns}
    col = cols.get(asset_col.lower()) if asset_col.lower() in cols else None
    if not col:
        # try fallbacks
        col = cols.get("pair_key") or cols.get("wsname") or cols.get("asset")
        if not col:
            raise KeyError(f"Could not find asset column in diagnostics/weights ({asset_col} / pair_key / wsname / asset)")
    # If diagnostics, filter to selected weights only
    if "weight" in cols and col != "asset":
        sel = df[df[cols["weight"]] > 0]
    else:
        sel = df
    names = sorted(set(sel[col].astype(str).tolist()))
    return names, df

# ----------------------------------- CLI -----------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Build a EUR price panel from Kraken OHLC for given assets (with optional baseline includes)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-ensemble-latest", action="store_true", help="Use latest run under --ensemble-root to pick assets")
    g.add_argument("--weights", type=str, help="Weights or diagnostics CSV to derive assets from")

    ap.add_argument("--ensemble-root", type=str, default="model_ensemble/outputs", help="Root folder of ensemble outputs")
    ap.add_argument("--asset-col", type=str, default="pair_key", help="Which column in weights/diagnostics represents the asset (pair_key|wsname|asset)")

    d = ap.add_mutually_exclusive_group(required=False)
    d.add_argument("--since-days", type=int, default=400, help="History window (days back from today UTC)")
    d.add_argument("--start", type=str, help="Start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (UTC), default today")

    ap.add_argument("--sleep-sec", type=float, default=0.25, help="Delay between OHLC calls to be polite")
    ap.add_argument("--out", type=str, default="data/prices_eur.csv", help="Output CSV path")

    # Extra assets to include explicitly (repeatable). Examples: --include XBTEUR --include ETHEUR
    ap.add_argument("--include", action="append", default=None, help="Extra assets to include (pair_key/wsname/altname). Can repeat.")

    args = ap.parse_args()

    # Determine asset names
    if args.from_ensemble_latest:
        names, _df = from_latest_ensemble(Path(args.ensemble_root), args.asset_col)
    else:
        df = pd.read_csv(args.weights)
        col = args.asset_col if args.asset_col in df.columns else None
        if not col:
            # fallback lookup
            for c in ("pair_key","wsname","asset"):
                if c in df.columns:
                    col = c; break
        if not col:
            raise KeyError(f"Asset column not found in {args.weights}")
        # If the file has weights, prefer positive-weight rows
        if "weight" in df.columns:
            df = df[df["weight"] > 0]
        names = sorted(set(df[col].astype(str).tolist()))

    # Merge in explicit includes (may be altname like XBTEUR, pair_key like XXBTZEUR, or wsname like XBT/EUR)
    if args.include:
        extra: List[str] = []
        for inc in args.include:
            for tok in str(inc).split(","):
                tok = tok.strip()
                if tok:
                    extra.append(tok)
        names = sorted(set(list(names) + extra))

    # Map names to Kraken altname for OHLC
    pairs = get_asset_pairs()
    a2alt = map_assets_to_altname(names, args.asset_col, pairs)

    # Time range
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    # Normalize to tz-naive UTC for pandas comparisons (df['time'] is tz-naive UTC)
    start_naive = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
    end_naive = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
    start_unix = int(start_dt.timestamp())

    # Fetch and assemble long panel
    rows = []
    for asset_out, altname in a2alt.items():
        print(f"Fetching {asset_out} (Kraken altname={altname})...")
        try:
            df = fetch_ohlc_altname(altname, start_unix=start_unix, interval=1440)
        except Exception as e:
            print(f"  ERROR fetching {asset_out}: {e}")
            time.sleep(args.sleep_sec)
            continue
        if df.empty:
            print(f"  No data for {asset_out}")
            time.sleep(args.sleep_sec)
            continue
        df = df[(df["time"] >= start_naive) & (df["time"] <= end_naive)]
        for t, c in df[["time","close"]].itertuples(index=False):
            rows.append((t, asset_out, float(c)))
        time.sleep(args.sleep_sec)

    if not rows:
        raise SystemExit("No price rows fetched; aborting")

    out = pd.DataFrame(rows, columns=["date","asset","close"]).sort_values(["date","asset"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved prices to {out_path} [{len(out)} rows across {len(a2alt)} assets]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
