#!/usr/bin/env python3
"""
Build a long-format EUR price panel from Kraken for Model C (offline mode).

Outputs a CSV with columns: date, asset, close, vwap, volume
- `asset` uses Kraken *altname* (e.g., ADAEUR, XBTEUR), which matches Model C's `--csv-key altname`.
- Designed to be idempotent (overwrites output unless you change path).
- Polite retries + small sleep between OHLC calls.

Examples (PowerShell one-liners):
  # Live pair discovery + fetch since a date
  python data\build_kraken_universe.py --list-live --since 2022-01-01 --max-pairs 120 --out data\kraken\kraken_eur_universe.csv

  # Use an existing pairs CSV from kraken_list_eur_pairs.py
  python data\build_kraken_universe.py --pairs-csv data\kraken\kraken_eur_pairs_20250907_121000Z.csv --since 2022-01-01 --out data\kraken\kraken_eur_universe.csv

Then run Model C offline:
  python model_c_momentum\model_c_v1.py --offline --price-csv data\kraken\kraken_eur_universe.csv --csv-key altname --N 10 --cap 0.25 --floor-eur 25 --notional-eur 1000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

API_BASE = "https://api.kraken.com/0"
USER_AGENT = "kraken-universe-builder/1.0"

# ------------------------------ HTTP util ------------------------------

def _session_with_retries(retries: int = 3, backoff: float = 0.25, timeout: float = 30.0) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    s.request_timeout = timeout  # for clarity; we pass timeout explicitly too
    return s

# ------------------------------ Pair discovery ------------------------------

def fetch_asset_pairs(session: requests.Session) -> Dict[str, dict]:
    url = f"{API_BASE}/public/AssetPairs"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or "error" not in payload or "result" not in payload:
        raise RuntimeError("Unexpected response shape from Kraken AssetPairs")
    if payload["error"]:
        raise RuntimeError(f"Kraken API error(s): {payload['error']}")
    result = payload["result"]
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected 'result' type from Kraken AssetPairs")
    return result

def _is_eur_quote(meta: dict) -> bool:
    quote = str(meta.get("quote", ""))
    wsname = str(meta.get("wsname", ""))
    altname = str(meta.get("altname", ""))
    if quote.upper() == "ZEUR":
        return True
    if wsname.upper().endswith("/EUR"):
        return True
    if altname.upper().endswith("EUR"):
        return True
    return False

def extract_eur_altnames(session: requests.Session, include_inactive: bool, max_pairs: Optional[int]) -> List[str]:
    result = fetch_asset_pairs(session)
    rows: List[Tuple[str, str, Optional[str]]] = []
    for key, meta in result.items():
        if not _is_eur_quote(meta):
            continue
        status = meta.get("status")
        if (not include_inactive) and (status is not None) and (str(status).lower() != "online"):
            continue
        alt = meta.get("altname")
        wsname = meta.get("wsname")
        if not alt:  # very rare, but guard
            alt = key
        rows.append((alt, key, wsname))
    # sort by wsname if present, else alt, for determinism
    rows.sort(key=lambda t: (t[2] or t[0]))
    alts = [t[0] for t in rows]
    if max_pairs and len(alts) > max_pairs:
        alts = alts[:max_pairs]
    return alts

def read_pairs_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # prefer altname if present; otherwise fall back to pair_key; last-resort wsname->alt-like
    if "altname" in cols:
        alts = df[cols["altname"]].dropna().astype(str).tolist()
    elif "pair_key" in cols:
        alts = df[cols["pair_key"]].dropna().astype(str).tolist()
    else:
        # try to convert wsname 'XXX/EUR' → 'XXXEUR'
        if "wsname" not in cols:
            raise ValueError("Pairs CSV must include one of: altname, pair_key, wsname")
        tmp = df[cols["wsname"]].dropna().astype(str).tolist()
        alts = [t.replace("/", "") for t in tmp]
    # unique, keep order
    seen = set(); out = []
    for a in alts:
        if a not in seen:
            seen.add(a); out.append(a)
    return out

# ------------------------------ OHLC fetch ------------------------------

def fetch_ohlc_daily(session: requests.Session, altname: str, since_unix: int, end_unix: Optional[int] = None) -> pd.DataFrame:
    """Return daily OHLC as DataFrame indexed by date (naive), cols: close, vwap, volume."""
    params = {"pair": altname, "interval": 1440, "since": since_unix}
    url = f"{API_BASE}/public/OHLC"
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC error for {altname}: {payload['error']}")
    result = payload.get("result", {})
    rows = None
    for k, v in result.items():
        if k == "last":
            continue
        if isinstance(v, list):
            rows = v
            break
    if not rows:
        return pd.DataFrame(columns=["date", "close", "vwap", "volume"]).set_index("date")

    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","trades"])
    # types
    for c, typ in [("time", int), ("open", float), ("high", float), ("low", float), ("close", float), ("vwap", float), ("volume", float)]:
        df[c] = df[c].astype(typ)

    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor("D").dt.tz_convert(None)
    if end_unix is not None:
        end_dt = pd.to_datetime(end_unix, unit="s", utc=True).tz_convert(None).floor("D")
        df = df[df["date"] <= end_dt]

    out = df.set_index("date")[["close","vwap","volume"]].sort_index()
    # Daily continuity (ffill close/vwap only; volume left as-is)
    out = out.asfreq("D")
    out["close"] = out["close"].ffill()
    out["vwap"] = out["vwap"].ffill()
    return out

# ------------------------------ Main build ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build Kraken EUR universe price panel for Model C (offline).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-live", action="store_true", help="Discover EUR altnames live from Kraken")
    g.add_argument("--pairs-csv", type=str, help="Existing pairs CSV from kraken_list_eur_pairs.py")

    ap.add_argument("--include-inactive", action="store_true", help="Include non-online pairs (if status present)")
    ap.add_argument("--max-pairs", type=int, default=120, help="Cap number of pairs to fetch (API economy)")

    ap.add_argument("--since", type=str, default=None, help="Start date YYYY-MM-DD (mutually exclusive with --since-days)")
    ap.add_argument("--since-days", type=int, default=None, help="Start N days ago (mutually exclusive with --since)")
    ap.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD (default: today)")

    ap.add_argument("--sleep-sec", type=float, default=0.25, help="Sleep between pair requests")
    ap.add_argument("--out", type=str, default="data/kraken/kraken_eur_universe.csv", help="Output CSV (overwritten)")

    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = _session_with_retries()

    # Determine universe (altnames)
    if args.list_live:
        alts = extract_eur_altnames(session, include_inactive=args.include_inactive, max_pairs=args.max_pairs)
    else:
        alts = read_pairs_csv(Path(args.pairs_csv))
        if args.max_pairs and len(alts) > args.max_pairs:
            alts = alts[:args.max_pairs]

    if not alts:
        print("No EUR altnames found; nothing to fetch.")
        return 1

    # Time window
    if (args.since is None) == (args.since_days is None):
        raise SystemExit("Provide exactly one of --since or --since-days.")
    if args.since:
        start_dt = pd.Timestamp(args.since).tz_localize("UTC")
    else:
        start_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=int(args.since_days))
    end_dt = pd.Timestamp(args.end).tz_localize("UTC") if args.end else pd.Timestamp.utcnow()

    since_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    # Fetch loop
    all_rows: List[pd.DataFrame] = []
    for i, alt in enumerate(alts, 1):
        try:
            print(f"[{i}/{len(alts)}] Fetching {alt} ...")
            df = fetch_ohlc_daily(session, alt, since_unix, end_unix)
            if df.empty:
                continue
            tmp = df.copy()
            tmp["asset"] = alt
            tmp = tmp.reset_index()  # date, close, vwap, volume, asset
            all_rows.append(tmp[["date","asset","close","vwap","volume"]])
        except Exception as e:
            print(f"  -> skip {alt}: {e}")
        finally:
            import time as _t; _t.sleep(args.sleep_sec)

    if not all_rows:
        print("No OHLC data fetched; aborting.")
        return 1

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    # Deduplicate in case of overlaps, keep last
    out = out.sort_values(["asset","date"]).drop_duplicates(subset=["asset","date"], keep="last")

    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows across {out['asset'].nunique()} assets → {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
