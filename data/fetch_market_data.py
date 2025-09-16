r"""
Fetch daily crypto prices from Binance or Kraken and write long CSVs
===================================================================

Output schema (long, backtester-ready):
    date (UTC, YYYY-MM-DD), asset (string), close (float), source (binance|kraken)

Examples (Windows PowerShell one-liners)
----------------------------------------
# Binance: 1000-day window limit per request is handled via pagination
python data\fetch_market_data.py binance --symbols BTCEUR ETHEUR SOLEUR --since 2022-01-01 --out data\binance\binance_daily.csv

# Kraken: altname/pair_key (e.g., XBTEUR, ADAEUR, SOLEUR) with 'since' paging
python data\fetch_market_data.py kraken  --symbols XBTEUR ETHEUR SOLEUR --since 2022-01-01 --out data\kraken\kraken_daily.csv

# Append more assets to the same file safely (de-duped)
python data\fetch_market_data.py binance --symbols AD AEUR --since 2023-01-01 --out data\binance\binance_daily.csv

Notes
-----
- Binance market-data endpoints are public; no key needed. We page using startTime/endTime with interval=1d and limit=1000.
- Kraken OHLC endpoint is public; we pass 'since' (unix seconds). We normalize to daily UTC dates.
- Asset names are written exactly as you request (e.g., 'BTCEUR', 'XBTEUR') so they match your weights (pair_key) directly.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests


# --------------------------- shared helpers ---------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _to_utc_date(ts_ms: int) -> pd.Timestamp:
    """Convert milliseconds since epoch to tz-naive UTC date (normalized)."""
    return pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None).normalize()

def _to_utc_date_sec(ts_sec: int) -> pd.Timestamp:
    return pd.to_datetime(ts_sec, unit="s", utc=True).tz_convert(None).normalize()

def _daterange_utc(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[int, int]:
    """Return UTC ms for API params (inclusive window)."""
    s = pd.Timestamp(start).tz_localize("UTC")
    e = pd.Timestamp(end + pd.Timedelta(days=1)).tz_localize("UTC") - pd.Timedelta(milliseconds=1)
    return int(s.timestamp() * 1000), int(e.timestamp() * 1000)


# =============================== BINANCE ===============================

BINANCE_API = "https://api.binance.com"

@dataclass
class BinanceParams:
    symbols: List[str]
    since: pd.Timestamp
    until: pd.Timestamp
    limit: int = 1000      # Binance max per request for klines
    interval: str = "1d"
    sleep_sec: float = 0.15

def fetch_binance_daily(p: BinanceParams) -> pd.DataFrame:
    """
    Page /api/v3/klines for each symbol across [since, until], 1d candles.
    Returns long DataFrame: date, asset, close, source='binance'
    """
    out_rows = []
    s_ms, u_ms = _daterange_utc(p.since, p.until)

    for sym in p.symbols:
        start = s_ms
        while start <= u_ms:
            url = f"{BINANCE_API}/api/v3/klines"
            params = {
                "symbol": sym,
                "interval": p.interval,
                "limit": p.limit,
                "startTime": start,
                "endTime": u_ms,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not data:
                break

            # Each entry: [open time, open, high, low, close, volume, close time, ...]
            for k in data:
                open_time_ms = int(k[0])
                close_price = float(k[4])
                d = _to_utc_date(open_time_ms)
                out_rows.append((d, sym, close_price, "binance"))

            # Advance start to the next ms after the last open_time returned
            last_open = int(data[-1][0])
            next_start = last_open + 24 * 60 * 60 * 1000  # next day
            if next_start <= start:  # safety
                next_start = start + 24 * 60 * 60 * 1000
            start = next_start
            time.sleep(p.sleep_sec)

    if not out_rows:
        return pd.DataFrame(columns=["date", "asset", "close", "source"])
    df = pd.DataFrame(out_rows, columns=["date", "asset", "close", "source"])
    df = df.drop_duplicates(subset=["date", "asset"]).sort_values(["asset", "date"])
    return df


# =============================== KRAKEN ================================

KRAKEN_API = "https://api.kraken.com/0/public"

@dataclass
class KrakenParams:
    symbols: List[str]          # Kraken altname/pair_key (e.g., XBTEUR, ADAEUR)
    since: pd.Timestamp
    until: pd.Timestamp
    interval: int = 1440        # minutes; 1440 = 1 day
    sleep_sec: float = 0.25

def fetch_kraken_daily(p: KrakenParams) -> pd.DataFrame:
    """
    Use /0/public/OHLC with 'since' paging. Kraken returns 'last' token (in seconds).
    Returns long DataFrame: date, asset, close, source='kraken'
    """
    out_rows = []
    start_sec = int(pd.Timestamp(p.since).tz_localize("UTC").timestamp())
    until_sec = int(pd.Timestamp(p.until + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())  # exclusive

    for altname in p.symbols:
        since = start_sec
        last_seen = -1
        while since < until_sec:
            url = f"{KRAKEN_API}/OHLC"
            params = {"pair": altname, "interval": p.interval, "since": since}
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            if j.get("error"):
                raise RuntimeError(f"Kraken error for {altname}: {j['error']}")
            res = j.get("result", {})
            last = int(res.get("last", since))
            # Result key may be altname or internal pair key. Grab the first list-like.
            series = None
            for k, v in res.items():
                if k != "last" and isinstance(v, list):
                    series = v
                    break
            if not series:
                break

            for row in series:
                t = int(row[0])  # seconds
                close = float(row[4])
                d = _to_utc_date_sec(t)
                if d <= pd.Timestamp(p.until).normalize():
                    out_rows.append((d, altname, close, "kraken"))

            if last == last_seen:
                break
            last_seen = last
            since = last
            time.sleep(p.sleep_sec)

    if not out_rows:
        return pd.DataFrame(columns=["date", "asset", "close", "source"])
    df = pd.DataFrame(out_rows, columns=["date", "asset", "close", "source"])
    df = df.drop_duplicates(subset=["date", "asset"]).sort_values(["asset", "date"])
    return df


# ================================ CLI =================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch daily crypto prices (Binance/Kraken) to long CSV")
    sub = ap.add_subparsers(dest="exchange", required=True)

    # Binance
    ap_b = sub.add_parser("binance", help="Fetch from Binance /api/v3/klines (interval=1d, limit=1000)")
    ap_b.add_argument("--symbols", nargs="+", required=True, help="e.g., BTCEUR ETHEUR SOLEUR (Binance symbols)")
    ap_b.add_argument("--since", required=True, help="YYYY-MM-DD (UTC)")
    ap_b.add_argument("--until", default=None, help="YYYY-MM-DD (UTC), default=today")
    ap_b.add_argument("--out", required=True, help="Output CSV path (will append & de-dup)")
    ap_b.add_argument("--sleep-sec", type=float, default=0.15, help="Throttle between API calls (sec)")

    # Kraken
    ap_k = sub.add_parser("kraken", help="Fetch from Kraken /0/public/OHLC (interval=1440)")
    ap_k.add_argument("--symbols", nargs="+", required=True, help="e.g., XBTEUR ETHEUR ADAEUR (Kraken altnames)")
    ap_k.add_argument("--since", required=True, help="YYYY-MM-DD (UTC)")
    ap_k.add_argument("--until", default=None, help="YYYY-MM-DD (UTC), default=today")
    ap_k.add_argument("--out", required=True, help="Output CSV path (will append & de-dup)")
    ap_k.add_argument("--sleep-sec", type=float, default=0.25, help="Throttle between API calls (sec)")

    return ap.parse_args()

def main() -> int:
    args = parse_args()
    until = args.until or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    since_dt = pd.to_datetime(args.since, utc=True).tz_convert(None)
    until_dt = pd.to_datetime(until, utc=True).tz_convert(None)

    out_path = Path(args.out)
    _ensure_parent(out_path)

    if args.exchange == "binance":
        df_new = fetch_binance_daily(
            BinanceParams(
                symbols=[str(s).upper() for s in args.symbols],
                since=since_dt,
                until=until_dt,
                sleep_sec=float(args.sleep_sec),
            )
        )
    else:
        df_new = fetch_kraken_daily(
            KrakenParams(
                symbols=[str(s).upper() for s in args.symbols],
                since=since_dt,
                until=until_dt,
                sleep_sec=float(args.sleep_sec),
            )
        )

    # Append/de-dup existing
    if out_path.exists():
        old = pd.read_csv(out_path, parse_dates=["date"])
        old["date"] = pd.to_datetime(old["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        df_all = pd.concat([old, df_new], ignore_index=True)
    else:
        df_all = df_new

    if not df_all.empty:
        df_all["date"] = pd.to_datetime(df_all["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        df_all = df_all.dropna(subset=["date", "asset", "close"]).drop_duplicates(subset=["date", "asset", "source"])
        df_all = df_all.sort_values(["asset", "date"])

    df_all.to_csv(out_path, index=False)
    print(f"Saved {len(df_new)} new rows; total {len(df_all)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
