"""
ModelB-v1.0 — Trend Strength / Risk-On–Off via Efficiency Ratio (ER)
--------------------------------------------------------------------

This module implements **Model B** from the locked spec:
- Input: same barometer Close series as Model A (BTC / BTC+ETH 50/50 / BTC+ETH+SOL 1/3), daily @ 00:00 UTC
- Signal: Efficiency Ratio ER_N with N=20 calendar days on log price
- States: Strong / Neutral / Weak with hysteresis and a 3-day dwell
- Outputs: continuous ER (R_t in [0,1]) and discrete strength_state

Default data source: CoinGecko "market_chart" (daily). You can also pass a local long CSV
(date, asset, close) via --price-csv and map coin_id->asset via --symbol-map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# -----------------------------------------------------------------------------
# Versioning & constants
# -----------------------------------------------------------------------------
MODEL_B_VERSION = "ModelB-v1.0"
USER_AGENT = "ModelB/1.0"

COIN_IDS = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "solana": "solana",
}

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class FetchConfig:
    """Network & API configuration for CoinGecko fetches.

    api_tier: 'demo' uses api.coingecko.com with header x-cg-demo-api-key;
              'pro'  uses pro-api.coingecko.com with header x-cg-pro-api-key.
    """
    vs_currency: str = "usd"
    days: int = 365
    interval: str = "daily"
    precision: int = 5
    timeout_sec: float = 20.0
    retries: int = 3
    backoff_factor: float = 0.3
    api_tier: str = "demo"  # 'demo' | 'pro'

    def base_url(self) -> str:
        return "https://pro-api.coingecko.com/api/v3" if self.api_tier == "pro" else "https://api.coingecko.com/api/v3"


@dataclass
class ModelBConfig:
    """Holds model parameters & thresholds (locked for v1.0)."""

    N: int = 20  # ER window (calendar days)

    # Hysteresis bands for Strong/Weak
    enter_strong: float = 0.30
    exit_strong: float = 0.25
    enter_weak: float = 0.20
    exit_weak: float = 0.25

    dwell_days: int = 3

    # Data requirements (comfort checks)
    min_history_days: int = 60  # enough to compute ER_20 reliably

    # State labels
    strong: str = "Strong"
    neutral: str = "Neutral"
    weak: str = "Weak"


# -----------------------------------------------------------------------------
# API key utilities
# -----------------------------------------------------------------------------
def load_api_key(flag_key: Optional[str] = None, file_path: Optional[str] = None) -> Optional[str]:
    """Resolve the CoinGecko API key.

    Order: explicit flag → explicit file → ./coingecko.key → env var COINGECKO_API_KEY.
    Never prints the secret; logs sources at DEBUG.
    """
    if flag_key and flag_key.strip():
        return flag_key.strip()

    candidates: List[Path] = []
    if file_path:
        candidates.append(Path(file_path))
    candidates.append(Path.cwd() / "coingecko.key")

    for p in candidates:
        try:
            if p.exists():
                content = p.read_text(encoding="utf-8").splitlines()
                for line in content:
                    tok = line.strip()
                    if tok and not tok.startswith("#"):
                        logging.debug("Loaded API key from file %s", p)
                        return tok
        except Exception as e:
            logging.debug("Could not read API key from %s: %s", p, e)

    env = os.getenv("COINGECKO_API_KEY")
    if env and env.strip():
        logging.debug("Loaded API key from environment variable COINGECKO_API_KEY")
        return env.strip()

    return None


# -----------------------------------------------------------------------------
# HTTP client (demo/pro) & barometer construction
# -----------------------------------------------------------------------------
class CoinGeckoClientB:
    def __init__(self, api_key: Optional[str], cfg: FetchConfig):
        self.api_key = api_key
        self.cfg = cfg
        self.session = requests.Session()
        retry = Retry(
            total=cfg.retries,
            read=cfg.retries,
            connect=cfg.retries,
            backoff_factor=cfg.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT})
        if api_key:
            header = "x-cg-pro-api-key" if cfg.api_tier == "pro" else "x-cg-demo-api-key"
            self.session.headers.update({header: api_key})

    def ping(self) -> None:
        url = f"{self.cfg.base_url()}/ping"
        resp = self.session.get(url, timeout=self.cfg.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"Ping failed {resp.status_code}: {resp.text[:200]} | url={url} | tier={self.cfg.api_tier}")

    def get_market_chart(self, coin_id: str) -> pd.Series:
        params = {
            "vs_currency": self.cfg.vs_currency,
            "days": str(self.cfg.days),
            "interval": self.cfg.interval,
            "precision": str(self.cfg.precision),
        }
        url = f"{self.cfg.base_url()}/coins/{coin_id}/market_chart"
        resp = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"CoinGecko error {resp.status_code}: {resp.text[:300]} | url={url}")
        data = resp.json()
        if "prices" not in data:
            raise ValueError("Unexpected response: 'prices' missing")
        arr = data["prices"]
        df = pd.DataFrame(arr, columns=["ts_ms", "close"])  # [timestamp_ms, price]
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("D").dt.tz_convert(None)
        daily = df.groupby("date", as_index=True)["close"].last().sort_index()
        daily.name = coin_id
        return daily


# -----------------------------------------------------------------------------
# Local CSV client (Binance/Kraken long-format -> Series)
# -----------------------------------------------------------------------------
@dataclass
class LocalCSVSourceConfig:
    csv_path: str                          # path to long CSV: date,asset,close
    symbol_map: Dict[str, str]             # coin_id -> CSV asset symbol (e.g., 'bitcoin'->'XBTEUR')

class LocalCSVClient:
    """
    Minimal drop-in replacement for CoinGeckoClientB.
    Expects CSV columns: date, asset, close (as produced by data/fetch_market_data.py).
    """

    def __init__(self, cfg: LocalCSVSourceConfig):
        self.cfg = cfg
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.cfg.csv_path)
            cols = {c.lower(): c for c in df.columns}
            for c in ("date", "asset", "close"):
                if c not in cols:
                    raise ValueError(f"Local CSV must have columns date, asset, close (missing '{c}')")
            df = df[[cols["date"], cols["asset"], cols["close"]]].copy()
            df.columns = ["date", "asset", "close"]
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
            df["asset"] = df["asset"].astype(str)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "asset", "close"])
            self._df = df.sort_values(["asset", "date"]).drop_duplicates(subset=["date", "asset"], keep="last")
        return self._df

    def get_market_chart(self, coin_id: str) -> pd.Series:
        sym = self.cfg.symbol_map.get(str(coin_id).lower())
        if not sym:
            raise KeyError(f"LocalCSVClient: no mapping for coin_id '{coin_id}'. Provide --symbol-map.")
        df = self._load()
        sub = df[df["asset"] == sym]
        if sub.empty:
            raise ValueError(f"LocalCSVClient: no rows for asset '{sym}' in {self.cfg.csv_path}")
        s = sub.set_index("date")["close"].sort_index().asfreq("D").ffill()
        s.name = sym
        return s


# -----------------------------------------------------------------------------
# Barometer construction (duck-typed client)
# -----------------------------------------------------------------------------
def build_barometer(client, kind: str = "bitcoin") -> pd.Series:
    """Return a normalized daily barometer price series."""
    k = kind.lower().strip()
    if k in ("bitcoin", "btc"):
        out = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC")
    elif k in ("ethereum", "eth"):
        out = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
    elif k in ("solana", "sol"):
        out = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
    elif k == "btc_eth_5050":
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC"); time.sleep(0.2)
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
        df = pd.concat([btc, eth], axis=1).dropna()
        comb = np.log(df).diff().mean(axis=1)
        out = 100.0 * np.exp(comb.cumsum()); out.name = "BTC_ETH_50_50"
    elif k in ("btc_eth_sol_333", "btc_eth_sol", "btc_eth_sol_ew"):
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC"); time.sleep(0.2)
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH"); time.sleep(0.2)
        sol = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
        df = pd.concat([btc, eth, sol], axis=1).dropna()
        comb = np.log(df).diff().mean(axis=1)
        out = 100.0 * np.exp(comb.cumsum()); out.name = "BTC_ETH_SOL_EQ"
    else:
        raise ValueError("Unknown barometer kind. Use 'bitcoin'|'ethereum'|'solana'|'btc_eth_5050'|'btc_eth_sol_333'.")

    out = out[~out.index.duplicated(keep="last")].sort_index().asfreq("D").ffill()
    return out


# -----------------------------------------------------------------------------
# Signal computation — Efficiency Ratio (ER)
# -----------------------------------------------------------------------------
def compute_er(series: pd.Series, N: int) -> pd.Series:
    """Compute Efficiency Ratio on log price:
       ER_N(t) = |log P_t - log P_{t-N}| / sum_{i=1..N} |log P_{t-i} - log P_{t-i-1}|"""
    s = series.dropna().astype(float)
    logp = np.log(s)
    num = (logp - logp.shift(N)).abs()
    den = logp.diff().abs().rolling(N).sum()
    er = (num / (den + 1e-12)).clip(lower=0.0, upper=1.0)
    er.name = f"ER_{N}"
    return er


# -----------------------------------------------------------------------------
# State machine — Strong / Neutral / Weak with hysteresis & dwell
# -----------------------------------------------------------------------------
def run_strength_state_machine(er: pd.Series, cfg: ModelBConfig) -> pd.DataFrame:
    idx = er.index
    out = pd.DataFrame(index=idx)
    out["er"] = er

    current = cfg.neutral
    dwell_left = 0

    states, dwells, reasons = [], [], []

    for _, val in er.items():
        reason = ""
        if pd.isna(val):
            states.append(current); dwells.append(dwell_left > 0); reasons.append("missing ER; hold")
            continue

        if dwell_left > 0:
            next_state = current
            if current != cfg.strong and val >= cfg.enter_strong:
                next_state = cfg.strong; dwell_left = cfg.dwell_days
                reason = f"Enter Strong (override dwell): ER={val:.3f} ≥ {cfg.enter_strong:.2f}"
            elif current != cfg.weak and val <= cfg.enter_weak:
                next_state = cfg.weak; dwell_left = cfg.dwell_days
                reason = f"Enter Weak (override dwell): ER={val:.3f} ≤ {cfg.enter_weak:.2f}"
            else:
                dwell_left -= 1; reason = f"dwell({dwell_left})"
            current = next_state
        else:
            if current == cfg.strong:
                if val <= cfg.exit_strong:
                    current = cfg.neutral; dwell_left = cfg.dwell_days
                    reason = f"Exit Strong→Neutral: ER={val:.3f} ≤ {cfg.exit_strong:.2f}"
            elif current == cfg.weak:
                if val >= cfg.exit_weak:
                    current = cfg.neutral; dwell_left = cfg.dwell_days
                    reason = f"Exit Weak→Neutral: ER={val:.3f} ≥ {cfg.exit_weak:.2f}"
            else:
                if val >= cfg.enter_strong:
                    current = cfg.strong; dwell_left = cfg.dwell_days
                    reason = f"Enter Strong: ER={val:.3f} ≥ {cfg.enter_strong:.2f}"
                elif val <= cfg.enter_weak:
                    current = cfg.weak; dwell_left = cfg.dwell_days
                    reason = f"Enter Weak: ER={val:.3f} ≤ {cfg.enter_weak:.2f}"

        states.append(current)
        dwells.append(dwell_left > 0)
        reasons.append(reason or "hold")

    out["strength_state"] = states
    out["in_dwell"] = dwells
    out["reason"] = reasons
    return out


# -----------------------------------------------------------------------------
# Orchestrator & helpers
# -----------------------------------------------------------------------------
@dataclass
class ModelBResult:
    version: str
    barometer_kind: str
    data: pd.DataFrame  # columns: er, strength_state, in_dwell, reason

    def latest_row(self) -> pd.Series:
        return self.data.iloc[-1]

    def save_csv(self, path: str) -> None:
        self.data.to_csv(path, index_label="date")


def parse_symbol_map_arg(arg: Optional[str]) -> Dict[str, str]:
    """Parse 'bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR' into a dict (lowercased keys)."""
    if not arg:
        return {}
    out: Dict[str, str] = {}
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def run_model_b(
    api_key: Optional[str] = None,
    barometer_kind: str = "bitcoin",
    fetch_cfg: Optional[FetchConfig] = None,
    model_cfg: Optional[ModelBConfig] = None,
    client: Optional[object] = None,   # LocalCSVClient or None (CoinGecko)
) -> ModelBResult:
    logging.info("Running Model B — %s", MODEL_B_VERSION)
    fetch_cfg = fetch_cfg or FetchConfig()
    model_cfg = model_cfg or ModelBConfig()

    # Choose client; ping only when using CoinGecko
    if client is None:
        cg = CoinGeckoClientB(api_key=api_key, cfg=fetch_cfg)
        try:
            cg.ping()
        except Exception as e:
            logging.warning("CoinGecko ping failed (continuing): %s", e)
        client = cg

    series = build_barometer(client, kind=barometer_kind)
    if len(series) < model_cfg.min_history_days:
        raise RuntimeError(f"Insufficient history: have {len(series)} days, need >= {model_cfg.min_history_days}.")

    er = compute_er(series, model_cfg.N)
    states = run_strength_state_machine(er, model_cfg)

    result = ModelBResult(version=MODEL_B_VERSION, barometer_kind=barometer_kind, data=states)
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModelB-v1.0 — Trend Strength via ER")

    # CoinGecko options (still supported)
    parser.add_argument("--api-key", type=str, default=None, help="CoinGecko API key (optional)")
    parser.add_argument("--api-key-file", type=str, default=None, help="Path to a .key file with ONLY the API key (default: ./coingecko.key)")
    parser.add_argument("--api-tier", type=str, default="demo", choices=["demo","pro"], help="API tier: 'demo' public or 'pro' paid")
    parser.add_argument("--days", type=int, default=365, help="CoinGecko days (demo cap 365; Pro can use 540+)")
    parser.add_argument("--vs", type=str, default="usd", help="CoinGecko quote currency")

    # Local CSV options
    parser.add_argument("--price-csv", type=str, default=None,
                        help="Path to local long-format CSV (date,asset,close) to avoid API caps.")
    parser.add_argument("--symbol-map", type=str, default=None,
                        help="Mapping for local CSV (coin_id -> asset), e.g. 'bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR' or USDT symbols.")

    # Barometer / output
    parser.add_argument(
        "--barometer-kind",
        type=str,
        default="bitcoin",
        choices=["bitcoin","btc","ethereum","eth","solana","sol","btc_eth_5050","btc_eth_sol_333","btc_eth_sol","btc_eth_sol_ew"],
        help="Barometer: 'bitcoin'|'ethereum'|'solana'|'btc_eth_5050'|'btc_eth_sol_333'",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV path. If omitted, a timestamped filename is used.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging verbosity")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    # Choose client
    client = None
    if args.price_csv:
        smap = parse_symbol_map_arg(args.symbol_map)
        if not smap:
            # try to infer mapping from the CSV
            _peek = pd.read_csv(args.price_csv, nrows=200)
            acols = {c.lower(): c for c in _peek.columns}
            if "asset" not in acols:
                raise SystemExit("Local CSV must have an 'asset' column.")
            assets = set(_peek[acols["asset"]].astype(str).unique())
            if "BTCUSDT" in assets:
                smap = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT"}
            elif "XBTEUR" in assets or "BTCEUR" in assets:
                btc = "XBTEUR" if "XBTEUR" in assets else "BTCEUR"
                smap = {"bitcoin": btc, "ethereum": "ETHEUR", "solana": "SOLEUR"}
            else:
                raise SystemExit("Please provide --symbol-map for the local CSV (could not infer).")
        client = LocalCSVClient(LocalCSVSourceConfig(csv_path=args.price_csv, symbol_map=smap))

    fetch_cfg = FetchConfig(vs_currency=args.vs, days=args.days, api_tier=args.api_tier)
    api_key = load_api_key(args.api_key, args.api_key_file)
    if api_key is None and not client:
        logging.warning("No CoinGecko API key and no --price-csv provided; proceeding unauthenticated CoinGecko (rate limits apply).")

    try:
        res = run_model_b(
            api_key=api_key,
            barometer_kind=args.barometer_kind,
            fetch_cfg=fetch_cfg,
            model_cfg=ModelBConfig(),
            client=client,
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_path = args.out or f"modelB_output_{args.barometer_kind}_{ts}.csv"
        res.save_csv(out_path)
        last = res.latest_row()
        logging.info("Latest: date=%s ER=%.3f state=%s dwell=%s",
                     last.name.date(), last["er"], last["strength_state"], last["in_dwell"])
        print(f"Saved daily log to {out_path} — version {res.version}")
    except Exception as e:
        logging.exception("Model B failed: %s", e)
        raise
