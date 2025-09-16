"""
ModelA-v1.0 — Direction Classifier (Time-Series Momentum Ensemble)
------------------------------------------------------------------

This module implements **Model A** from the locked spec:
- Price-only, once-per-day decision at 00:00 UTC
- Direction signal D_t built from a 120/180/300-day (demo-tier) TS-momentum ensemble with a 7-day skip window (use 360 on Pro)
- 3-state machine: Up / Range / Down with hysteresis and a 5-day dwell
- Audit-friendly logs and clear configuration for future maintenance

Default data source: CoinGecko v3 Market Chart endpoint.
Docs: https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart

You can also provide a local long-format CSV (from Binance/Kraken fetchers) via:
  --price-csv data/kraken/kraken_daily.csv
  --symbol-map "bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR"
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import logging
import math
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# -----------------------------------------------------------------------------
# Versioning & Constants
# -----------------------------------------------------------------------------
MODEL_A_VERSION = "ModelA-v1.1-demo"
USER_AGENT = "ModelA/1.0"

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
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
    """Network configuration for CoinGecko fetches."""
    vs_currency: str = "usd"
    days: int = 365
    interval: str = "daily"
    precision: int = 5
    timeout_sec: float = 20.0
    retries: int = 3
    backoff_factor: float = 0.3


@dataclass
class ModelAConfig:
    """All model parameters (transparent & auditable)."""
    lookbacks: Tuple[int, int, int] = (120, 180, 300)
    skip_days: int = 7

    # Hysteresis bands (enter vs exit)
    enter_up_z: float = +0.25
    exit_up_z: float = +0.10
    enter_down_z: float = -0.25
    exit_down_z: float = -0.10

    dwell_days: int = 5            # freeze period after any state change

    # Data quality caps / requirements
    min_history_days: int = 365
    cap_abs_z: float = 3.0

    # Light outlier clipping for returns
    clip_return_pct: float = 0.001

    # State names
    state_up: str = "Up"
    state_range: str = "Range"
    state_down: str = "Down"


# -----------------------------------------------------------------------------
# CoinGecko HTTP client
# -----------------------------------------------------------------------------
class CoinGeckoClient:
    """Thin client around CoinGecko 'market_chart' endpoint."""

    def __init__(self, api_key: Optional[str] = None, fetch_cfg: Optional[FetchConfig] = None):
        self.api_key = api_key
        self.fetch_cfg = fetch_cfg or FetchConfig()
        self.session = requests.Session()

        retry = Retry(
            total=self.fetch_cfg.retries,
            read=self.fetch_cfg.retries,
            connect=self.fetch_cfg.retries,
            backoff_factor=self.fetch_cfg.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": USER_AGENT})
        if self.api_key:
            # Demo keys typically use this header; adjust if your tier differs.
            self.session.headers.update({"x-cg-demo-api-key": self.api_key})

    def get_market_chart(self, coin_id: str) -> pd.Series:
        """Return a UTC-date-indexed Series of closes for the given coin_id."""
        params = {
            "vs_currency": self.fetch_cfg.vs_currency,
            "days": str(self.fetch_cfg.days),
            "interval": self.fetch_cfg.interval,
            "precision": str(self.fetch_cfg.precision),
        }
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        resp = self.session.get(url, params=params, timeout=self.fetch_cfg.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"CoinGecko error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if "prices" not in data:
            raise ValueError("Unexpected response: 'prices' missing")

        df = pd.DataFrame(data["prices"], columns=["ts_ms", "close"])
        # Map to UTC day; CG timestamps are ms since epoch
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
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
    Minimal drop-in replacement for CoinGeckoClient.
    Expects the CSV to have columns: date, asset, close (like our fetchers produce).
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
# Barometer construction
# -----------------------------------------------------------------------------
def build_barometer(client, kind: str = "bitcoin") -> pd.Series:
    """Return a normalized daily barometer price series."""
    k = kind.lower().strip()
    if k in ("bitcoin", "btc"):
        series = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC")
    elif k in ("ethereum", "eth"):
        series = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
    elif k in ("solana", "sol"):
        series = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
    elif k == "btc_eth_5050":
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC"); time.sleep(0.2)
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
        df = pd.concat([btc, eth], axis=1).dropna()
        comb = np.log(df).diff().mean(axis=1)
        series = 100.0 * np.exp(comb.cumsum()); series.name = "BTC_ETH_50_50"
    elif k in ("btc_eth_sol_333", "btc_eth_sol", "btc_eth_sol_ew"):
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC"); time.sleep(0.2)
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH"); time.sleep(0.2)
        sol = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
        df = pd.concat([btc, eth, sol], axis=1).dropna()
        comb = np.log(df).diff().mean(axis=1)
        series = 100.0 * np.exp(comb.cumsum()); series.name = "BTC_ETH_SOL_EQ"
    else:
        raise ValueError("Unknown barometer kind. Use 'bitcoin'|'ethereum'|'solana'|'btc_eth_5050'|'btc_eth_sol_333'.")

    # Daily, deduped, ffilled
    series = series[~series.index.duplicated(keep="last")].sort_index().asfreq("D").ffill()
    return series


# -----------------------------------------------------------------------------
# Signal math (price-only, per spec)
# -----------------------------------------------------------------------------
def _clip_outliers(ret: pd.Series, pct: float) -> pd.Series:
    if ret.isna().all():
        return ret
    lo, hi = ret.quantile([pct, 1 - pct])
    return ret.clip(lower=lo, upper=hi)


def compute_direction_signal(series: pd.Series, cfg: ModelAConfig) -> pd.DataFrame:
    s = series.dropna().copy(); s.name = "close"
    logret = _clip_outliers(np.log(s).diff(), cfg.clip_return_pct)

    out = pd.DataFrame(index=s.index)
    out["close"] = s

    for K in cfg.lookbacks:
        # sum of log returns excluding the most recent skip_days
        roll_sum_total = logret.rolling(K + cfg.skip_days).sum()
        roll_sum_skip = logret.rolling(cfg.skip_days).sum()
        m = roll_sum_total - roll_sum_skip
        sigma = logret.rolling(K).std()
        out[f"z{K}"] = m / (sigma + 1e-12)

    out["D"] = out[[f"z{K}" for K in cfg.lookbacks]].median(axis=1)
    out["D_capped"] = out["D"].clip(lower=-cfg.cap_abs_z, upper=cfg.cap_abs_z)
    return out


@dataclass
class StateRecord:
    date: pd.Timestamp
    close: float
    z_values: Dict[int, float]
    D: float
    state: str
    in_dwell: bool
    transition_reason: str


def run_state_machine(signal_df: pd.DataFrame, cfg: ModelAConfig) -> pd.DataFrame:
    out = signal_df[["close", "D"] + [f"z{K}" for K in cfg.lookbacks]].copy()

    state_series: List[str] = []
    dwell_series: List[bool] = []
    reason_series: List[str] = []

    current_state = cfg.state_range
    dwell_left = 0

    for _, row in out.iterrows():
        D_t = row["D"]
        reason = ""

        if dwell_left > 0:
            next_state = current_state
            reason = f"dwell({dwell_left})"
            if current_state != cfg.state_up and D_t >= cfg.enter_up_z:
                next_state = cfg.state_up; dwell_left = cfg.dwell_days
                reason = f"Enter Up: D={D_t:.3f} >= {cfg.enter_up_z:+.2f} (override dwell)"
            elif current_state != cfg.state_down and D_t <= cfg.enter_down_z:
                next_state = cfg.state_down; dwell_left = cfg.dwell_days
                reason = f"Enter Down: D={D_t:.3f} <= {cfg.enter_down_z:+.2f} (override dwell)"
            else:
                dwell_left -= 1
        else:
            if current_state == cfg.state_up:
                if D_t <= cfg.exit_up_z:
                    current_state = cfg.state_range; dwell_left = cfg.dwell_days
                    reason = f"Exit Up→Range: D={D_t:.3f} <= {cfg.exit_up_z:+.2f}"
            elif current_state == cfg.state_down:
                if D_t >= cfg.exit_down_z:
                    current_state = cfg.state_range; dwell_left = cfg.dwell_days
                    reason = f"Exit Down→Range: D={D_t:.3f} >= {cfg.exit_down_z:+.2f}"
            else:
                if D_t >= cfg.enter_up_z:
                    current_state = cfg.state_up; dwell_left = cfg.dwell_days
                    reason = f"Enter Up: D={D_t:.3f} >= {cfg.enter_up_z:+.2f}"
                elif D_t <= cfg.enter_down_z:
                    current_state = cfg.state_down; dwell_left = cfg.dwell_days
                    reason = f"Enter Down: D={D_t:.3f} <= {cfg.enter_down_z:+.2f}"
            next_state = current_state

        state_series.append(next_state)
        dwell_series.append(dwell_left > 0)
        reason_series.append(reason)

    out["state"] = state_series
    out["direction"] = out["state"]
    out["in_dwell"] = dwell_series
    out["reason"] = reason_series

    out["D"] = out["D"].clip(-cfg.cap_abs_z, cfg.cap_abs_z)
    zcols = [f"z{K}" for K in cfg.lookbacks]
    out["data_constrained"] = out[zcols].isna().any(axis=1)
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_api_key(flag_key: Optional[str] = None, file_path: Optional[str] = None) -> Optional[str]:
    """Resolve CoinGecko API key without printing the secret."""
    if flag_key and flag_key.strip():
        return flag_key.strip()
    candidates: List[Path] = []
    if file_path:
        candidates.append(Path(file_path))
    candidates.append(Path.cwd() / "coingecko.key")
    for p in candidates:
        try:
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    tok = line.strip()
                    if tok and not tok.startswith("#"):
                        return tok
        except Exception:
            continue
    env = os.getenv("COINGECKO_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


def parse_symbol_map_arg(arg: Optional[str]) -> Dict[str, str]:
    """
    Parse 'bitcoin:BTCUSDT,ethereum:ETHUSDT,solana:SOLUSDT' into a dict.
    Keys are lowercased coin_ids ('bitcoin','ethereum','solana').
    """
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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@dataclass
class ModelAResult:
    """Container for outputs used by the allocator and for auditing."""
    version: str
    barometer_kind: str
    data: pd.DataFrame  # columns: close, z120,z180,z360, D, state, in_dwell, reason

    def latest_row(self) -> pd.Series:
        return self.data.iloc[-1]

    def save_csv(self, path: str) -> None:
        self.data.to_csv(path, index_label="date")


def run_model_a(
    api_key: Optional[str] = None,
    barometer_kind: str = "btc",
    fetch_cfg: Optional[FetchConfig] = None,
    model_cfg: Optional[ModelAConfig] = None,
    client: Optional[object] = None,   # pass LocalCSVClient or leave None for CoinGecko
) -> ModelAResult:
    logging.info("Running Model A — %s", MODEL_A_VERSION)
    fetch_cfg = fetch_cfg or FetchConfig()
    model_cfg = model_cfg or ModelAConfig()

    if client is None:
        client = CoinGeckoClient(api_key=api_key, fetch_cfg=fetch_cfg)

    series = build_barometer(client, kind=barometer_kind)

    if len(series) < model_cfg.min_history_days:
        raise RuntimeError(f"Insufficient history: have {len(series)} days, need >= {model_cfg.min_history_days}.")

    sig = compute_direction_signal(series, model_cfg)
    states = run_state_machine(sig, model_cfg)

    return ModelAResult(version=MODEL_A_VERSION, barometer_kind=barometer_kind, data=states)


# -----------------------------------------------------------------------------
# Minimal CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModelA-v1.0 — Direction Classifier")

    parser.add_argument("--api-key", type=str, default=None, help="CoinGecko API key (optional)")
    parser.add_argument("--api-key-file", type=str, default=None, help="Path to a .key file (default: ./coingecko.key)")

    parser.add_argument(
        "--barometer-kind",
        type=str,
        default="bitcoin",
        choices=["bitcoin","btc","ethereum","eth","solana","sol","btc_eth_5050","btc_eth_sol_333","btc_eth_sol","btc_eth_sol_ew"],
        help="Barometer construction",
    )

    parser.add_argument("--days", type=int, default=365, help="CoinGecko days (demo cap 365; Pro can use 540+)")
    parser.add_argument("--vs", type=str, default="usd", help="CoinGecko quote currency")

    parser.add_argument("--price-csv", type=str, default=None,
                        help="Read prices from local long CSV (date,asset,close) instead of CoinGecko.")
    parser.add_argument("--symbol-map", type=str, default=None,
                        help="Mapping for local CSV, e.g. 'bitcoin:XBTEUR,ethereum:ETHEUR,solana:SOLEUR' or USDT symbols.")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (auto-named if omitted).")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    fetch_cfg = FetchConfig(vs_currency=args.vs, days=args.days)
    model_cfg = ModelAConfig()

    try:
        # choose data source (local CSV vs CoinGecko)
        client = None
        if args.price_csv:
            smap = parse_symbol_map_arg(args.symbol_map)
            if not smap:
                # Try to infer from the CSV
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

        api_key = load_api_key(args.api_key, args.api_key_file)

        res = run_model_a(
            api_key=api_key,
            barometer_kind=args.barometer_kind,
            fetch_cfg=fetch_cfg,
            model_cfg=model_cfg,
            client=client,
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_path = args.out or f"modelA_output_{args.barometer_kind}_{ts}.csv"
        res.save_csv(out_path)
        last = res.latest_row()
        logging.info(
            "Latest: date=%s close=%.2f D=%.3f state=%s dwell=%s",
            last.name.date(), last["close"], last["D"], last["state"], last["in_dwell"]
        )
        print(f"Saved daily log to {out_path} — version {res.version}")
    except Exception as e:
        logging.exception("Model A failed: %s", e)
        raise
