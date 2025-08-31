"""
ModelA-v1.0 — Direction Classifier (Time-Series Momentum Ensemble)
------------------------------------------------------------------

This module implements **Model A** from the locked spec:
- Price-only, once-per-day decision at 00:00 UTC
- Direction signal D_t built from a 120/180/300-day (demo-tier) TS-momentum ensemble with a 7-day skip window (use 360 on Pro)
- 3-state machine: Up / Range / Down with hysteresis and a 5-day dwell
- Audit-friendly logs and clear configuration for future maintenance

Data source: Coingecko v3 Market Chart endpoint.
Docs: https://docs.coingecko.com/v3.0.1/reference/coins-id-market-chart
Example endpoint:
  https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=540&interval=daily&precision=5

Notes
-----
- For a **BTC-only** barometer, call `fetch_price_series('bitcoin')`.
- For a **50/50 BTC+ETH** barometer, use `build_barometer('btc_eth_5050')`.
- Coingecko rate limits: ~30 req/min. This module typically makes 1–2 calls/run.
- This file avoids any scheduler/cron specifics; it focuses on signal math + a minimal CLI.

Dependencies
------------
Python 3.9+
requests, pandas, numpy

Maintenance principles
----------------------
- All thresholds and lookbacks live in `ModelAConfig` (single source of truth)
- The state machine is pure and testable (`run_state_machine`)
- The fetcher is isolated (`CoinGeckoClient`)
- Logged outputs are versioned and auditable (see `MODEL_A_VERSION`)

"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

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
USER_AGENT = "ModelA/1.0 (+https://tether.to)"

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
    """Network configuration for Coingecko fetches.

    Attributes
    ----------
    vs_currency : str
        Fiat/quote currency for prices.
    days : int
        Number of calendar days to request from the API. Demo cap: 365 days. If/when you move to Pro, use >= 540 to comfortably cover the 360-day lookback + 7-day skip + buffer.
    interval : str
        "daily" recommended for our use; the endpoint aggregates intraday to daily.
    precision : int
        Decimal precision in the response.
    timeout_sec : float
        Per-request timeout.
    retries : int
        HTTP retry attempts on transient errors.
    backoff_factor : float
        Backoff factor for urllib3 Retry.
    """

    vs_currency: str = "usd"
    days: int = 365
    interval: str = "daily"
    precision: int = 5
    timeout_sec: float = 20.0
    retries: int = 3
    backoff_factor: float = 0.3


@dataclass
class ModelAConfig:
    """Holds all model parameters so changes are transparent & auditable.

    Lookbacks are in **calendar days** to align with API outputs.
    Thresholds are in **z-units** (signal-to-noise), per spec.
    """

    lookbacks: Tuple[int, int, int] = (120, 180, 300)
    skip_days: int = 7

    # Hysteresis bands (enter vs exit)
    enter_up_z: float = +0.25
    exit_up_z: float = +0.10
    enter_down_z: float = -0.25
    exit_down_z: float = -0.10

    # Dwell (freeze) period after any state change
    dwell_days: int = 5

    # Data quality caps / requirements
    min_history_days: int = 365  # must have this many days before going live
    cap_abs_z: float = 3.0       # cap |D_t| to avoid pathological values in logs

    # Outlier clipping for raw returns (very light)
    clip_return_pct: float = 0.001  # two-sided percentile (0.1%)

    # State names (kept here to avoid magic strings elsewhere)
    state_up: str = "Up"
    state_range: str = "Range"
    state_down: str = "Down"


# -----------------------------------------------------------------------------
# HTTP client for Coingecko
# -----------------------------------------------------------------------------
class CoinGeckoClient:
    """Thin client around Coingecko 'market_chart' endpoint.

    We intentionally avoid SDKs to keep control over retries and response parsing.
    """

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
            # As of writing, Coingecko **demo** API key is passed via header 'x-cg-demo-api-key'
            self.session.headers.update({"x-cg-demo-api-key": self.api_key})

    def get_market_chart(self, coin_id: str) -> pd.Series:
        """Fetch daily closing prices for a coin and return a UTC-indexed Series.

        The endpoint returns 'prices' as [[timestamp_ms, price], ...].
        We map to end-of-day UTC dates and pick one value per date.
        """
        params = {
            "vs_currency": self.fetch_cfg.vs_currency,
            "days": str(self.fetch_cfg.days),
            "interval": self.fetch_cfg.interval,
            "precision": str(self.fetch_cfg.precision),
        }
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        resp = self.session.get(url, params=params, timeout=self.fetch_cfg.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"Coingecko error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if "prices" not in data:
            raise ValueError("Unexpected response: 'prices' missing")

        # Convert to DataFrame
        arr = data["prices"]
        df = pd.DataFrame(arr, columns=["ts_ms", "close"])
        # Convert to UTC date; Coingecko timestamps are in ms since epoch
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("D").dt.tz_convert(None)

        # Aggregate by date (there should usually be one row per day already)
        daily = df.groupby("date", as_index=True)["close"].last().sort_index()
        daily.name = coin_id
        return daily


# -----------------------------------------------------------------------------
# Barometer construction
# -----------------------------------------------------------------------------
def build_barometer(client: CoinGeckoClient, kind: str = "bitcoin") -> pd.Series:
    """Return a normalized daily barometer price series.

    kind options (case-insensitive):
      - 'bitcoin' / 'btc'
      - 'ethereum' / 'eth'
      - 'solana' / 'sol'
      - 'btc_eth_5050' -> equal-weight BTC+ETH synthetic index
      - 'btc_eth_sol_333' / 'btc_eth_sol' / 'btc_eth_sol_ew' -> equal-weight BTC+ETH+SOL index

    We normalize to start at 100.0 for interpretability; this does not affect
    the model because we work in log returns.
    """
    k = kind.lower().strip()
    if k in ("bitcoin", "btc"):
        ser = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC")
        series = ser
    elif k in ("ethereum", "eth"):
        ser = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
        series = ser
    elif k in ("solana", "sol"):
        ser = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
        series = ser
    elif k == "btc_eth_5050":
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC")
        time.sleep(0.2)  # be gentle with the API
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
        df = pd.concat([btc, eth], axis=1).dropna()
        logret = np.log(df).diff()
        comb = logret.mean(axis=1)
        idx = 100.0 * np.exp(comb.cumsum())
        idx.name = "BTC_ETH_50_50"
        series = idx
    elif k in ("btc_eth_sol_333", "btc_eth_sol", "btc_eth_sol_ew"):
        btc = client.get_market_chart(COIN_IDS["bitcoin"]).rename("BTC")
        time.sleep(0.2)
        eth = client.get_market_chart(COIN_IDS["ethereum"]).rename("ETH")
        time.sleep(0.2)
        sol = client.get_market_chart(COIN_IDS["solana"]).rename("SOL")
        df = pd.concat([btc, eth, sol], axis=1).dropna()
        logret = np.log(df).diff()
        comb = logret.mean(axis=1)
        idx = 100.0 * np.exp(comb.cumsum())
        idx.name = "BTC_ETH_SOL_EQ"
        series = idx
    else:
        raise ValueError("Unknown barometer kind. Use 'bitcoin', 'ethereum', 'solana', or 'btc_eth_5050'.")

    # Deduplicate daily index and sort
    series = series[~series.index.duplicated(keep="last")].sort_index()
    # Forward-fill occasional gaps if any
    series = series.asfreq("D").ffill()
    return series


# -----------------------------------------------------------------------------
# Signal math (price-only, per spec)
# -----------------------------------------------------------------------------
def _clip_outliers(ret: pd.Series, pct: float) -> pd.Series:
    """Two-sided percentile clipping for returns.

    Intended to be *very* light-touch; e.g., pct=0.001 clips the outer 0.1%.
    """
    if ret.isna().all():
        return ret
    lo, hi = ret.quantile([pct, 1 - pct])
    return ret.clip(lower=lo, upper=hi)


def compute_direction_signal(series: pd.Series, cfg: ModelAConfig) -> pd.DataFrame:
    """Compute z-scores for each lookback and aggregate into D_t per day.

    Returns DataFrame with columns:
      close, z120, z180, z360, D, D_capped
    """
    s = series.dropna().copy()
    s.name = "close"

    # Log returns and light outlier clipping
    logret = np.log(s).diff()
    logret = _clip_outliers(logret, cfg.clip_return_pct)

    out = pd.DataFrame(index=s.index)
    out["close"] = s

    for K in cfg.lookbacks:
        # rolling sum of log returns excluding the most recent skip_days
        # m_t = sum_{i=skip+1}^{K+skip} r_{t-i}
        roll_sum_total = logret.rolling(K + cfg.skip_days).sum()
        roll_sum_skip = logret.rolling(cfg.skip_days).sum()
        m = roll_sum_total - roll_sum_skip
        # rolling stdev over K days (no skip)
        sigma = logret.rolling(K).std()
        z = m / (sigma + 1e-12)
        out[f"z{K}"] = z

    # Aggregate via median across lookbacks
    out["D"] = out[[f"z{K}" for K in cfg.lookbacks]].median(axis=1)
    # Cap |D| for sanity (logging/reporting only)
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
    """Apply hysteresis & dwell to produce the 3-state classification over time.

    Returns DataFrame with columns:
      close, z120, z180, z360, D, state, in_dwell, reason
    """
    cols = {f"z{K}": f"z{K}" for K in cfg.lookbacks}
    out = signal_df[["close", "D"] + list(cols.keys())].copy()

    state_series: List[str] = []
    dwell_series: List[bool] = []
    reason_series: List[str] = []

    current_state = cfg.state_range
    dwell_left = 0

    for idx, row in out.iterrows():
        D_t = row["D"]
        reason = ""

        # Dwell rule: if we're still in dwell, keep the state unless a hard opposite entry is triggered
        if dwell_left > 0:
            next_state = current_state
            reason = f"dwell({dwell_left})"
            # However, allow an opposite *enter* to override (rare, but defined)
            if current_state != cfg.state_up and D_t >= cfg.enter_up_z:
                next_state = cfg.state_up
                dwell_left = cfg.dwell_days  # reset dwell
                reason = f"Enter Up: D={D_t:.3f} >= {cfg.enter_up_z:+.2f} (override dwell)"
            elif current_state != cfg.state_down and D_t <= cfg.enter_down_z:
                next_state = cfg.state_down
                dwell_left = cfg.dwell_days
                reason = f"Enter Down: D={D_t:.3f} <= {cfg.enter_down_z:+.2f} (override dwell)"
            else:
                dwell_left -= 1
        else:
            # Not in dwell: evaluate hysteresis bands
            if current_state == cfg.state_up:
                if D_t <= cfg.exit_up_z:
                    current_state = cfg.state_range
                    dwell_left = cfg.dwell_days
                    reason = f"Exit Up→Range: D={D_t:.3f} <= {cfg.exit_up_z:+.2f}"
            elif current_state == cfg.state_down:
                if D_t >= cfg.exit_down_z:
                    current_state = cfg.state_range
                    dwell_left = cfg.dwell_days
                    reason = f"Exit Down→Range: D={D_t:.3f} >= {cfg.exit_down_z:+.2f}"
            else:  # current_state == Range
                if D_t >= cfg.enter_up_z:
                    current_state = cfg.state_up
                    dwell_left = cfg.dwell_days
                    reason = f"Enter Up: D={D_t:.3f} >= {cfg.enter_up_z:+.2f}"
                elif D_t <= cfg.enter_down_z:
                    current_state = cfg.state_down
                    dwell_left = cfg.dwell_days
                    reason = f"Enter Down: D={D_t:.3f} <= {cfg.enter_down_z:+.2f}"
            next_state = current_state

        state_series.append(next_state)
        dwell_series.append(dwell_left > 0)
        reason_series.append(reason)

    out["state"] = state_series
    out["direction"] = out["state"]  # alias for downstream readability
    out["in_dwell"] = dwell_series
    out["reason"] = reason_series

    # Cap D for reporting
    out["D"] = out["D"].clip(-cfg.cap_abs_z, cfg.cap_abs_z)
    # Flag rows where some lookback couldn't be computed (e.g., 360d on demo tier)
    zcols = [f"z{K}" for K in cfg.lookbacks]
    out["data_constrained"] = out[zcols].isna().any(axis=1)
    return out


# -----------------------------------------------------------------------------
# API key loading helper (flag → file → env)
# -----------------------------------------------------------------------------
def load_api_key(flag_key: Optional[str] = None, file_path: Optional[str] = None) -> Optional[str]:
    """Resolve the CoinGecko API key without ever printing the secret.

    Resolution order:
    1) Explicit flag value (if non-empty)
    2) Explicit file path (first non-comment, non-empty line)
    3) Default local file ./coingecko.key (same parsing)
    4) Environment variable COINGECKO_API_KEY
    """
    # 1) direct flag
    if flag_key and flag_key.strip():
        return flag_key.strip()

    # 2) explicit file, then 3) default file
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
            continue

    # 4) environment variable
    env = os.getenv("COINGECKO_API_KEY")
    if env and env.strip():
        logging.debug("Loaded API key from environment variable COINGECKO_API_KEY")
        return env.strip()

    return None

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
) -> ModelAResult:
    """High-level orchestrator: fetch → compute signal → classify state.

    Parameters
    ----------
    api_key : Optional[str]
        Coingecko API key (header 'x-cg-pro-api-key'). If None, unauthenticated call.
    barometer_kind : str
        'btc' or 'btc_eth_5050'.
    fetch_cfg : FetchConfig
        Network/data window configuration.
    model_cfg : ModelAConfig
        Model parameters per locked spec.
    """
    logging.info("Running Model A — %s", MODEL_A_VERSION)
    fetch_cfg = fetch_cfg or FetchConfig()
    model_cfg = model_cfg or ModelAConfig()

    client = CoinGeckoClient(api_key=api_key, fetch_cfg=fetch_cfg)
    series = build_barometer(client, kind=barometer_kind)

    # Data quality checks
    if len(series) < model_cfg.min_history_days:
        raise RuntimeError(
            f"Insufficient history: have {len(series)} days, need >= {model_cfg.min_history_days}."
        )

    # Compute D and states
    sig = compute_direction_signal(series, model_cfg)
    states = run_state_machine(sig, model_cfg)

    result = ModelAResult(version=MODEL_A_VERSION, barometer_kind=barometer_kind, data=states)
    return result


# -----------------------------------------------------------------------------
# Minimal CLI (optional, for ad-hoc runs)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ModelA-v1.0 — Direction Classifier")

    parser.add_argument("--api-key", type=str, default=None, help="Coingecko API key (optional)")
    parser.add_argument("--api-key-file", type=str, default=None, help="Path to a .key file containing ONLY the API key (default: ./coingecko.key if present)")

    parser.add_argument(
        "--barometer-kind",
        type=str,
        default="bitcoin",
        choices=["bitcoin","btc","ethereum","eth","solana","sol","btc_eth_5050","btc_eth_sol_333","btc_eth_sol","btc_eth_sol_ew"],
        help="Barometer: 'bitcoin'|'ethereum'|'solana' or 'btc_eth_5050' (BTC+ETH equal-weight) or 'btc_eth_sol_333' (BTC+ETH+SOL equal-weight)",
    )

    parser.add_argument("--days", type=int, default=365, help="Days to request from API (demo cap 365; Pro can use 540+)")
    parser.add_argument("--vs", type=str, default="usd", help="Quote currency for prices (default: usd)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path. If omitted, auto-generates a timestamped filename.")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    fetch_cfg = FetchConfig(vs_currency=args.vs, days=args.days)
    model_cfg = ModelAConfig()  # use locked defaults; bump version if you change

    try:
        api_key = load_api_key(args.api_key, args.api_key_file)
        if api_key is None:
            logging.warning("No API key found via flag, file, or env; proceeding unauthenticated (rate limits may be lower)")
        res = run_model_a(api_key=api_key, barometer_kind=args.barometer_kind, fetch_cfg=fetch_cfg, model_cfg=model_cfg)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        default_name = f"modelA_output_{args.barometer_kind}_{ts}.csv"
        out_path = args.out or default_name
        res.save_csv(out_path)
        last = res.latest_row()
        logging.info("Latest: date=%s close=%.2f D=%.3f state=%s dwell=%s", last.name.date(), last["close"], last["D"], last["state"], last["in_dwell"])
        print(f"Saved daily log to {out_path} — version {res.version}")
    except Exception as e:
        logging.exception("Model A failed: %s", e)
        raise
