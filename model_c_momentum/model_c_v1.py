"""
ModelC-v1.0 — Cross-Sectional Momentum (Kraken EUR universe, long-only)
-----------------------------------------------------------------------

MVP implementation for **Model C** using **Kraken REST API** (public endpoints).
- Universe: EUR-quoted spot pairs from /0/public/AssetPairs
- Signals: R90 and R300 on **daily close**, both with a **7-day skip** (t-7 to t-97 / t-307)
- Selection: long-only, **N≈10** names by composite rank (50% R90 rank, 50% R300 rank)
- Weights: **equal-weight** (defaults), with **cap** and **absolute floor** in EUR
- Exclusions: stablecoins/pegs, leveraged tokens (UP/DOWN/3L/3S etc.), dust-price
- Outputs (timestamped):
    - weights CSV:  asset, pair_key, wsname, weight (sums to 1)
    - diagnostics CSV: asset, ranks/scores, R90/R300 values, ADV30(EUR), applied filters

Notes
-----
- Size is small (~€1k), so we bias for **simplicity and low ticket count** (N=10).
- API etiquette: retries + small delay between OHLC calls; optional --max-pairs cap.
- If you provide a previous weights file, we apply **entry/exit buffers** to reduce churn.
- This script focuses on **portfolio construction**; gross exposure & volatility targeting live in the allocator.

Docs
----
- Kraken REST API: Asset Pairs   → GET /0/public/AssetPairs
- Kraken REST API: OHLC (daily)  → GET /0/public/OHLC?pair=<PAIR>&interval=1440&since=<unix>

"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

MODEL_C_VERSION = "ModelC-v1.0"
API_BASE = "https://api.kraken.com/0"
USER_AGENT = "Tether-Econ/ModelC (https://tether.to)"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class UniverseConfig:
    include_inactive: bool = False   # include non-online pairs if status is present
    max_pairs: int = 120             # limit to first N EUR pairs for API economy (sorted by name)
    dust_price_eur: float = 0.01     # exclude bases with median close < dust
    stable_keywords: Tuple[str, ...] = (
        # common stablecoins / pegs (case-insensitive, matched on cleaned symbol)
        "usdt","usdc","dai","tusd","eusd","eurt","eurl","eurs","usde","fdusd","gusd","usdp","pax","pyusd","susd",
    )
    leveraged_keywords: Tuple[str, ...] = (
        # generic leveraged token markers
        "up","down","3l","3s","bull","bear","5l","5s","2l","2s",
    )

@dataclass
class SignalConfig:
    lookbacks: Tuple[int,int] = (90, 300)
    skip_days: int = 7
    since_days: int = 400           # how far back to request OHLC

@dataclass
class PortfolioConfig:
    N_target: int = 10
    N_min: int = 8
    N_max: int = 12
    entry_rank: int = 8             # add only if composite rank <= entry_rank
    exit_rank: int = 15             # remove only if rank >= exit_rank
    weight_cap: float = 0.25        # per-name max weight
    notional_floor_eur: float = 25.0
    portfolio_notional_eur: float = 1000.0

# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

def _session_with_retries(retries: int = 3, backoff: float = 0.25) -> requests.Session:
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
    return s

# -----------------------------------------------------------------------------
# Kraken endpoints
# -----------------------------------------------------------------------------

def fetch_asset_pairs(session: requests.Session) -> Dict[str, dict]:
    url = f"{API_BASE}/public/AssetPairs"
    r = session.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    payload = r.json()
    if not isinstance(payload, dict) or "error" not in payload or "result" not in payload:
        raise RuntimeError("Unexpected response shape from Kraken AssetPairs")
    if payload["error"]:
        raise RuntimeError(f"Kraken API error(s): {payload['error']}")
    result = payload["result"]
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected 'result' type from Kraken AssetPairs")
    return result


def fetch_ohlc_daily(session: requests.Session, pair_key: str, since_unix: int) -> pd.DataFrame:
    """Fetch daily OHLC for a Kraken pair_key since a UNIX timestamp.

    Returns DataFrame with columns: [time, open, high, low, close, vwap, volume, trades]
    time = POSIX seconds (UTC), one row per day (interval=1440)
    """
    params = {"pair": pair_key, "interval": 1440, "since": since_unix}
    url = f"{API_BASE}/public/OHLC"
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC error {payload['error']}")
    result = payload.get("result", {})
    # result has key == pair_key (or alt key) mapping to list of rows
    # also has 'last' cursor we ignore here
    # find first list value
    rows = None
    for k, v in result.items():
        if k == "last":
            continue
        if isinstance(v, list):
            rows = v
            break
    if rows is None:
        return pd.DataFrame(columns=["time","open","high","low","close","vwap","volume","trades"])  # empty
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","trades"])\
            .astype({"time": int, "open": float, "high": float, "low": float, "close": float, "vwap": float, "volume": float, "trades": int})
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor("D").dt.tz_convert(None)
    df = df.set_index("date").sort_index()
    return df

# -----------------------------------------------------------------------------
# Universe building & filters
# -----------------------------------------------------------------------------

def _is_eur_quote(meta: dict) -> bool:
    quote = str(meta.get("quote", ""))
    wsname = str(meta.get("wsname", ""))
    altname = str(meta.get("altname", ""))
    if quote.upper() == "ZEUR":
        return True
    if wsname.upper().endswith("/EUR") or altname.upper().endswith("EUR"):
        return True
    return False


def _clean_symbol_from_ws(wsname: Optional[str]) -> Optional[str]:
    if not wsname:
        return None
    base = wsname.split("/")[0]
    # Kraken often uses XBT for BTC; keep as-is, but lowercase for matching
    return base.strip()


def _is_stable_or_peg(sym: Optional[str], keywords: Tuple[str,...]) -> bool:
    if not sym:
        return False
    s = sym.lower()
    # strip common prefixes like 'x', 'z'
    s = s.lstrip("xz")
    for kw in keywords:
        if kw in s:
            return True
    return False


def _is_leveraged(sym: Optional[str], keywords: Tuple[str,...]) -> bool:
    if not sym:
        return False
    s = sym.lower()
    # detect suffix patterns like btc3l, ethup, solbull
    for kw in keywords:
        if s.endswith(kw) or s.find(kw) >= 0:
            return True
    return False


def list_eur_pairs(session: requests.Session, ucfg: UniverseConfig) -> List[dict]:
    raw = fetch_asset_pairs(session)
    rows = []
    for key, meta in raw.items():
        if not _is_eur_quote(meta):
            continue
        status = meta.get("status")
        if (not ucfg.include_inactive) and (status is not None) and (str(status).lower() != "online"):
            continue
        wsname = meta.get("wsname")
        base_sym = _clean_symbol_from_ws(wsname)
        if _is_stable_or_peg(base_sym, ucfg.stable_keywords):
            continue
        if _is_leveraged(base_sym, ucfg.leveraged_keywords):
            continue
        rows.append({"pair_key": key, "wsname": wsname, "altname": meta.get("altname"), "base": meta.get("base"), "quote": meta.get("quote")})
    # Sort by wsname for determinism and cap to max_pairs
    rows.sort(key=lambda r: (r["wsname"] or r["pair_key"]))
    if ucfg.max_pairs and len(rows) > ucfg.max_pairs:
        rows = rows[:ucfg.max_pairs]
    return rows

# -----------------------------------------------------------------------------
# Signals & ranks
# -----------------------------------------------------------------------------

def _ts_momentum_with_skip(close: pd.Series, K: int, skip: int) -> float:
    """Return log(P_{t-skip}) - log(P_{t-(K+skip)}) for the latest date t.
    Returns np.nan if insufficient history.
    """
    if len(close) < (K + skip + 1):
        return np.nan
    p2 = float(close.iloc[-(skip+1)])  # P_{t-skip}
    p1 = float(close.iloc[-(K+skip+1)])  # P_{t-(K+skip)}
    if p1 <= 0 or p2 <= 0:
        return np.nan
    return math.log(p2) - math.log(p1)


def compute_asset_stats(ohlc: pd.DataFrame, scfg: SignalConfig, dust_price_eur: float) -> dict:
    """Compute per-asset stats used for ranking and filtering.
    Expects OHLC with columns [open, high, low, close, vwap, volume].
    """
    if ohlc.empty:
        return {"ok": False, "why": "no_ohlc"}
    # Ensure daily continuity
    s = ohlc["close"].astype(float).copy()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.asfreq("D").ffill()

    median_close = float(pd.Series(s).tail(30).median()) if len(s) >= 1 else np.nan
    if not np.isnan(median_close) and median_close < dust_price_eur:
        return {"ok": False, "why": "dust_price"}

    # ADV30 in EUR using vwap*volume if possible
    df_last30 = ohlc.tail(30)
    if {"vwap","volume"}.issubset(df_last30.columns):
        adv30_eur = float((df_last30["vwap"] * df_last30["volume"]).sum() / max(len(df_last30),1))
    else:
        adv30_eur = float((df_last30["close"] * df_last30.get("volume", 0)).sum() / max(len(df_last30),1))

    r90 = _ts_momentum_with_skip(s, K=90, skip=scfg.skip_days)
    r300 = _ts_momentum_with_skip(s, K=300, skip=scfg.skip_days)

    return {
        "ok": True,
        "median_close": median_close,
        "adv30_eur": adv30_eur,
        "r90": r90,
        "r300": r300,
        "len": len(s),
    }


def rank_and_score(df_stats: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional ranks & composite score (higher is better)."""
    out = df_stats.copy()
    # Rank descending by raw R values; missing treated as worst
    out["rank90"] = (-out["r90"]).rank(method="average", na_option="bottom")
    out["rank300"] = (-out["r300"]).rank(method="average", na_option="bottom")

    # Composite: penalize missing long leg by adding +5 to rank
    penalty = 5.0
    r300_missing = out["r300"].isna()
    rank300_eff = out["rank300"] + penalty * r300_missing.astype(float)

    out["score"] = 0.5 * out["rank90"] + 0.5 * rank300_eff
    # Lower score (rank) is better; convert to ordinal position for readability
    out["pos"] = out["score"].rank(method="first")
    return out

# -----------------------------------------------------------------------------
# Membership & weights
# -----------------------------------------------------------------------------

def apply_membership_with_buffers(ranked: pd.DataFrame, pcfg: PortfolioConfig, prev_holdings: Optional[Dict[str,float]] = None) -> List[str]:
    """Return the selected asset list applying entry/exit buffers if prev holdings provided.
    ranked: DataFrame indexed by asset key with column 'pos' (1=best), 'score'.
    """
    sorted_ids = list(ranked.sort_values("score").index)
    if not prev_holdings:
        return sorted_ids[:pcfg.N_target]

    current = set(prev_holdings.keys())
    keep = [aid for aid in current if ranked.loc[aid, "pos"] <= pcfg.exit_rank] if len(ranked) else []

    needed = max(pcfg.N_min, min(pcfg.N_target, pcfg.N_max)) - len(keep)
    additions = [aid for aid in sorted_ids if (aid not in current) and (ranked.loc[aid, "pos"] <= pcfg.entry_rank)]
    selected = keep + additions[:max(0, needed)]

    # If still short of N_min, fill by best-available even if above entry threshold
    i = 0
    while len(selected) < pcfg.N_min and i < len(sorted_ids):
        cand = sorted_ids[i]
        if cand not in selected:
            selected.append(cand)
        i += 1

    # Trim to N_max if we somehow exceeded
    return selected[:pcfg.N_max]


def compute_equal_weights(selected: List[str], pcfg: PortfolioConfig) -> Dict[str, float]:
    if not selected:
        return {}
    w = 1.0 / float(len(selected))
    return {aid: w for aid in selected}


def apply_caps_and_floors(weights: Dict[str, float], pcfg: PortfolioConfig) -> Dict[str, float]:
    if not weights:
        return {}

    # Normalize positives to 1.0
    raw = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(raw.values())
    if s <= 0:
        return {}
    base = {k: v / s for k, v in raw.items()}

    cap = float(pcfg.weight_cap)
    # Initial clip
    w = {k: min(v, cap) for k, v in base.items()}

    # Water-fill: redistribute remaining mass without breaching cap
    def total(wd): return sum(wd.values())
    deficit = 1.0 - total(w)

    # If impossible to reach sum=1 because cap*N < 1, we stop at the feasible sum.
    feasible_sum = min(1.0, cap * len(w))
    if feasible_sum < 1.0:
        # Keep clipped allocation (sum == feasible_sum)
        # Floors still apply below (may reduce further).
        pass
    else:
        # Iteratively allocate deficit to names with headroom, proportional to base
        while deficit > 1e-12:
            headroom = {k: cap - w[k] for k in w if w[k] < cap - 1e-12}
            if not headroom:
                break
            base_mass = sum(base[k] for k in headroom)
            if base_mass <= 0:
                break
            changed = False
            for k in list(headroom.keys()):
                add = deficit * (base[k] / base_mass)
                new_w = min(cap, w[k] + add)
                if new_w > w[k] + 1e-15:
                    changed = True
                w[k] = new_w
            new_deficit = 1.0 - total(w)
            if not changed or abs(new_deficit - deficit) < 1e-12:
                break
            deficit = new_deficit

    # Apply absolute EUR floor
    floor_w = pcfg.notional_floor_eur / max(pcfg.portfolio_notional_eur, 1e-9)
    kept = {k: v for k, v in w.items() if v >= floor_w}

    if not kept:
        # Fallback: keep the largest within cap
        k_top = max(w, key=w.get)
        return {k_top: w[k_top]}

    # Try a gentle renorm first (will often land exactly at target if no caps bind)
    sum_kept = sum(kept.values())
    target = min(1.0, cap * len(kept))
    if sum_kept > 0 and sum_kept <= target + 1e-12:
        scale = target / sum_kept
        kept = {k: min(v * scale, cap) for k, v in kept.items()}

    # If scaling hit a cap and left a deficit, do a post-floor water-fill to reach target
    curr = sum(kept.values())
    if curr + 1e-12 < target:
        deficit = target - curr
        while deficit > 1e-12:
            headroom = {k: cap - kept[k] for k in kept if kept[k] < cap - 1e-12}
            if not headroom:
                break
            # Prefer base-weighted redistribution; fall back to equal if base mass is zero
            base_mass = sum(base.get(k, 0.0) for k in headroom)
            changed = False
            if base_mass > 0:
                for k in list(headroom.keys()):
                    add = deficit * (base.get(k, 0.0) / base_mass)
                    inc = min(add, headroom[k])
                    if inc > 1e-15:
                        kept[k] += inc
                        changed = True
            else:
                share = deficit / len(headroom)
                for k in list(headroom.keys()):
                    inc = min(share, headroom[k])
                    if inc > 1e-15:
                        kept[k] += inc
                        changed = True
            new_deficit = target - sum(kept.values())
            if not changed or abs(new_deficit - deficit) < 1e-12:
                break
            deficit = new_deficit

    return kept

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def run_model_c(
    ucfg: UniverseConfig,
    scfg: SignalConfig,
    pcfg: PortfolioConfig,
    prev_weights_path: Optional[str] = None,
    sleep_sec: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weights_df, diagnostics_df). Index by asset key (pair_key)."""
    session = _session_with_retries()

    # 1) Universe
    pairs = list_eur_pairs(session, ucfg)
    if not pairs:
        raise RuntimeError("No EUR pairs found after filters.")

    since_unix = int((datetime.now(timezone.utc) - timedelta(days=scfg.since_days)).timestamp())

    # 2) Pull OHLC & compute stats
    stats_rows = []
    ohlc_map: Dict[str,pd.DataFrame] = {}
    for i, meta in enumerate(pairs, 1):
        pair_key = meta["pair_key"]
        wsname = meta.get("wsname")
        try:
            df = fetch_ohlc_daily(session, pair_key, since_unix)
            if not df.empty:
                ohlc_map[pair_key] = df
                st = compute_asset_stats(df, scfg, ucfg.dust_price_eur)
                st.update({"pair_key": pair_key, "wsname": wsname})
                if st.get("ok") and st.get("len",0) < (300 + scfg.skip_days + 1):
                    # Flag limited history (may miss R300); still allowed with penalty in ranking
                    st["limited_history"] = True
                stats_rows.append(st)
        except Exception as e:
            # Keep going on individual failures
            continue
        time.sleep(sleep_sec)

    if not stats_rows:
        raise RuntimeError("No OHLC stats available.")

    stats = pd.DataFrame(stats_rows).set_index("pair_key")
    # Filter to valid entries
    elig = stats[stats["ok"] == True].copy()
    if elig.empty:
        raise RuntimeError("No eligible assets after filters.")

    # 3) Ranks & composite
    ranked = rank_and_score(elig)

    # 4) Membership
    prev = None
    if prev_weights_path:
        try:
            wprev = pd.read_csv(prev_weights_path)
            prev = {str(r["pair_key"]): float(r["weight"]) for _, r in wprev.iterrows() if r.get("pair_key") is not None}
        except Exception:
            prev = None
    selected = apply_membership_with_buffers(ranked, pcfg, prev)

    # 5) Weights
    w0 = compute_equal_weights(selected, pcfg)
    w = apply_caps_and_floors(w0, pcfg)

    # 6) Build outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    weights_df = pd.DataFrame(
        [{"pair_key": k, "wsname": stats.loc[k, "wsname"], "weight": v} for k, v in w.items()]
    ).set_index("pair_key").sort_values("weight", ascending=False)

    diag = ranked.copy()
    diag["selected"] = diag.index.isin(selected)
    diag["weight"] = diag.index.map(w).fillna(0.0)
    diag["version"] = MODEL_C_VERSION
    diag["timestamp_utc"] = ts

    return weights_df, diag

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ModelC-v1.0 — Cross-Sectional Momentum on Kraken EUR pairs")

    # Universe
    ap.add_argument("--max-pairs", type=int, default=120, help="Max EUR pairs to consider (API economy)")
    ap.add_argument("--include-inactive", action="store_true", help="Include non-online pairs if API returns statuses")
    ap.add_argument("--dust-price-eur", type=float, default=0.01, help="Exclude bases with median close < this")

    # Signals
    ap.add_argument("--since-days", type=int, default=400, help="Days of OHLC history to request")
    ap.add_argument("--skip-days", type=int, default=7, help="Skip window for momentum legs")

    # Portfolio
    ap.add_argument("--N", type=int, default=10, help="Target number of names")
    ap.add_argument("--entry-rank", type=int, default=8, help="Entry threshold (pos<=) if prev weights provided")
    ap.add_argument("--exit-rank", type=int, default=15, help="Exit threshold (pos>=) if prev weights provided")
    ap.add_argument("--cap", type=float, default=0.25, help="Per-name weight cap")
    ap.add_argument("--floor-eur", type=float, default=25.0, help="Absolute notional floor per name in EUR")
    ap.add_argument("--notional-eur", type=float, default=1000.0, help="Portfolio notional in EUR")

    # Previous holdings for buffers
    ap.add_argument("--prev-weights", type=str, default=None, help="Path to previous weights CSV to apply buffers")

    # Outputs
    ap.add_argument("--out-weights", type=str, default=None, help="Output path for weights CSV (timestamped if omitted)")
    ap.add_argument("--out-diag", type=str, default=None, help="Output path for diagnostics CSV (timestamped if omitted)")

    args = ap.parse_args(argv)

    ucfg = UniverseConfig(include_inactive=args.include_inactive, max_pairs=args.max_pairs, dust_price_eur=args.dust_price_eur)
    scfg = SignalConfig(lookbacks=(90,300), skip_days=args.skip_days, since_days=args.since_days)
    pcfg = PortfolioConfig(N_target=args.N, N_min=max(1,args.N-2), N_max=args.N+2, entry_rank=args.entry_rank, exit_rank=args.exit_rank,
                           weight_cap=args.cap, notional_floor_eur=args.floor_eur, portfolio_notional_eur=args.notional_eur)

    try:
        weights_df, diag = run_model_c(ucfg, scfg, pcfg, prev_weights_path=args.prev_weights)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

        out_w = args.out_weights or f"modelC_weights_EUR_{ts}.csv"
        out_d = args.out_diag or f"modelC_diagnostics_EUR_{ts}.csv"

        weights_df.to_csv(out_w, index_label="pair_key")
        diag.to_csv(out_d, index_label="pair_key")

        print(f"Saved weights to {out_w} and diagnostics to {out_d} — version {MODEL_C_VERSION}")
        return 0
    except Exception as e:
        print(f"Model C failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
