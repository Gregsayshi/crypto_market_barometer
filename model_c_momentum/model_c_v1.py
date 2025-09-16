"""
ModelC-v1.0 — Cross-Sectional Momentum (Kraken EUR universe, long-only)
-----------------------------------------------------------------------

Now supports cached prices:
  - Pass --price-csv to read a long CSV (date,asset,close) instead of calling Kraken OHLC
  - Choose how to match assets via --csv-key: altname|pair_key|wsname (default: altname)
  - Use --offline to forbid network calls entirely

Signals: R90 & R300 on daily close, both with a 7-day skip
Selection: long-only, N≈10 by composite rank (50% R90 rank, 50% R300 rank)
Weights: equal-weight with cap & absolute EUR floor
Outputs: weights CSV + diagnostics CSV (timestamped)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

MODEL_C_VERSION = "ModelC-v1.0"
API_BASE = "https://api.kraken.com/0"
USER_AGENT = "ModelC"

# ------------------------------- Config -------------------------------

@dataclass
class UniverseConfig:
    include_inactive: bool = False
    max_pairs: int = 120
    dust_price_eur: float = 0.01
    stable_keywords: Tuple[str, ...] = (
        "usdt","usdc","dai","tusd","eusd","eurt","eurl","eurs","usde","fdusd","gusd","usdp","pax","pyusd","susd",
    )
    leveraged_keywords: Tuple[str, ...] = ("up","down","3l","3s","bull","bear","5l","5s","2l","2s")

@dataclass
class SignalConfig:
    lookbacks: Tuple[int,int] = (90, 300)
    skip_days: int = 7
    since_days: int = 400

@dataclass
class PortfolioConfig:
    N_target: int = 10
    N_min: int = 8
    N_max: int = 12
    entry_rank: int = 8
    exit_rank: int = 15
    weight_cap: float = 0.25
    notional_floor_eur: float = 25.0
    portfolio_notional_eur: float = 1000.0

# --------------------------- HTTP helpers -----------------------------

def _session_with_retries(retries: int = 3, backoff: float = 0.25) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries, read=retries, connect=retries,
        backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",), raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s

# ---------------------------- Kraken API ------------------------------

def fetch_asset_pairs(session: requests.Session) -> Dict[str, dict]:
    url = f"{API_BASE}/public/AssetPairs"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken AssetPairs error: {payload['error']}")
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected 'result' type from Kraken AssetPairs")
    return result

def fetch_ohlc_daily(session: requests.Session, pair_key: str, since_unix: int) -> pd.Series:
    """Return daily close series (UTC date index) for a pair_key since given UNIX time."""
    params = {"pair": pair_key, "interval": 1440, "since": since_unix}
    url = f"{API_BASE}/public/OHLC"
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken OHLC error {payload['error']}")
    result = payload.get("result", {})
    rows = None
    for k, v in result.items():
        if k == "last": continue
        if isinstance(v, list): rows = v; break
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","trades"])
    df = df.astype({"time": int, "close": float})
    s = pd.to_datetime(df["time"], unit="s", utc=True).dt.floor("D").dt.tz_convert(None)
    close = pd.Series(df["close"].values, index=s).sort_index()
    return close.asfreq("D").ffill()

# ----------------------- Universe & name hygiene ----------------------

def _is_eur_quote(meta: dict) -> bool:
    quote = str(meta.get("quote", ""))
    wsname = str(meta.get("wsname", ""))
    altname = str(meta.get("altname", ""))
    return (quote.upper() == "ZEUR") or wsname.upper().endswith("/EUR") or altname.upper().endswith("EUR")

def _clean_symbol_from_ws(wsname: Optional[str]) -> Optional[str]:
    if not wsname: return None
    return wsname.split("/")[0].strip()

def _is_stable_or_peg(sym: Optional[str], keywords: Tuple[str,...]) -> bool:
    if not sym: return False
    s = sym.lower().lstrip("xz")
    return any(kw in s for kw in keywords)

def _is_leveraged(sym: Optional[str], keywords: Tuple[str,...]) -> bool:
    if not sym: return False
    s = sym.lower()
    return any(s.endswith(kw) or (kw in s) for kw in keywords)

def list_eur_pairs(session: requests.Session, ucfg: UniverseConfig) -> List[dict]:
    raw = fetch_asset_pairs(session)
    rows = []
    for key, meta in raw.items():
        if not _is_eur_quote(meta): continue
        status = meta.get("status")
        if (not ucfg.include_inactive) and (status is not None) and (str(status).lower() != "online"):
            continue
        wsname = meta.get("wsname")
        base_sym = _clean_symbol_from_ws(wsname)
        if _is_stable_or_peg(base_sym, ucfg.stable_keywords): continue
        if _is_leveraged(base_sym, ucfg.leveraged_keywords): continue
        rows.append({
            "pair_key": key,
            "wsname": wsname,
            "altname": meta.get("altname"),
            "base": meta.get("base"),
            "quote": meta.get("quote"),
        })
    rows.sort(key=lambda r: (r["wsname"] or r["pair_key"]))
    if ucfg.max_pairs and len(rows) > ucfg.max_pairs:
        rows = rows[:ucfg.max_pairs]
    return rows

# --------------------------- Signals & ranks --------------------------

def _ts_momentum_with_skip(close: pd.Series, K: int, skip: int) -> float:
    if len(close) < (K + skip + 1): return np.nan
    p2 = float(close.iloc[-(skip+1)])
    p1 = float(close.iloc[-(K+skip+1)])
    if p1 <= 0 or p2 <= 0: return np.nan
    return math.log(p2) - math.log(p1)

def compute_asset_stats_from_close(close: pd.Series, skip_days: int) -> Dict[str, float]:
    """Compute median_close, r90, r300, len from a daily close series."""
    s = close[~close.index.duplicated(keep="last")].sort_index().asfreq("D").ffill()
    median_close = float(pd.Series(s).tail(30).median()) if len(s) else np.nan
    r90  = _ts_momentum_with_skip(s,  K=90,  skip=skip_days)
    r300 = _ts_momentum_with_skip(s, K=300, skip=skip_days)
    return {"median_close": median_close, "r90": r90, "r300": r300, "len": len(s)}

def rank_and_score(df_stats: pd.DataFrame) -> pd.DataFrame:
    out = df_stats.copy()
    out["rank90"]  = (-out["r90"]).rank(method="average",  na_option="bottom")
    out["rank300"] = (-out["r300"]).rank(method="average", na_option="bottom")
    penalty = 5.0
    rank300_eff = out["rank300"] + penalty * out["r300"].isna().astype(float)
    out["score"] = 0.5 * out["rank90"] + 0.5 * rank300_eff
    out["pos"] = out["score"].rank(method="first")
    return out

# ---------------------- Membership & weights --------------------------

def apply_membership_with_buffers(ranked: pd.DataFrame, pcfg: PortfolioConfig, prev_holdings: Optional[Dict[str,float]] = None) -> List[str]:
    sorted_ids = list(ranked.sort_values("score").index)
    if not prev_holdings:
        return sorted_ids[:pcfg.N_target]
    current = set(prev_holdings.keys())
    keep = [aid for aid in current if ranked.loc[aid, "pos"] <= pcfg.exit_rank] if len(ranked) else []
    needed = max(pcfg.N_min, min(pcfg.N_target, pcfg.N_max)) - len(keep)
    additions = [aid for aid in sorted_ids if (aid not in current) and (ranked.loc[aid, "pos"] <= pcfg.entry_rank)]
    selected = keep + additions[:max(0, needed)]
    i = 0
    while len(selected) < pcfg.N_min and i < len(sorted_ids):
        cand = sorted_ids[i]
        if cand not in selected: selected.append(cand)
        i += 1
    return selected[:pcfg.N_max]

def compute_equal_weights(selected: List[str], pcfg: PortfolioConfig) -> Dict[str, float]:
    if not selected: return {}
    w = 1.0 / float(len(selected))
    return {aid: w for aid in selected}

def apply_caps_and_floors(weights: Dict[str, float], pcfg: PortfolioConfig) -> Dict[str, float]:
    if not weights: return {}
    raw = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(raw.values())
    if s <= 0: return {}
    base = {k: v / s for k, v in raw.items()}
    cap = float(pcfg.weight_cap)
    w = {k: min(v, cap) for k, v in base.items()}
    feasible_sum = min(1.0, cap * len(w))
    # Water-fill up to feasible_sum
    def _sum(d): return sum(d.values())
    deficit = feasible_sum - _sum(w)
    while deficit > 1e-12:
        headroom = {k: cap - w[k] for k in w if w[k] < cap - 1e-12}
        if not headroom: break
        mass = sum(base[k] for k in headroom)
        if mass <= 0: break
        changed = False
        for k in list(headroom.keys()):
            add = deficit * (base[k] / mass)
            new_w = min(cap, w[k] + add)
            if new_w > w[k] + 1e-15: changed = True
            w[k] = new_w
        new_deficit = feasible_sum - _sum(w)
        if not changed or abs(new_deficit - deficit) < 1e-12: break
        deficit = new_deficit
    # Absolute EUR floor → drop names below floor
    floor_w = pcfg.notional_floor_eur / max(pcfg.portfolio_notional_eur, 1e-9)
    kept = {k: v for k, v in w.items() if v >= floor_w}
    if not kept:
        k_top = max(w, key=w.get)
        return {k_top: w[k_top]}
    # Try scale to feasible_sum again
    sum_kept = _sum(kept)
    target = min(1.0, cap * len(kept))
    if sum_kept > 0 and sum_kept <= target + 1e-12:
        scale = target / sum_kept
        kept = {k: min(v * scale, cap) for k, v in kept.items()}
    curr = _sum(kept)
    if curr + 1e-12 < target:
        deficit = target - curr
        while deficit > 1e-12:
            headroom = {k: cap - kept[k] for k in kept if kept[k] < cap - 1e-12}
            if not headroom: break
            mass = sum(base.get(k, 0.0) for k in headroom)
            changed = False
            if mass > 0:
                for k in list(headroom.keys()):
                    inc = min(deficit * (base.get(k,0.0)/mass), headroom[k])
                    if inc > 1e-15: kept[k] += inc; changed = True
            else:
                share = deficit / len(headroom)
                for k in list(headroom.keys()):
                    inc = min(share, headroom[k])
                    if inc > 1e-15: kept[k] += inc; changed = True
            new_deficit = target - _sum(kept)
            if not changed or abs(new_deficit - deficit) < 1e-12: break
            deficit = new_deficit
    return kept

# ----------------------------- IO helpers -----------------------------

def _load_price_csv(path: str) -> pd.DataFrame:
    """Return long DataFrame with columns: date, asset, close (types normalized)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for c in ("date", "asset", "close"):
        if c not in cols:
            raise ValueError(f"{path} must have columns date, asset, close (missing '{c}')")
    df = df[[cols["date"], cols["asset"], cols["close"]]].copy()
    df.columns = ["date", "asset", "close"]
    df["date"]  = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["asset"] = df["asset"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date","asset","close"]).sort_values(["asset","date"])
    return df

def _close_series_from_csv(df: pd.DataFrame, asset_symbol: str) -> pd.Series:
    sub = df[df["asset"] == asset_symbol]
    if sub.empty: return pd.Series(dtype=float)
    s = sub.set_index("date")["close"].sort_index()
    return s.asfreq("D").ffill()

# ----------------------------- Orchestrator ---------------------------

def run_model_c(
    ucfg: UniverseConfig,
    scfg: SignalConfig,
    pcfg: PortfolioConfig,
    prev_weights_path: Optional[str] = None,
    price_csv: Optional[str] = None,
    csv_key: str = "altname",      # how to match CSV's 'asset' with universe: 'altname'|'pair_key'|'wsname'
    offline: bool = False,
    sleep_sec: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weights_df, diagnostics_df). Index by pair_key."""
    session = _session_with_retries()

    # 1) Universe
    pairs = list_eur_pairs(session, ucfg)
    if not pairs:
        raise RuntimeError("No EUR pairs found after filters.")
    since_unix = int((datetime.now(timezone.utc) - timedelta(days=scfg.since_days)).timestamp())

    # Optional local panel
    price_df = _load_price_csv(price_csv) if price_csv else None

    # 2) Build per-asset stats (prefer local; optionally fall back to API)
    stats_rows: List[Dict[str, object]] = []
    for meta in pairs:
        pair_key = meta["pair_key"]
        wsname   = meta.get("wsname")
        altname  = meta.get("altname")
        symbol_to_match = {
            "altname": altname, "pair_key": pair_key, "wsname": wsname
        }.get(csv_key, altname)

        close_series = pd.Series(dtype=float)

        if price_df is not None and symbol_to_match:
            close_series = _close_series_from_csv(price_df, str(symbol_to_match))

        used_source = "csv" if not close_series.empty else "api"

        if close_series.empty:
            if offline:
                # Skip if strictly offline and not in CSV
                continue
            try:
                close_series = fetch_ohlc_daily(session, pair_key, since_unix)
            except Exception:
                continue  # skip problematic names

        if close_series.empty:
            continue

        st = compute_asset_stats_from_close(close_series, scfg.skip_days)
        # Basic quality filters
        if (not math.isnan(st["median_close"])) and (st["median_close"] < ucfg.dust_price_eur):
            continue

        row = {
            "pair_key": pair_key,
            "wsname": wsname,
            "altname": altname,
            "ok": True,
            "median_close": st["median_close"],
            "adv30_eur": np.nan,  # not available from local CSV; keep for diagnostics
            "r90": st["r90"],
            "r300": st["r300"],
            "len": st["len"],
            "source": used_source,
        }
        if st["len"] < (300 + scfg.skip_days + 1):
            row["limited_history"] = True
        stats_rows.append(row)
        time.sleep(sleep_sec if used_source == "api" else 0.0)

    if not stats_rows:
        raise RuntimeError("No eligible assets after building stats (check CSV coverage or relax filters).")

    stats = pd.DataFrame(stats_rows).set_index("pair_key")

    # 3) Ranks & composite
    ranked = rank_and_score(stats)

    # 4) Membership with buffers
    prev = None
    if prev_weights_path:
        try:
            wprev = pd.read_csv(prev_weights_path)
            if "pair_key" in wprev.columns and "weight" in wprev.columns:
                prev = {str(r["pair_key"]): float(r["weight"]) for _, r in wprev.iterrows()}
        except Exception:
            prev = None
    selected = apply_membership_with_buffers(ranked, pcfg, prev)

    # 5) Weights
    w0 = compute_equal_weights(selected, pcfg)
    w  = apply_caps_and_floors(w0, pcfg)

    # 6) Outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    weights_df = pd.DataFrame(
        [{"pair_key": k, "wsname": stats.loc[k, "wsname"], "weight": v} for k, v in w.items()]
    ).set_index("pair_key").sort_values("weight", ascending=False)

    diag = ranked.copy()
    diag["selected"] = diag.index.isin(selected)
    diag["weight"]   = diag.index.map(w).fillna(0.0)
    diag["version"]  = MODEL_C_VERSION
    diag["timestamp_utc"] = ts

    return weights_df, diag

# -------------------------------- CLI ---------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ModelC-v1.0 — Cross-Sectional Momentum on Kraken EUR pairs (cached or API)")

    # Universe
    ap.add_argument("--max-pairs", type=int, default=120)
    ap.add_argument("--include-inactive", action="store_true")
    ap.add_argument("--dust-price-eur", type=float, default=0.01)

    # Signals
    ap.add_argument("--since-days", type=int, default=400)
    ap.add_argument("--skip-days", type=int, default=7)

    # Portfolio
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--entry-rank", type=int, default=8)
    ap.add_argument("--exit-rank", type=int, default=15)
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--floor-eur", type=float, default=25.0)
    ap.add_argument("--notional-eur", type=float, default=1000.0)

    # Previous holdings
    ap.add_argument("--prev-weights", type=str, default=None)

    # Cached prices
    ap.add_argument("--price-csv", type=str, default=None, help="Long CSV: date,asset,close")
    ap.add_argument("--csv-key", type=str, default="altname", choices=["altname","pair_key","wsname"],
                    help="Which universe field your CSV 'asset' matches (default: altname)")
    ap.add_argument("--offline", action="store_true", help="Do not call Kraken; skip assets missing in CSV")

    # Outputs
    ap.add_argument("--out-weights", type=str, default=None)
    ap.add_argument("--out-diag", type=str, default=None)
    ap.add_argument("--sleep-sec", type=float, default=0.2, help="Throttle for API calls (ignored in offline or CSV hits)")

    args = ap.parse_args(argv)

    ucfg = UniverseConfig(include_inactive=args.include_inactive, max_pairs=args.max_pairs, dust_price_eur=args.dust_price_eur)
    scfg = SignalConfig(lookbacks=(90,300), skip_days=args.skip_days, since_days=args.since_days)
    pcfg = PortfolioConfig(
        N_target=args.N, N_min=max(1,args.N-2), N_max=args.N+2,
        entry_rank=args.entry_rank, exit_rank=args.exit_rank,
        weight_cap=args.cap, notional_floor_eur=args.floor_eur, portfolio_notional_eur=args.notional_eur,
    )

    try:
        weights_df, diag = run_model_c(
            ucfg, scfg, pcfg,
            prev_weights_path=args.prev_weights,
            price_csv=args.price_csv,
            csv_key=args.csv_key,
            offline=bool(args.offline),
            sleep_sec=args.sleep_sec,
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_w = args.out_weights or f"modelC_weights_EUR_{ts}.csv"
        out_d = args.out_diag    or f"modelC_diagnostics_EUR_{ts}.csv"
        weights_df.to_csv(out_w, index_label="pair_key")
        diag.to_csv(out_d, index_label="pair_key")
        print(f"Saved weights to {out_w} and diagnostics to {out_d} — version {MODEL_C_VERSION}")
        return 0
    except Exception as e:
        print(f"Model C failed: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
