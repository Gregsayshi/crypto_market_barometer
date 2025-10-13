# model_c_v1.py  (v1.1)
from __future__ import annotations
import pandas as pd
import numpy as np

class CParams:
    def __init__(
        self,
        mode="equal_weight",   # 'equal_weight' | 'light_momo' | 'cap_weight' | 'vol_target'
        N=10,
        cap=0.25,
        floor_eur=250.0,
        portfolio_eur=10000.0,
        skip_days=3,
        rebalance="4W-SUN",
        hysteresis=3,          # NEW: sell if rank > N + hysteresis
        momo_lb=60,            # used if mode == 'light_momo'
    ):
        self.mode = str(mode)
        self.N = int(N)
        self.cap = float(cap)
        self.floor_eur = float(floor_eur)
        self.portfolio_eur = float(portfolio_eur)
        self.skip_days = int(skip_days)
        self.rebalance = str(rebalance)
        self.hysteresis = int(hysteresis)
        self.momo_lb = int(momo_lb)

def _rank_momo(wide: pd.DataFrame, lb: int) -> pd.DataFrame:
    ret = wide.pct_change(lb)
    return ret

def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.clip(0, cap)
    s = w.sum()
    return w / s if s > 0 else w

def _ew_allocator(eligible_cols: list[str]) -> pd.Series:
    if not eligible_cols:
        return pd.Series(dtype=float)
    w = pd.Series(1.0/len(eligible_cols), index=eligible_cols)
    return w

def _select_with_hysteresis(ranks_today: pd.Series, prev_hold: set[str], N: int, hysteresis: int) -> list[str]:
    # buy if rank <= N; keep existing until rank > N + hysteresis
    keep = [a for a in prev_hold if ranks_today.get(a, np.inf) <= N + hysteresis]
    missing = N - len(keep)
    if missing > 0:
        candidates = ranks_today.sort_values().index.tolist()
        for a in candidates:
            if a not in keep and ranks_today[a] <= N:
                keep.append(a)
                if len(keep) >= N:
                    break
    return keep[:N]

def build_weights_history(universe_csv: str, p: CParams, start: str|None=None, end: str|None=None) -> pd.DataFrame:
    """
    Input: long CSV (date,asset,close[,...]) for the filtered universe
    Output: DataFrame with columns ['date','asset','weight'] (cross-section only; sums to 1 when invested)
    """
    long_df = pd.read_csv(universe_csv, parse_dates=["date"])
    if start:
        long_df = long_df[long_df["date"] >= pd.to_datetime(start)]
    if end:
        long_df = long_df[long_df["date"] <= pd.to_datetime(end)]
    long_df = long_df.sort_values(["date","asset"])
    wide = long_df.pivot_table(index="date", columns="asset", values="close").sort_index().ffill()

    # Rebalance calendar
    cal = wide.index.to_series().asfreq("D").index  # daily calendar
    if p.rebalance:
        rb = wide.resample(p.rebalance).last().index
    else:
        rb = wide.index

    prev_hold: set[str] = set()
    rows = []
    for dt in rb:
        if dt not in wide.index:
            # align to previous available day
            ix = wide.index[wide.index.get_indexer([dt], method="ffill")]
            if len(ix) == 0:
                continue
            dt_eff = ix[0]
        else:
            dt_eff = dt

        px = wide.loc[dt_eff]
        eligible = px.dropna().index.tolist()

        if p.mode == "light_momo":
            score = _rank_momo(wide.loc[:dt_eff], p.momo_lb).iloc[-1].dropna()
            # lower rank = better (descending return)
            ranks = (-score).rank(method="first")
        else:
            # equal-weight ranks (arbitrary stable order)
            ranks = pd.Series(np.arange(1, len(eligible)+1), index=eligible)

        sel = _select_with_hysteresis(ranks, prev_hold, p.N, p.hysteresis)
        prev_hold = set(sel)

        if p.mode == "equal_weight" or p.mode == "light_momo":
            w = _ew_allocator(sel)
        else:
            w = _ew_allocator(sel)  # keep simple for v1.1

        w = _apply_cap(w, p.cap)
        # floor is enforced at the exposure level in backtester via portfolio size; keep CS weights normalized here
        for a, v in w.items():
            rows.append({"date": dt_eff, "asset": a, "weight": float(v)})

    out = pd.DataFrame(rows).sort_values(["date","asset"]).reset_index(drop=True)
    return out
