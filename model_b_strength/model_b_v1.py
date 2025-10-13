# model_b_v1.py  (v1.1)
from __future__ import annotations
import pandas as pd
import numpy as np

class BParams:
    def __init__(
        self,
        er_window=20,
        enter_strong=0.30, exit_strong=0.25,
        enter_weak=0.20,  exit_weak=0.25,
        dwell_days=3,
        smooth_ema_days=0,   # NEW
    ):
        self.er_window = int(er_window)
        self.enter_strong = float(enter_strong)
        self.exit_strong  = float(exit_strong)
        self.enter_weak   = float(enter_weak)
        self.exit_weak    = float(exit_weak)
        self.dwell_days   = int(dwell_days)
        self.smooth_ema_days = int(smooth_ema_days)

def _efficiency_ratio(s, w):
    # Perry Kaufman ER: |price_t - price_{t-w}| / sum(|delta_i|)
    px = s.ffill()
    ch = px.diff().abs()
    num = (px - px.shift(w)).abs()
    den = ch.rolling(w).sum()
    er = (num / (den + 1e-12)).clip(0,1)
    return er

def compute_exposure_B(price_series_dict: dict[str, pd.Series], p: BParams) -> pd.DataFrame:
    """
    price_series_dict: {'XBTEUR': Series, 'ETHEUR': Series, ...} (barometer basket)
    Returns DataFrame ['date','exp_B'] in [0,1]
    """
    # equal-weight ER across provided series
    ers = []
    for sym, ser in price_series_dict.items():
        ers.append(_efficiency_ratio(ser.sort_index(), p.er_window).rename(sym))
    er_df = pd.concat(ers, axis=1).mean(axis=1).rename("er").fillna(0.0)

    # 3-state like A but on ER thresholds
    state = []
    cur = 0.5
    dwell = 0
    for dt, val in er_df.items():
        if dwell > 0:
            dwell -= 1
            state.append(cur)
            continue
        nxt = cur
        if cur >= 0.75:
            if val < p.exit_strong:
                nxt = 0.5; dwell = p.dwell_days
        elif cur <= 0.25:
            if val > p.exit_weak:
                nxt = 0.5; dwell = p.dwell_days
        else:
            if val >= p.enter_strong:
                nxt = 1.0; dwell = p.dwell_days
            elif val <= p.enter_weak:
                nxt = 0.0; dwell = p.dwell_days
        cur = nxt
        state.append(cur)

    exp = pd.Series(state, index=er_df.index, name="exp_B")
    if p.smooth_ema_days and p.smooth_ema_days > 0:
        exp = exp.ewm(span=p.smooth_ema_days, min_periods=1, adjust=False).mean()
    out = exp.clip(0,1).reset_index().rename(columns={"index":"date"})
    return out
