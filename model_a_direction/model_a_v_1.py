# model_a_v_1.py  (v1.1)
from __future__ import annotations
import pandas as pd
import numpy as np

class AParams:
    def __init__(
        self,
        lookbacks=(120, 180, 300),
        skip_days=7,
        enter_up_z=0.25, exit_up_z=0.10,
        enter_down_z=-0.25, exit_down_z=-0.10,
        dwell_days=5,
        smooth_ema_days=0,     # NEW: 0 = off
    ):
        self.lookbacks = tuple(lookbacks)
        self.skip_days = int(skip_days)
        self.enter_up_z = float(enter_up_z)
        self.exit_up_z = float(exit_up_z)
        self.enter_down_z = float(enter_down_z)
        self.exit_down_z = float(exit_down_z)
        self.dwell_days = int(dwell_days)
        self.smooth_ema_days = int(smooth_ema_days)

def _zscore(s, lb):
    r = s.pct_change(lb)
    return (r - r.rolling(lb).mean()) / (r.rolling(lb).std(ddof=1) + 1e-12)

def compute_exposure_A(price_series: pd.Series, p: AParams) -> pd.DataFrame:
    """
    Input: price_series (daily close) indexed by date
    Output: DataFrame with columns ['date','exp_A'] in [0,1]
    """
    s = price_series.sort_index().ffill()
    zs = [ _zscore(s, lb) for lb in p.lookbacks ]
    z = pd.concat(zs, axis=1).mean(axis=1).rename("z")
    z = z.shift(p.skip_days)

    # 3-state with dwell: up (1), range (0.5), down (0)
    state = []
    cur = 0.5
    dwell = 0
    for dt, val in z.items():
        if dwell > 0:
            dwell -= 1
            state.append(cur)
            continue
        nxt = cur
        if cur >= 0.75:  # was up
            if val < p.exit_up_z:
                nxt = 0.5
                dwell = p.dwell_days
        elif cur <= 0.25:  # was down
            if val > p.exit_down_z:
                nxt = 0.5
                dwell = p.dwell_days
        else:  # was range
            if val >= p.enter_up_z:
                nxt = 1.0
                dwell = p.dwell_days
            elif val <= p.enter_down_z:
                nxt = 0.0
                dwell = p.dwell_days
        cur = nxt
        state.append(cur)

    exp = pd.Series(state, index=z.index, name="exp_A")

    if p.smooth_ema_days and p.smooth_ema_days > 0:
        exp = exp.ewm(span=p.smooth_ema_days, min_periods=1, adjust=False).mean()

    out = exp.clip(0,1).reset_index().rename(columns={"index":"date"})
    return out
