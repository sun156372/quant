# strategy.py -- the frozen M1 signal. No exchange code, no sizing, no state.
"""Pure functions: candles -> features -> probability -> signal.

The feature definitions MUST stay identical to mvp/features.py in the research
repo. Any divergence invalidates the backtest that produced the parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["ret_1", "vol_24", "rsi_14", "vol_z", "trend"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("open_time").reset_index(drop=True)
    c, v = d["close"], d["volume"]
    r = np.log(c).diff()

    d["ret_1"] = c.pct_change(1)
    d["vol_24"] = r.rolling(24).std() * np.sqrt(24 * 365)
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["vol_z"] = (v - v.rolling(48).mean()) / v.rolling(48).std()
    d["trend"] = c / c.ewm(span=48, adjust=False).mean() - 1
    return d


def latest_signal(df: pd.DataFrame, model, entry_prob: float) -> dict | None:
    """Score the LAST COMPLETED bar. Returns None when features are not ready."""
    d = add_features(df)
    if len(d) == 0:
        return None
    last = d.iloc[-1]
    if not np.isfinite(last[FEATURES].to_numpy(dtype=float)).all():
        return None
    p = float(model.predict(d[FEATURES].tail(1).to_numpy(dtype=float))[0])
    return {
        "open_time": int(last["open_time"]),
        "close": float(last["close"]),
        "prob": p,
        "signal": bool(p >= entry_prob),
        "vol_24": float(last["vol_24"]),
    }


def exit_due(entry_time_ms: int, now_ms: int, hold_bars: int,
             bar_ms: int = 3_600_000) -> bool:
    return now_ms >= entry_time_ms + hold_bars * bar_ms
