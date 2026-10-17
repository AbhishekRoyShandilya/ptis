"""
intelligence.session_clock — the intraday SESSION clock (IST), first-class.

The discovery/story/confluence engines were time-blind. For intraday, time-of-day is
not a flag — it is structure: the SAME pattern means different things at 9:30 (open
drive), 11:30 (midday drift) and 15:15 (EOD / BTST positioning). This module gives
every bar its place in the day so nothing is ever pooled across session phases.

NSE/BSE cash session: 09:15–15:30 IST. Phases (frozen, definitional — not tuned):
  OPENING    09:15–09:45   first 30m — auction unwind, open drive, gap resolution
  MORNING    09:45–11:00   trend establishment
  MIDDAY     11:00–13:00   lunch drift / lowest participation
  AFTERNOON  13:00–14:45   second-wind moves
  CLOSING    14:45–15:30   EOD positioning; the BTST/EOD window lives here (~15:00+)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_CLOCK_VERSION = "1.0"

OPEN_MIN = 9 * 60 + 15      # 555
CLOSE_MIN = 15 * 60 + 30    # 930
BTST_MIN = 15 * 60          # 900  — last 30m, EOD/BTST positioning

PHASES = ["OPENING", "MORNING", "MIDDAY", "AFTERNOON", "CLOSING"]
_BOUNDS = [(555, 585, "OPENING"), (585, 660, "MORNING"), (660, 780, "MIDDAY"),
           (780, 885, "AFTERNOON"), (885, 931, "CLOSING")]


def _mins(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    return ((ts - ts.dt.normalize()).dt.total_seconds() / 60.0).to_numpy()


def session_phase(ts) -> str:
    m = _mins(pd.Series([pd.Timestamp(ts)]))[0]
    for lo, hi, name in _BOUNDS:
        if lo <= m < hi:
            return name
    return "OFF"


def add_session_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Per-bar session features (vectorized, point-in-time):
    phase, mins_from_open, mins_to_close, bars_since_open (per day), and the
    OPENING / BTST window flags. Returns a copy with columns added."""
    c = candles.copy()
    ts = pd.to_datetime(c["ts"])
    m = _mins(ts)
    phase = np.full(len(c), "OFF", dtype=object)
    for lo, hi, name in _BOUNDS:
        phase[(m >= lo) & (m < hi)] = name
    c["phase"] = phase
    c["mins_from_open"] = np.clip(m - OPEN_MIN, 0, None)
    c["mins_to_close"] = np.clip(CLOSE_MIN - m, 0, None)
    c["is_opening"] = (m >= OPEN_MIN) & (m < OPEN_MIN + 30)
    c["is_btst_window"] = m >= BTST_MIN
    c["bars_since_open"] = c.groupby(ts.dt.date).cumcount()
    return c


def phase_of(ts_series: pd.Series) -> pd.Series:
    m = _mins(ts_series)
    out = np.full(len(m), "OFF", dtype=object)
    for lo, hi, name in _BOUNDS:
        out[(m >= lo) & (m < hi)] = name
    return pd.Series(out, index=ts_series.index)
