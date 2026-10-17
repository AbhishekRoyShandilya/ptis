"""
intelligence.birth — the BIRTH recognizer (an observer organ): "has a move actually been BORN yet?"

The entry-edge transition, read by OBSERVATION (not a bar-count, not a probability forecast):
  STRONG-BREAK = price BROKE a prevailing structure (closed beyond it) WITH STRENGTH — a candle whose
    body is large vs the recent weak range (the CONTRAST a trader reads) that closed near its extreme
    (low wick) — AND PARTICIPATION (volume picked up; for indices with no volume, range/ATR expansion
    stands in). A single decisive candle can qualify; a 5-bar drift that pokes and returns cannot.
  PROBE = broke but weak / wicked back / no participation (an attempt, not a birth).

A strong break is then read IN THE CONTEXT of an ongoing move by the BirthTracker state machine:
  · from balance (no active move, or the opposite direction) → BIRTH  (REBIRTH if the SAME direction
    starts fresh again after the prior move was released)
  · while a same-direction move is still active             → CONTINUATION (a late leg, NOT a birth)
RELEASE is STRUCTURAL, never a depth/bar count: a move is alive as long as price HOLDS beyond the
level it broke to be born; it is released only when a close round-trips back through that birth level
(the breakout failed) — or an opposite-direction strong break takes control.

Optional level CONTEXT (the caller supplies it): a strong break running INTO stacked overhead
structure (small clear-air) is an attempt at a hurdle, not yet a confirmable birth — the classic
"fast open into resistance that pauses, then breaks clean an hour later." Pure price-structure here;
the hurdle distance comes from the eye's level gatherer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BIRTH_VERSION = "1.0"

LOOKBACK = 20            # the prevailing structure (range) we ask whether price broke + held
RECENT = 15             # window for the strength CONTRAST (body/range vs the recent weak candles)

# event kinds
BIRTH, REBIRTH, CONTINUATION = "BIRTH", "REBIRTH", "CONTINUATION"


def birth_state(h, l, o, cl, vol, atr, rvol, i):
    """Per-bar read → ('NO-MOVE' | 'PROBE' | 'STRONG-BREAK', direction, metrics).

    STRONG-BREAK is a strength-led acceptance candidate; the BirthTracker decides birth vs
    continuation. Participation CONFIRMS but does not veto when the body is already clearly strong."""
    if i < LOOKBACK + RECENT:
        return 'NO-MOVE', '', {}
    a = atr[i]
    if not (a and a > 0):
        return 'NO-MOVE', '', {}
    rhigh = np.nanmax(h[i - LOOKBACK:i]); rlow = np.nanmin(l[i - LOOKBACK:i])
    body = abs(cl[i] - o[i]); rng = max(h[i] - l[i], 1e-9)
    rbody = np.nanmedian(np.abs(cl[i - RECENT:i] - o[i - RECENT:i])) or 1e-9
    rrng = np.nanmedian(h[i - RECENT:i] - l[i - RECENT:i]) or 1e-9
    body_ratio = body / rbody; rng_ratio = rng / rrng
    has_vol = np.nanmedian(vol[i - RECENT:i]) > 0
    up_break = cl[i] > rhigh + 0.1 * a
    dn_break = cl[i] < rlow - 0.1 * a
    R = dict(body_x=round(body_ratio, 1), rng_x=round(rng_ratio, 1),
             rvol=round(float(rvol[i]), 1) if has_vol else None,
             lvl=round(rhigh if up_break else rlow, 2))
    if up_break or dn_break:
        d = 'up' if up_break else 'down'
        close_pos = (cl[i] - l[i]) / rng if up_break else (h[i] - cl[i]) / rng   # 1 = closed at the extreme
        R['close_pos'] = round(close_pos, 2)
        strong = body_ratio >= 1.8 and close_pos >= 0.6
        carry = (rng_ratio >= 1.5) or (body_ratio >= 3.0) or (has_vol and rvol[i] >= 1.8)
        if strong and carry:
            return 'STRONG-BREAK', d, R
        return 'PROBE', d, R
    # wicked beyond structure then closed back inside = a rejected probe
    if h[i] > rhigh and cl[i] <= rhigh:
        return 'PROBE', 'up', {**R, 'note': 'wick-rejected'}
    if l[i] < rlow and cl[i] >= rlow:
        return 'PROBE', 'down', {**R, 'note': 'wick-rejected'}
    return 'NO-MOVE', '', {}


@dataclass
class BirthEvent:
    i: int
    ts: pd.Timestamp
    kind: str                       # BIRTH | REBIRTH | CONTINUATION
    direction: str                  # up | down
    level: float                    # the structure level it broke (also the move's release level)
    metrics: dict = field(default_factory=dict)
    into_hurdle: bool = False       # strong break running into stacked overhead structure (set by caller)
    note: str = ""


class BirthTracker:
    """Stateful reader: feed it bars in order; it emits a BirthEvent on each STRONG-BREAK, classified
    as BIRTH / REBIRTH / CONTINUATION, with structural release in between. No bar-counts, no ATR
    pullback thresholds — a move lives while price holds beyond its birth level."""

    def __init__(self):
        self.move_active = False
        self.mdir = ''
        self.morigin = np.nan       # the level the move broke to be born (also its release level)
        self.mext = np.nan          # the move's running extreme (high for up, low for down)
        self.mbirth_atr = np.nan    # ATR at birth — a stable denominator for extension-from-birth
        self.last_dir = ''

    def _release_check(self, close: float) -> None:
        if not self.move_active:
            return
        if (self.mdir == 'up' and close < self.morigin) or (self.mdir == 'down' and close > self.morigin):
            self.move_active = False              # breakout round-tripped → back to balance

    def ext_atr(self) -> float:
        """How far the move has run from its birth level to its extreme, in birth-ATRs. Measured from
        the BIRTH (not a local leg) so maturity grows monotonically over the move — no reset-to-early
        on pullbacks. 0 when no move is active."""
        if not self.move_active or not (self.mbirth_atr and self.mbirth_atr > 0):
            return 0.0
        return abs(self.mext - self.morigin) / self.mbirth_atr

    def update(self, i, ts, h, l, o, cl, vol, atr, rvol, clear_air_atr: float | None = None):
        """Advance one bar. Returns a BirthEvent on a strong break, else None. `clear_air_atr` (ATRs
        of room beyond the break before the nearest opposing level) lets the caller flag a break that
        runs INTO a hurdle — supply None when level context isn't available."""
        self._release_check(cl[i])
        if self.move_active:                      # track the move's extreme each bar
            self.mext = np.nanmax([self.mext, h[i]]) if self.mdir == 'up' else np.nanmin([self.mext, l[i]])
        st, d, R = birth_state(h, l, o, cl, vol, atr, rvol, i)
        if st != 'STRONG-BREAK':
            return None
        if self.move_active and d == self.mdir:
            kind = CONTINUATION
        else:
            kind = REBIRTH if (not self.move_active and d == self.last_dir) else BIRTH
            self.move_active = True
            self.mdir = d
            self.last_dir = d
            self.morigin = R.get('lvl')
            self.mext = h[i] if d == 'up' else l[i]
            self.mbirth_atr = atr[i]
        into_hurdle = bool(clear_air_atr is not None and clear_air_atr < 1.0)
        note = "into a stacked hurdle — attempt, expect a pause" if into_hurdle else ""
        return BirthEvent(i=i, ts=ts, kind=kind, direction=d, level=R.get('lvl'),
                          metrics=R, into_hurdle=into_hurdle, note=note)


def read_births(symbol: str, interval: str, date, candles: pd.DataFrame | None = None) -> list[BirthEvent]:
    """Walk one session point-in-time and return its BIRTH/REBIRTH/CONTINUATION events (pure
    price-structure; no level context). Convenience for research + tests."""
    from marketdata import features as F
    day = pd.Timestamp(date).normalize()
    if candles is None:
        from marketdata.candle_store import get_candle_store
        candles = get_candle_store().get(symbol, interval, day - pd.Timedelta(days=8),
                                         day + pd.Timedelta(days=1))
    c = candles.sort_values('ts').reset_index(drop=True)
    ts = pd.to_datetime(c['ts'])
    feat = F.compute_features(c, symbol, interval)
    atr = feat['atr'].to_numpy(float); rvol = feat['rvol'].to_numpy(float)
    h = c['high'].to_numpy(float); l = c['low'].to_numpy(float)
    o = c['open'].to_numpy(float); cl = c['close'].to_numpy(float); vol = c['volume'].to_numpy(float)
    sess = np.where(ts.dt.date.to_numpy() == day.date())[0]
    tracker = BirthTracker()
    events: list[BirthEvent] = []
    for i in sess:
        ev = tracker.update(int(i), ts.iloc[int(i)], h, l, o, cl, vol, atr, rvol)
        if ev is not None:
            events.append(ev)
    return events
