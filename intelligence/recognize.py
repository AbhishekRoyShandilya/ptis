"""
intelligence.recognize — L1 RECOGNITION (the peripheral-vision pass).

The recognition/understanding split (the scaling key, [[project_day_story_substrate]]): the full eye read
(`cognition_service.read`) is ~10s narration + ~5s LLM/symbol, so a live scan of a whole watchlist is
impractical — the current `watchlist_scan` is forced to bound to 5 symbols + cache. But the QUESTION "is a
money-move building here?" does NOT need the whole understanding — it needs only the BirthTracker's cheap,
already-incremental recognition (birth? held? how far extended?).

COST (measured): the recognition COMPUTE (compute_features + feed the tracker) is 0.09s/symbol = ~119x
cheaper than the ~11s narration. End-to-end COLD it is ~0.8s/symbol because the lake candle-read (~0.7s,
which the full read ALSO pays) dominates → still ~18x cheaper than the full read, and a whole watchlist
becomes feasible. WARM (live feed keeps candles in memory, or `candles=` supplied) it drops back to ~0.1s.

So: run `recognize` on EVERY symbol every scan (peripheral vision) → run the expensive
`cognition_service.read` UNDERSTANDING only on the few that light up (focal attention). Same BirthTracker
the full eye uses → bit-consistent births; NO read_move/read_balance/read_bias/reasoner here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from intelligence.staged import CONFIRM_EXT, STALL_BARS

# ── the ATTENTION taxonomy: recognition emits a CLASS, not a boolean (the event pipeline subscribes to
#    these). NONE/POTENTIAL/ACCEPTANCE/MONEY_MOVE/EXHAUSTION are what the cheap tracker can SEE on one
#    symbol; FAILURE is a TRANSITION (money_move/acceptance → none, derived by the event differ),
#    POSITION_RISK is the same read overlaid on a HELD position, STRATEGY_MATCH is the taught-strategy
#    signature firing — those live in the event/overlay layers, not here. ──
ATTENTION_NONE = "ATTENTION_NONE"
ATTENTION_POTENTIAL = "ATTENTION_POTENTIAL"      # a candidate appeared (birth) — not proven yet
ATTENTION_ACCEPTANCE = "ATTENTION_ACCEPTANCE"    # held + extending, proving it is moving (pre-money-move)
ATTENTION_MONEY_MOVE = "ATTENTION_MONEY_MOVE"    # held + extended >= CONFIRM_EXT — a real move underway
ATTENTION_EXHAUSTION = "ATTENTION_EXHAUSTION"    # a money-move that has stalled out (rolling over)
ATTENTION_FAILURE = "ATTENTION_FAILURE"          # (event-layer) a move died before proving — see detect_events
ATTENTION_POSITION_RISK = "ATTENTION_POSITION_RISK"
ATTENTION_STRATEGY_MATCH = "ATTENTION_STRATEGY_MATCH"

_ATTENTION_RANK = {ATTENTION_NONE: 0, ATTENTION_POTENTIAL: 1, ATTENTION_ACCEPTANCE: 2,
                   ATTENTION_EXHAUSTION: 3, ATTENTION_MONEY_MOVE: 4}
_ACCEPT_EXT = 1.0                                # held + extended this far = ACCEPTANCE (past a mere poke)


@dataclass
class Recognition:
    symbol: str
    ok: bool = True
    attention: str = ATTENTION_NONE   # the attention CLASS (the event the pipeline subscribes to)
    active: bool = False          # a move is alive right now (price holds beyond its birth level)
    direction: str = ""           # up / down (of the active move)
    ext_atr: float = 0.0          # extension-from-birth in ATRs — the money-move confidence
    money_move: bool = False      # active AND extended >= CONFIRM_EXT = a confirmed money-move (staged)
    stalled_bars: int = 0         # bars since the move last made new ground (for EXHAUSTION)
    n_births: int = 0             # strong-break births this session (activity level)
    last_kind: str = ""           # last event kind (BIRTH / REBIRTH / CONTINUATION)
    price: float = float("nan")
    as_of: str = ""
    note: str = ""

    @property
    def rank(self) -> int:
        return _ATTENTION_RANK.get(self.attention, 0)

    @property
    def worth_understanding(self) -> bool:
        """The attention gate: pay for the full eye read ONLY when recognition sees a real move (>= ACCEPTANCE)."""
        return self.rank >= _ATTENTION_RANK[ATTENTION_ACCEPTANCE]


def _classify(active: bool, ext: float, stalled: int, n_births: int) -> str:
    """The attention class from the tracker's cheap state — held/extension/stall (no read_move)."""
    if active and ext >= CONFIRM_EXT:
        return ATTENTION_EXHAUSTION if stalled > 2 * STALL_BARS else ATTENTION_MONEY_MOVE
    if active and ext >= _ACCEPT_EXT:
        return ATTENTION_ACCEPTANCE
    if active or n_births > 0:
        return ATTENTION_POTENTIAL
    return ATTENTION_NONE


def recognize(symbol: str, interval: str = "5m", as_of=None, candles: pd.DataFrame | None = None) -> Recognition:
    """Cheap point-in-time recognition of the move active at `as_of` — feed the BirthTracker over the
    session and read its final state. Reuses the same tracker/features as the full eye (bit-consistent)."""
    from marketdata import features as F
    from intelligence.birth import BirthTracker, BIRTH, REBIRTH

    now = pd.Timestamp(as_of) if as_of is not None else None
    day = (now or pd.Timestamp.now()).normalize()
    if candles is None:
        if now is None:                              # LIVE: use the warm per-symbol cache (loop-friendly)
            from intelligence.warm_candles import warm_candles
            candles = warm_candles(symbol, interval)
        else:                                        # historical/point-in-time: load the exact window
            from marketdata.candle_store import get_candle_store
            candles = get_candle_store().get(symbol, interval, day - pd.Timedelta(days=10),
                                             day + pd.Timedelta(days=1), live=False)
    if candles is None or len(candles) == 0:
        return Recognition(symbol=symbol, ok=False, note="no candles")
    c = candles.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(c["ts"])
    if now is not None:                              # point-in-time: only bars up to as_of
        m = (ts <= now).to_numpy()
        c = c[m].reset_index(drop=True); ts = ts[m].reset_index(drop=True)
    if len(c) < 40:
        return Recognition(symbol=symbol, ok=False, note="not enough history")
    feat = F.compute_features(c, symbol, interval)
    atr = feat["atr"].to_numpy(float)
    rvol = feat["rvol"].to_numpy(float) if "rvol" in feat else np.full(len(c), np.nan)
    h = c["high"].to_numpy(float); l = c["low"].to_numpy(float)
    o = c["open"].to_numpy(float); cl = c["close"].to_numpy(float); vol = c["volume"].to_numpy(float)
    day_d = ts.dt.date.to_numpy()[-1]               # the session of the as_of (last) bar
    sess = np.where(ts.dt.date.to_numpy() == day_d)[0]
    if len(sess) == 0:
        return Recognition(symbol=symbol, ok=False, note="no session bars")
    tr = BirthTracker()
    n_births = 0; last_kind = ""
    last_ext = 0.0; stalled = 0            # bars since the active move last made new ground (for EXHAUSTION)
    for k in range(len(sess)):
        i = int(sess[k])
        ev = tr.update(i, ts.iloc[i], h, l, o, cl, vol, atr, rvol)
        if ev is not None:
            last_kind = ev.kind
            if ev.kind in (BIRTH, REBIRTH):
                n_births += 1
        if tr.move_active:
            e = tr.ext_atr()
            if e > last_ext + 0.05:
                last_ext = e; stalled = 0
            else:
                stalled += 1
        else:
            last_ext = 0.0; stalled = 0
    ext = tr.ext_atr()
    attention = _classify(bool(tr.move_active), ext, stalled, n_births)
    return Recognition(
        symbol=symbol, attention=attention, active=bool(tr.move_active),
        direction=tr.mdir if tr.move_active else "",
        ext_atr=round(float(ext), 2), money_move=(attention == ATTENTION_MONEY_MOVE),
        stalled_bars=int(stalled), n_births=n_births, last_kind=last_kind,
        price=round(float(cl[-1]), 2), as_of=f"{ts.iloc[-1]:%Y-%m-%d %H:%M}",
    )


def recognize_many(symbols: list[str], interval: str = "5m", as_of=None) -> list[Recognition]:
    """Peripheral vision over a whole watchlist — cheap recognition on EVERY symbol, most-built first
    (money-moves, then active, then by extension). The caller runs the full understanding on the top few."""
    out: list[Recognition] = []
    for s in symbols:
        try:
            out.append(recognize(s, interval, as_of))
        except Exception:
            out.append(Recognition(symbol=s, ok=False, note="recognize failed"))
    out.sort(key=lambda r: (r.rank, r.ext_atr), reverse=True)
    return out
