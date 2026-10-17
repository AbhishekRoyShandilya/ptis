"""
intelligence.avt_trader — AVT (the autonomous face) ON THE BRAIN: the operating process, emotion-free.

The second face over the same eye. It does NOT predict — it OPERATES the brain's recognition with
discipline:
  · SELECTIVITY — it only participates when the brain RECOGNISES a money move (the staged Candidate →
    Money-Move confirmation). It sits out the 61% that never confirm.
  · STRUCTURAL RISK — the stop is the move's BIRTH LEVEL (where the move would be proven wrong), NOT a
    fixed ATR/-1R. Risk = |entry − birth level|.
  · ASYMMETRIC EXIT — it does NOT cap the winner at +1.5R. It HOLDS while the move is alive and exits
    when the STORY CHANGES (structural release) or at session close. Let the right tail run.

`simulate_session()` replays one session through the brain and returns the virtual trades + their R.
This is both the AVT face AND the plan's "operating-process edge" experiment (structural risk +
exit-when-story-changes → measure cumulative R), run on the validated recognition rather than a rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class VirtualTrade:
    symbol: str
    date: str
    direction: str
    entry_ts: str
    entry: float
    stop: float                  # the move's birth level (structural invalidation)
    exit_ts: str = ""
    exit: float = float("nan")
    exit_reason: str = ""        # story-changed / session-close
    R: float = float("nan")      # realised R (signed, vs |entry-stop|)
    mfe_R: float = float("nan")  # best favorable excursion in R (how much was on the table)
    bars_held: int = 0
    entry_tags: list = field(default_factory=list)   # the eye's tags AT ENTRY → attribution (phase/room/effort/…)

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_session(symbol: str, interval: str = "5m", date=None, candles=None,
                     strategy_filter=None) -> dict:
    """Replay one session through the brain and operate it: enter on the staged MONEY-MOVE confirmation,
    risk to the move's birth level, exit on STORY-CHANGED (structural release) or session close. Returns
    {trades, n, wins, total_R, avg_R, expectancy}. `strategy_filter(direction, tags) -> bool` (optional)
    restricts entries to a taught strategy's signature; default = take any recognised money move."""
    import pandas as pd
    from intelligence.state_narrator import narrate_session
    from intelligence.staged import stage_session, CANDIDATE, MONEY_MOVE, STORY_CHANGED

    date = pd.Timestamp(date).normalize() if date is not None else pd.Timestamp.today().normalize()
    steps = narrate_session(symbol, interval, date, candles=candles, with_recall=False)
    if not steps:
        return {"trades": [], "n": 0, "wins": 0, "total_R": 0.0, "avg_R": 0.0, "expectancy": 0.0}
    stages = stage_session(steps)
    cand_at = {s.ts: s for s in stages if s.kind == CANDIDATE}
    mm_at = {s.ts: s for s in stages if s.kind == MONEY_MOVE}
    sc_at = {s.ts: s for s in stages if s.kind == STORY_CHANGED}

    trades: list[VirtualTrade] = []
    open_t = None
    birth_px = None
    entry_i = 0
    for k, s in enumerate(steps):
        if s.ts in cand_at:
            birth_px = s.price                       # the candidate's break level = structural risk anchor
        if open_t is None and s.ts in mm_at:
            mm = mm_at[s.ts]
            if strategy_filter is not None and not strategy_filter(mm.direction, mm.tags):
                continue                                  # not this strategy's setup — sit out
            d = mm.direction
            stop = birth_px if (birth_px is not None and birth_px != s.price) else (
                s.price + (1 if d == "down" else -1) * (s.atr or 0) * 2)   # fallback: ~2 ATR structural
            open_t = VirtualTrade(symbol=symbol, date=str(date.date()), direction=d,
                                  entry_ts=f"{s.ts:%H:%M}", entry=s.price, stop=stop,
                                  entry_tags=list(mm.tags))
            open_t.peak = s.price
            entry_i = k
            continue
        if open_t is not None:
            down = open_t.direction == "down"
            open_t.peak = min(open_t.peak, s.price) if down else max(open_t.peak, s.price)
            last = k == len(steps) - 1
            if s.ts in sc_at or last:
                risk = abs(open_t.entry - open_t.stop) or (s.atr or 1e-9)
                gain = (open_t.entry - s.price) if down else (s.price - open_t.entry)
                mfe = (open_t.entry - open_t.peak) if down else (open_t.peak - open_t.entry)
                open_t.exit_ts = f"{s.ts:%H:%M}"; open_t.exit = s.price
                open_t.exit_reason = "story-changed" if s.ts in sc_at else "session-close"
                open_t.R = round(gain / risk, 2); open_t.mfe_R = round(mfe / risk, 2)
                open_t.bars_held = k - entry_i
                trades.append(open_t); open_t = None

    n = len(trades)
    total_R = round(sum(t.R for t in trades), 2)
    wins = sum(1 for t in trades if t.R > 0)
    return {"trades": trades, "n": n, "wins": wins, "total_R": total_R,
            "avg_R": round(total_R / n, 2) if n else 0.0,
            "expectancy": round(total_R / n, 2) if n else 0.0}


def render(result: dict) -> str:
    L = [f"AVT session: {result['n']} trades · {result['wins']} wins · total {result['total_R']}R · avg {result['avg_R']}R"]
    for t in result["trades"]:
        L.append(f"  {t.direction:<4} {t.entry_ts}→{t.exit_ts}  entry {t.entry} stop {t.stop} exit {t.exit}"
                 f"  → {t.R:+.2f}R (MFE {t.mfe_R:+.2f}R, {t.bars_held} bars, {t.exit_reason})")
    return "\n".join(L)
