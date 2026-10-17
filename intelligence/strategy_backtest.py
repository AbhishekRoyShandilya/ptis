"""
intelligence.strategy_backtest — backtest a TAUGHT STRATEGY on the brain's experience.

A "strategy" here is a SIGNATURE over the brain's validated vocabulary — a direction + required/excluded
recognition tags (e.g. an opening-momentum long = up · {AT_OPEN, WITH_TREND, ROOM_CLEAR}). Backtesting
it = asking the episodic LIBRARY how candidates matching that signature actually RESOLVED: how often they
became money moves, how far they ran, on WHICH stocks, in WHICH session phase — vs the base rate — and
whether the edge SURVIVES both-halves (propose→validate). This is instant (uses the stored library, no
re-narration) and is the "useful stats + rich insights → only take well-performing trades" layer.

`library_stats(strategy)` = recognition performance (became-a-move + how far). The deeper trade-R version
(entry/exit/R via the operating process) is `intelligence.avt_trader` filtered to the signature — heavier
(needs replay), wired here as `trade_backtest()`.
"""
from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass, field, asdict


@dataclass
class Strategy:
    name: str
    direction: str = "both"               # up / down / both
    require: set = field(default_factory=set)   # tags that MUST be present
    exclude: set = field(default_factory=set)   # tags that must be ABSENT

    def matches(self, ep) -> bool:
        if self.direction != "both" and ep.direction != self.direction:
            return False
        tags = set(ep.tags)
        return self.require.issubset(tags) and not (self.exclude & tags)


@dataclass
class StrategyStats:
    name: str
    n: int = 0
    ran: int = 0                  # became a money move (peak >= 5 ATR)
    modest: int = 0
    failed: int = 0
    ran_rate: float = 0.0
    base_ran_rate: float = 0.0    # over ALL episodes — the bar to beat
    lift: float = 0.0             # ran_rate - base_ran_rate
    median_run: float = float("nan")
    validated: bool = False       # does the lift hold, same sign, both halves?
    per_symbol: list = field(default_factory=list)   # [(symbol, n, ran_rate)]
    by_phase: list = field(default_factory=list)      # [(phase, n, ran_rate)]
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _ran_rate(eps) -> float:
    return sum(e.peak_ext >= 5 for e in eps) / len(eps) if eps else 0.0


def library_stats(strategy: Strategy, episodes=None) -> StrategyStats:
    from intelligence.episodic import EpisodeStore
    eps = episodes if episodes is not None else EpisodeStore().load()
    eps = [e for e in eps if e.peak_ext == e.peak_ext]
    if not eps:
        return StrategyStats(name=strategy.name, note="empty library")
    base = _ran_rate(eps)
    m = [e for e in eps if strategy.matches(e)]
    if not m:
        return StrategyStats(name=strategy.name, base_ran_rate=round(base, 3),
                             note="no episodes match this signature")
    n = len(m)
    ran = [e for e in m if e.peak_ext >= 5]
    modest = [e for e in m if 3 <= e.peak_ext < 5]
    rr = len(ran) / n
    med = round(_stats.median(e.peak_ext for e in ran), 1) if ran else float("nan")

    # per-symbol (where it works) — sorted by ran_rate, min 3 samples to show a rate
    by_sym: dict[str, list] = {}
    for e in m:
        by_sym.setdefault(e.symbol, []).append(e)
    per_symbol = sorted(((s, len(v), round(_ran_rate(v), 2)) for s, v in by_sym.items() if len(v) >= 3),
                        key=lambda r: -r[2])
    # by session phase
    def phase(e):
        return next((t for t in e.tags if t.startswith("PHASE_")), "PHASE_?")
    by_ph: dict[str, list] = {}
    for e in m:
        by_ph.setdefault(phase(e), []).append(e)
    by_phase = sorted(((p, len(v), round(_ran_rate(v), 2)) for p, v in by_ph.items()),
                      key=lambda r: -r[2])

    # validate the LIFT both-halves (propose→validate): does matching beat base, same sign, in both halves?
    ms = sorted(m, key=lambda e: (e.symbol, e.date, e.ts))
    A, B = ms[0::2], ms[1::2]
    eA, eB = eps[0::2], eps[1::2]
    liftA = _ran_rate(A) - _ran_rate(eA)
    liftB = _ran_rate(B) - _ran_rate(eB)
    validated = (liftA > 0) == (liftB > 0) and abs(liftA) >= 0.03 and abs(liftB) >= 0.03

    return StrategyStats(
        name=strategy.name, n=n, ran=len(ran), modest=len(modest), failed=n - len(ran) - len(modest),
        ran_rate=round(rr, 3), base_ran_rate=round(base, 3), lift=round(rr - base, 3),
        median_run=med, validated=bool(validated), per_symbol=per_symbol, by_phase=by_phase,
        note=("validated edge — beats base both halves" if validated
              else "NOT validated — lift doesn't hold both halves (treat as noise)"),
    )


def render(s: StrategyStats) -> str:
    if s.n == 0:
        return f"[{s.name}] {s.note}"
    sign = "+" if s.lift >= 0 else ""
    L = [f"[{s.name}]  n={s.n}  ·  money-move rate {s.ran_rate*100:.0f}% vs base {s.base_ran_rate*100:.0f}% "
         f"({sign}{s.lift*100:.0f} lift)  ·  median run +{s.median_run} ATR  ·  {'✓ VALIDATED' if s.validated else '✗ not validated'}",
         f"   outcomes: {s.ran} money-moves / {s.modest} modest / {s.failed} failed",
         f"   {s.note}"]
    if s.per_symbol:
        top = ", ".join(f"{sym} {rr*100:.0f}%(n{n})" for sym, n, rr in s.per_symbol[:6])
        L.append(f"   best stocks: {top}")
    if s.by_phase:
        ph = ", ".join(f"{p.replace('PHASE_','')} {rr*100:.0f}%(n{n})" for p, n, rr in s.by_phase)
        L.append(f"   by phase: {ph}")
    return "\n".join(L)


def trade_backtest(strategy: Strategy, sessions: list[tuple[str, str]], interval: str = "5m") -> dict:
    """Deeper: the operating-process R for the strategy (entry on confirmation of a MATCHING candidate,
    structural stop, story-change exit) over `sessions` = [(symbol, date), …]. Heavier (replays each
    session through the brain). Reuses avt_trader's trade logic via a signature filter."""
    from intelligence.avt_trader import simulate_session
    flt = lambda direction, tags: strategy.matches(_EpShim(direction, tags))
    all_trades, total_R, n, wins = [], 0.0, 0, 0
    for sym, d in sessions:
        res = simulate_session(sym, interval, d, strategy_filter=flt)
        all_trades += res["trades"]; total_R += res["total_R"]; n += res["n"]; wins += res["wins"]
    return {"strategy": strategy.name, "n": n, "wins": wins, "total_R": round(total_R, 2),
            "avg_R": round(total_R / n, 2) if n else 0.0, "trades": all_trades}


@dataclass
class _EpShim:
    """Adapts (direction, tags) to the Strategy.matches() interface for the trade filter."""
    direction: str
    tags: list
