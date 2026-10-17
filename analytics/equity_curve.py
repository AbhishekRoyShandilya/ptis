"""
Equity Curve Analytics
========================
Build equity curves from trade P&L sequences.
Provides drawdown events, Sharpe/Sortino/Calmar ratios, underwater period analysis.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EquityPoint:
    date: str; equity: float; pnl: float; peak: float
    drawdown_abs: float; drawdown_pct: float; underwater: bool


@dataclass
class DrawdownEvent:
    start_date: str; trough_date: str; end_date: Optional[str]
    peak_equity: float; trough_equity: float
    drawdown_abs: float; drawdown_pct: float; recovered: bool


@dataclass
class EquityCurveReport:
    start_equity: float; end_equity: float; total_return_pct: float
    points: list; max_drawdown_pct: float; max_drawdown_abs: float
    avg_drawdown_pct: float; drawdown_events: list; current_drawdown_pct: float
    sharpe_ratio: float; sortino_ratio: float; calmar_ratio: float
    longest_underwater_days: int; total_underwater_days: int; pct_time_underwater: float
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_drawdown_abs": round(self.max_drawdown_abs, 2),
            "avg_drawdown_pct": round(self.avg_drawdown_pct, 4),
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "ratios": {"sharpe": round(self.sharpe_ratio, 3),
                       "sortino": round(self.sortino_ratio, 3),
                       "calmar": round(self.calmar_ratio, 3)},
            "underwater": {"longest_days": self.longest_underwater_days,
                           "total_days": self.total_underwater_days,
                           "pct_time": round(self.pct_time_underwater, 4)},
            "drawdown_events": [
                {"start": d.start_date, "trough": d.trough_date, "end": d.end_date,
                 "drawdown_pct": round(d.drawdown_pct, 4),
                 "drawdown_abs": round(d.drawdown_abs, 2), "recovered": d.recovered}
                for d in self.drawdown_events[:10]],
            "equity_series": [{"date": p.date, "equity": round(p.equity, 2),
                                "peak": round(p.peak, 2),
                                "drawdown_pct": round(p.drawdown_pct, 4)}
                               for p in self.points],
            "computed_at": self.computed_at,
        }


class EquityCurveBuilder:

    def from_pnl_series(self, pnls: list[float], dates: Optional[list[str]] = None,
                        start_equity: float = 100_000.0, risk_free_rate: float = 0.0,
                        periods_per_year: int = 252) -> EquityCurveReport:
        if not pnls:
            return self._empty(start_equity)
        if dates is None:
            dates = [f"T{i+1}" for i in range(len(pnls))]
        pts: list[EquityPoint] = []; eq = start_equity; pk = start_equity; rets: list[float] = []
        for pnl, dt in zip(pnls, dates):
            prev = eq; eq = prev + pnl; pk = max(pk, eq)
            dd = pk - eq; rets.append(pnl / prev if prev > 0 else 0.0)
            pts.append(EquityPoint(dt, eq, pnl, pk, dd, dd/pk if pk > 0 else 0.0, eq < pk))
        evts = self._dd_events(pts)
        rf = risk_free_rate / periods_per_year
        exc = [r - rf for r in rets]
        max_dd = max((p.drawdown_pct for p in pts), default=0.0)
        ann = self._ann(rets, periods_per_year)
        dd_pcts = [e.drawdown_pct for e in evts]
        uw = self._uw(pts)
        return EquityCurveReport(
            start_equity=start_equity, end_equity=pts[-1].equity,
            total_return_pct=(pts[-1].equity - start_equity) / start_equity,
            points=pts, max_drawdown_pct=max_dd,
            max_drawdown_abs=max((e.drawdown_abs for e in evts), default=0.0),
            avg_drawdown_pct=statistics.mean(dd_pcts) if dd_pcts else 0.0,
            drawdown_events=evts, current_drawdown_pct=pts[-1].drawdown_pct,
            sharpe_ratio=self._sharpe(exc, periods_per_year),
            sortino_ratio=self._sortino(exc, periods_per_year),
            calmar_ratio=ann / max_dd if max_dd > 0 else float("inf"),
            longest_underwater_days=max(uw, default=0),
            total_underwater_days=sum(uw),
            pct_time_underwater=sum(uw) / len(pts) if pts else 0.0,
        )

    def from_trades(self, trades: list, start_equity: float = 100_000.0) -> EquityCurveReport:
        c = sorted([t for t in trades if getattr(t,"pnl",None) is not None
                    and getattr(t,"exit_time",None) is not None], key=lambda t: t.exit_time)
        if not c:
            return self._empty(start_equity)
        return self.from_pnl_series([t.pnl for t in c],
                                    [t.exit_time.date().isoformat() for t in c], start_equity)

    def _empty(self, se: float) -> EquityCurveReport:
        return EquityCurveReport(se, se, 0.0, [], 0.0, 0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    def _dd_events(self, pts: list[EquityPoint]) -> list[DrawdownEvent]:
        evts, in_dd = [], False
        pk = sd = td = teq = None
        for p in pts:
            if not in_dd and p.underwater:
                in_dd, pk, sd, td, teq = True, p.peak, p.date, p.date, p.equity
            elif in_dd:
                if p.equity < teq: teq, td = p.equity, p.date
                if not p.underwater:
                    dd = pk - teq
                    evts.append(DrawdownEvent(sd, td, p.date, pk, teq, dd, dd/pk if pk else 0, True))
                    in_dd = False
        if in_dd and pk:
            dd = pk - teq
            evts.append(DrawdownEvent(sd, td, None, pk, teq, dd, dd/pk if pk else 0, False))
        return sorted(evts, key=lambda e: e.drawdown_pct, reverse=True)

    def _uw(self, pts: list[EquityPoint]) -> list[int]:
        runs, n = [], 0
        for p in pts:
            if p.underwater: n += 1
            elif n: runs.append(n); n = 0
        if n: runs.append(n)
        return runs

    def _sharpe(self, exc: list[float], periods: int) -> float:
        if len(exc) < 2: return 0.0
        std = statistics.stdev(exc)
        return (statistics.mean(exc) / std) * math.sqrt(periods) if std else 0.0

    def _sortino(self, exc: list[float], periods: int) -> float:
        neg = [r for r in exc if r < 0]
        if len(neg) < 2: return 0.0
        std = statistics.stdev(neg)
        return (statistics.mean(exc) / std) * math.sqrt(periods) if std else 0.0

    def _ann(self, rets: list[float], periods: int) -> float:
        if not rets: return 0.0
        t = 1.0
        for r in rets: t *= (1 + r)
        ny = len(rets) / periods
        return t ** (1 / ny) - 1 if ny > 0 else 0.0
