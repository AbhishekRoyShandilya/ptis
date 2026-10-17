"""
Regime Analytics
=================
Performance breakdown by market regime (trending, ranging, volatile, calm).
Helps traders understand which market conditions best suit their edge.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class RegimePerformance:
    regime: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    avg_r: float
    avg_discipline: float
    profitability_rank: int = 0     # 1 = best regime for this trader


@dataclass
class RegimeReport:
    total_trades: int = 0
    by_regime: list = field(default_factory=list)        # list[RegimePerformance]
    best_regime: Optional[str] = None
    worst_regime: Optional[str] = None
    regime_distribution: dict = field(default_factory=dict)   # {regime: count}
    # Advice flags
    avoid_regimes: list = field(default_factory=list)    # regimes with avg_pnl < 0
    focus_regimes: list = field(default_factory=list)    # regimes with win_rate > 0.6
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "by_regime": [
                {"regime": r.regime, "trade_count": r.trade_count,
                 "win_rate": round(r.win_rate, 4), "avg_pnl": round(r.avg_pnl, 2),
                 "total_pnl": round(r.total_pnl, 2), "avg_r": round(r.avg_r, 3),
                 "avg_discipline": round(r.avg_discipline, 2),
                 "profitability_rank": r.profitability_rank}
                for r in self.by_regime
            ],
            "best_regime": self.best_regime,
            "worst_regime": self.worst_regime,
            "regime_distribution": self.regime_distribution,
            "avoid_regimes": self.avoid_regimes,
            "focus_regimes": self.focus_regimes,
            "computed_at": self.computed_at,
        }


class RegimeAnalytics:

    def compute(self, trades: list) -> RegimeReport:
        closed = [t for t in trades if getattr(t, "pnl", None) is not None]
        if not closed:
            return RegimeReport()

        by_regime: dict[str, list] = {}
        for t in closed:
            reg = getattr(t, "market_regime", None)
            if reg is None:
                reg = "unknown"
            key = str(reg.value if hasattr(reg, "value") else reg)
            by_regime.setdefault(key, []).append(t)

        rp_list: list[RegimePerformance] = []
        for regime, ts in by_regime.items():
            pnls = [t.pnl for t in ts]
            wins = [p for p in pnls if p > 0]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            discs = [t.discipline_score for t in ts
                     if getattr(t, "discipline_score", None) is not None]
            rp_list.append(RegimePerformance(
                regime=regime,
                trade_count=len(ts),
                win_rate=len(wins) / len(ts),
                avg_pnl=statistics.mean(pnls),
                total_pnl=sum(pnls),
                avg_r=statistics.mean(rs) if rs else 0.0,
                avg_discipline=statistics.mean(discs) if discs else 0.0,
            ))

        # Rank by avg_pnl
        ranked = sorted(rp_list, key=lambda r: -r.avg_pnl)
        for i, rp in enumerate(ranked):
            rp.profitability_rank = i + 1

        valid = [r for r in rp_list if r.trade_count >= 3]
        best = max(valid, key=lambda r: r.avg_pnl).regime if valid else None
        worst = min(valid, key=lambda r: r.avg_pnl).regime if valid else None

        dist = {r.regime: r.trade_count for r in rp_list}
        avoid = [r.regime for r in valid if r.avg_pnl < 0]
        focus = [r.regime for r in valid if r.win_rate > 0.6]

        return RegimeReport(
            total_trades=len(closed),
            by_regime=sorted(rp_list, key=lambda r: -r.trade_count),
            best_regime=best,
            worst_regime=worst,
            regime_distribution=dist,
            avoid_regimes=avoid,
            focus_regimes=focus,
        )

    def from_db(self, db, lookback_days: int = 90) -> RegimeReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
