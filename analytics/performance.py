"""
Performance Analytics
======================
Aggregate metrics for completed trades: win rate, expectancy,
profit factor, R-distribution, per-symbol/strategy breakdowns.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional


@dataclass
class PerformanceSnapshot:
    trade_count: int = 0; win_count: int = 0; loss_count: int = 0
    breakeven_count: int = 0; win_rate: float = 0.0
    total_pnl: float = 0.0; avg_pnl: float = 0.0; median_pnl: float = 0.0
    gross_profit: float = 0.0; gross_loss: float = 0.0; profit_factor: float = 0.0
    avg_r: float = 0.0; median_r: float = 0.0; best_r: float = 0.0
    worst_r: float = 0.0; expectancy_r: float = 0.0
    avg_risk_pct: float = 0.0; avg_discipline: float = 0.0; avg_conviction: float = 0.0
    by_symbol: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_strategy: dict = field(default_factory=dict)
    from_date: Optional[str] = None; to_date: Optional[str] = None
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_count": self.trade_count, "win_count": self.win_count,
            "loss_count": self.loss_count, "breakeven_count": self.breakeven_count,
            "win_rate": round(self.win_rate, 4),
            "pnl": {"total": round(self.total_pnl, 2), "avg": round(self.avg_pnl, 2),
                    "median": round(self.median_pnl, 2),
                    "gross_profit": round(self.gross_profit, 2),
                    "gross_loss": round(self.gross_loss, 2),
                    "profit_factor": round(self.profit_factor, 3)},
            "r_multiples": {"avg": round(self.avg_r, 3), "median": round(self.median_r, 3),
                            "best": round(self.best_r, 3), "worst": round(self.worst_r, 3),
                            "expectancy": round(self.expectancy_r, 3)},
            "risk": {"avg_risk_pct": round(self.avg_risk_pct, 2)},
            "quality": {"avg_discipline": round(self.avg_discipline, 2),
                        "avg_conviction": round(self.avg_conviction, 2)},
            "by_symbol": self.by_symbol, "by_direction": self.by_direction,
            "by_strategy": self.by_strategy,
            "from_date": self.from_date, "to_date": self.to_date,
            "computed_at": self.computed_at,
        }


def _grp(trades: list) -> dict:
    c = [t for t in trades if getattr(t, "pnl", None) is not None]
    if not c:
        return {"trade_count": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_r": 0.0}
    pnls = [t.pnl for t in c]
    rs = [t.r_multiple for t in c if getattr(t, "r_multiple", None) is not None]
    wins = [p for p in pnls if p > 0]
    return {"trade_count": len(c), "win_rate": round(len(wins)/len(c), 4),
            "total_pnl": round(sum(pnls), 2),
            "avg_r": round(statistics.mean(rs), 3) if rs else 0.0}


class PerformanceAnalytics:

    def compute(self, trades: list, from_date: Optional[datetime] = None,
                to_date: Optional[datetime] = None) -> PerformanceSnapshot:
        c = [t for t in trades if getattr(t, "pnl", None) is not None]
        if not c:
            return PerformanceSnapshot(
                from_date=from_date.isoformat() if from_date else None,
                to_date=to_date.isoformat() if to_date else None)
        pnls = [t.pnl for t in c]
        rs = [t.r_multiple for t in c if getattr(t, "r_multiple", None) is not None]
        wins = [t for t in c if t.pnl > 0]; losses = [t for t in c if t.pnl < 0]
        beven = [t for t in c if t.pnl == 0]
        gp = sum(t.pnl for t in wins); gl = abs(sum(t.pnl for t in losses))
        pf = gp / gl if gl > 0 else float("inf")
        wr_ns = [t.r_multiple for t in wins if getattr(t, "r_multiple", None) is not None]
        lr_ns = [abs(t.r_multiple) for t in losses if getattr(t, "r_multiple", None) is not None]
        wr = len(wins) / len(c)
        awr = statistics.mean(wr_ns) if wr_ns else 0.0
        alr = statistics.mean(lr_ns) if lr_ns else 1.0
        exp = wr * awr - (1 - wr) * alr
        rps = [t.risk_pct for t in c if getattr(t, "risk_pct", None)]
        ds = [getattr(t, "discipline_score", None) for t in c
              if getattr(t, "discipline_score", None) is not None]
        cv = [t.conviction_score for t in c if getattr(t, "conviction_score", None) is not None]
        bsym: dict = {}; bdir: dict = {}; bst: dict = {}
        for t in c:
            bsym.setdefault(t.symbol, []).append(t)
            d = getattr(t, "direction", "unknown")
            bdir.setdefault(str(d.value if hasattr(d, "value") else d), []).append(t)
            bst.setdefault(getattr(t, "declared_strategy_id", None) or "unknown", []).append(t)
        return PerformanceSnapshot(
            trade_count=len(c), win_count=len(wins), loss_count=len(losses),
            breakeven_count=len(beven), win_rate=wr,
            total_pnl=sum(pnls), avg_pnl=statistics.mean(pnls), median_pnl=statistics.median(pnls),
            gross_profit=gp, gross_loss=gl, profit_factor=pf,
            avg_r=statistics.mean(rs) if rs else 0.0,
            median_r=statistics.median(rs) if rs else 0.0,
            best_r=max(rs) if rs else 0.0, worst_r=min(rs) if rs else 0.0,
            expectancy_r=exp,
            avg_risk_pct=statistics.mean(rps) if rps else 0.0,
            avg_discipline=statistics.mean(ds) if ds else 0.0,
            avg_conviction=statistics.mean(cv) if cv else 0.0,
            by_symbol={k: _grp(v) for k, v in bsym.items()},
            by_direction={k: _grp(v) for k, v in bdir.items()},
            by_strategy={k: _grp(v) for k, v in bst.items()},
            from_date=from_date.isoformat() if from_date else None,
            to_date=to_date.isoformat() if to_date else None,
        )

    def from_db(self, db, lookback_days: int = 90) -> PerformanceSnapshot:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        return self.compute(
            db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                   Trade.entry_time >= cutoff).all(),
            from_date=cutoff)
