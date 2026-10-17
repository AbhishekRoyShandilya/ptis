"""
Edge Analysis
==============
Statistical edge testing and decay detection.
Answers: "Is my edge still working?" and "Which setups still have an edge?"
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class SetupEdge:
    setup_type: str
    trade_count: int
    win_rate: float
    avg_r: float
    expectancy: float           # win_rate * avg_win_r + (1-win_rate) * avg_loss_r
    edge_score: float           # composite 0-100
    statistically_significant: bool    # trade_count >= 30 and p-value proxy < 0.05
    sample_adequate: bool       # trade_count >= 20


@dataclass
class EdgeReport:
    overall_expectancy: float = 0.0
    overall_edge_score: float = 0.0     # 0-100
    by_setup: list = field(default_factory=list)          # list[SetupEdge]
    by_direction: dict = field(default_factory=dict)      # {long/short: SetupEdge-like dict}
    rolling_win_rate_30d: float = 0.0
    rolling_win_rate_90d: float = 0.0
    edge_stable: bool = True
    edge_decay_detected: bool = False
    decay_signal: Optional[str] = None     # human-readable decay description
    # Profit factor breakdown
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "overall": {
                "expectancy": round(self.overall_expectancy, 3),
                "edge_score": round(self.overall_edge_score, 1),
                "profit_factor": round(self.profit_factor, 3),
                "gross_profit": round(self.gross_profit, 2),
                "gross_loss": round(self.gross_loss, 2),
            },
            "by_setup": [
                {"setup_type": s.setup_type, "trade_count": s.trade_count,
                 "win_rate": round(s.win_rate, 4), "avg_r": round(s.avg_r, 3),
                 "expectancy": round(s.expectancy, 3), "edge_score": round(s.edge_score, 1),
                 "statistically_significant": s.statistically_significant,
                 "sample_adequate": s.sample_adequate}
                for s in self.by_setup
            ],
            "by_direction": self.by_direction,
            "rolling": {
                "win_rate_30d": round(self.rolling_win_rate_30d, 4),
                "win_rate_90d": round(self.rolling_win_rate_90d, 4),
            },
            "edge_health": {
                "stable": self.edge_stable,
                "decay_detected": self.edge_decay_detected,
                "decay_signal": self.decay_signal,
            },
            "computed_at": self.computed_at,
        }


def _expectancy(pnls: list[float], rs: list[float]) -> float:
    """R-based expectancy. Falls back to PnL-based if no R data."""
    if rs:
        wins_r = [r for r in rs if r > 0]
        loss_r = [r for r in rs if r <= 0]
        wr = len(wins_r) / len(rs)
        avg_w = statistics.mean(wins_r) if wins_r else 0.0
        avg_l = statistics.mean(loss_r) if loss_r else 0.0
        return wr * avg_w + (1 - wr) * avg_l
    if pnls:
        wins_p = [p for p in pnls if p > 0]
        loss_p = [p for p in pnls if p <= 0]
        wr = len(wins_p) / len(pnls)
        avg_w = statistics.mean(wins_p) if wins_p else 0.0
        avg_l = statistics.mean(loss_p) if loss_p else 0.0
        return wr * avg_w + (1 - wr) * avg_l
    return 0.0


def _edge_score(win_rate: float, expectancy: float, trade_count: int) -> float:
    """
    Composite score 0–100.
    Components: win_rate (40%), expectancy (40%), sample adequacy (20%).
    """
    wr_score = min(win_rate * 200, 100) * 0.40           # 50% wr → 40 pts max
    exp_score = min(max(expectancy * 20 + 50, 0), 100) * 0.40  # 0R → 50 pts, 1R → 70 pts
    sample_score = min(trade_count / 30, 1.0) * 100 * 0.20     # 30 trades → full 20 pts
    return round(wr_score + exp_score + sample_score, 1)


def _stat_significant(win_count: int, total: int) -> bool:
    """
    Simple binomial z-test proxy against 50% null hypothesis.
    z = (k - n*p0) / sqrt(n*p0*(1-p0)); significant if |z| > 1.96 (95% CI).
    """
    if total < 10:
        return False
    p0 = 0.5
    z = (win_count - total * p0) / math.sqrt(total * p0 * (1 - p0))
    return abs(z) > 1.96


class EdgeAnalyzer:

    def compute(self, trades: list) -> EdgeReport:
        closed = [t for t in trades if getattr(t, "pnl", None) is not None]
        if not closed:
            return EdgeReport()

        now = datetime.utcnow()
        cutoff_30 = (now - timedelta(days=30)).isoformat()
        cutoff_90 = (now - timedelta(days=90)).isoformat()

        def _date_str(t) -> str:
            et = getattr(t, "exit_time", None) or getattr(t, "entry_time", None)
            if et:
                return et.isoformat()
            return ""

        trades_30 = [t for t in closed if _date_str(t) >= cutoff_30]
        trades_90 = [t for t in closed if _date_str(t) >= cutoff_90]

        def _wr(ts): return len([t for t in ts if t.pnl > 0]) / len(ts) if ts else 0.0

        rwr_30 = _wr(trades_30)
        rwr_90 = _wr(trades_90)

        # Overall metrics
        all_pnls = [t.pnl for t in closed]
        all_rs = [t.r_multiple for t in closed if getattr(t, "r_multiple", None) is not None]
        gross_profit = sum(p for p in all_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        overall_exp = _expectancy(all_pnls, all_rs)
        wins_total = sum(1 for t in closed if t.pnl > 0)
        overall_wr = wins_total / len(closed)
        overall_edge = _edge_score(overall_wr, overall_exp, len(closed))

        # By setup type
        by_setup: dict[str, list] = {}
        for t in closed:
            st = getattr(t, "setup_type", None) or getattr(t, "strategy_id", None) or "unknown"
            key = str(st.value if hasattr(st, "value") else st)
            by_setup.setdefault(key, []).append(t)

        setup_edges: list[SetupEdge] = []
        for setup, ts in sorted(by_setup.items(), key=lambda x: -len(x[1])):
            pnls = [t.pnl for t in ts]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(ts)
            exp = _expectancy(pnls, rs)
            avg_r_val = statistics.mean(rs) if rs else 0.0
            setup_edges.append(SetupEdge(
                setup_type=setup, trade_count=len(ts),
                win_rate=wr, avg_r=avg_r_val, expectancy=exp,
                edge_score=_edge_score(wr, exp, len(ts)),
                statistically_significant=_stat_significant(wins, len(ts)),
                sample_adequate=len(ts) >= 20,
            ))

        # By direction
        by_dir: dict[str, list] = {}
        for t in closed:
            d = getattr(t, "direction", None)
            if d:
                key = str(d.value if hasattr(d, "value") else d)
                by_dir.setdefault(key, []).append(t)
        dir_summary = {}
        for direction, ts in by_dir.items():
            pnls = [t.pnl for t in ts]
            rs = [t.r_multiple for t in ts if getattr(t, "r_multiple", None) is not None]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(ts)
            dir_summary[direction] = {
                "trade_count": len(ts),
                "win_rate": round(wr, 4),
                "avg_r": round(statistics.mean(rs), 3) if rs else 0.0,
                "expectancy": round(_expectancy(pnls, rs), 3),
            }

        # Edge decay detection: rolling 30d win rate < rolling 90d win rate by >10pp
        decay_detected = False
        decay_signal = None
        if trades_90 and trades_30:
            if rwr_30 < rwr_90 - 0.10:
                decay_detected = True
                decay_signal = (
                    f"Win rate dropped from {rwr_90:.0%} (90d) to {rwr_30:.0%} (30d) — "
                    f"edge may be deteriorating. Review recent setups."
                )
            elif rwr_30 < 0.40:
                decay_detected = True
                decay_signal = f"30-day win rate {rwr_30:.0%} is below 40% — edge under pressure."

        return EdgeReport(
            overall_expectancy=overall_exp,
            overall_edge_score=overall_edge,
            by_setup=setup_edges,
            by_direction=dir_summary,
            rolling_win_rate_30d=rwr_30,
            rolling_win_rate_90d=rwr_90,
            edge_stable=not decay_detected,
            edge_decay_detected=decay_detected,
            decay_signal=decay_signal,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=pf if pf != float("inf") else 999.0,
        )

    def from_db(self, db, lookback_days: int = 90) -> EdgeReport:
        from core.models import Trade
        from core.enums import TradeStatus
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED,
                                        Trade.entry_time >= cutoff).all()
        return self.compute(trades)
