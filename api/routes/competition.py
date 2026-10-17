# api/routes/competition.py
"""
Competition Framework API Routes
==================================
REST endpoints for the trader-vs-system competition framework.

Endpoints:
- GET  /api/v1/competition/report/weekly    — side-by-side weekly report
- GET  /api/v1/competition/report/monthly   — side-by-side monthly report
- GET  /api/v1/competition/missed           — missed opportunities
- GET  /api/v1/competition/risk             — risk-adjusted comparison
- GET  /api/v1/competition/coaching         — "where am I lagging" session
- GET  /api/v1/competition/journal/open     — open autonomous entries
- GET  /api/v1/competition/journal/history  — closed autonomous entries
- GET  /api/v1/competition/journal/performance — system performance summary
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from api.auth import get_current_user
from core.database import get_db
from sqlalchemy.orm import Session as DBSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/competition", tags=["competition"], dependencies=[Depends(get_current_user)])


# ── Side-by-Side Reports ──

@router.get("/report/weekly")
def get_weekly_report(
    week_offset: int = Query(0, description="0=current, -1=last week"),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get weekly side-by-side comparison report.

    Compares trader's actual trades against system's autonomous
    shadow trades for the specified week.
    """
    from competition.side_by_side_report import SideBySideReportGenerator
    generator = SideBySideReportGenerator(db)
    report = generator.weekly_report(week_offset)
    return report.to_dict()


@router.get("/report/monthly")
def get_monthly_report(
    month_offset: int = Query(0, description="0=current, -1=last month"),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get monthly side-by-side comparison report."""
    from competition.side_by_side_report import SideBySideReportGenerator
    generator = SideBySideReportGenerator(db)
    report = generator.monthly_report(month_offset)
    return report.to_dict()


# ── Missed Opportunities ──

@router.get("/missed")
def get_missed_opportunities(
    week_offset: int = Query(0, description="0=current, -1=last week"),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get missed opportunity report.

    Shows profitable setups the system identified but the trader didn't take.
    Categorized by severity: critical, notable, informational.
    """
    from competition.missed_opportunity_detector import MissedOpportunityDetector
    detector = MissedOpportunityDetector(db)
    report = detector.analyze_week(week_offset)
    return report.to_dict()


# ── Risk-Adjusted Comparison ──

@router.get("/risk")
def get_risk_comparison(
    week_offset: int = Query(0, description="0=current, -1=last week"),
    capital: float = Query(500_000.0, description="Capital base for % calculations"),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get risk-adjusted P&L comparison.

    Computes Sharpe, Sortino, Calmar ratios, max drawdown,
    Kelly criterion, and risk grade (A-F) for both trader and system.
    """
    from competition.risk_adjusted_comparison import RiskAdjustedComparator
    comparator = RiskAdjustedComparator(db, capital=capital)
    report = comparator.weekly_comparison(week_offset)
    return report.to_dict()


# ── Coaching Session ──

@router.get("/coaching")
def get_coaching_session(
    week_offset: int = Query(0, description="0=current, -1=last week"),
    capital: float = Query(500_000.0),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get "Where am I lagging?" coaching session.

    Generates prioritized coaching cards across 5 categories:
    setup selection, risk management, exit execution, discipline, awareness.
    Includes quick wins, category scores, and overall grade.
    """
    from competition.lagging_coach import LaggingCoach
    coach = LaggingCoach(db, capital=capital)
    session = coach.weekly_coaching(week_offset)
    return session.to_dict()


# ── System Journal ──

@router.get("/journal/open")
def get_open_journal_entries(
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get all currently open autonomous journal entries."""
    from automation.system_journal import SystemJournal
    journal = SystemJournal(db)
    entries = journal.get_open_entries()
    return {
        "count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }


@router.get("/journal/history")
def get_journal_history(
    limit: int = Query(50, ge=1, le=200),
    symbol: Optional[str] = Query(None),
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get closed autonomous journal entries for review."""
    from automation.system_journal import SystemJournal
    journal = SystemJournal(db)
    entries = journal.get_history(limit=limit, symbol=symbol)
    return {
        "count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }


@router.get("/journal/performance")
def get_journal_performance(
    db: DBSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get aggregated performance of the system's autonomous trades.

    Returns win rate, expectancy, total P&L, best/worst trade.
    """
    from automation.system_journal import SystemJournal
    journal = SystemJournal(db)
    return journal.get_performance_summary()


# ── Scanner / Orchestrator Status ──

@router.get("/scanner/status")
def get_scanner_status() -> Dict[str, Any]:
    """Get market scanner orchestrator status and stats."""
    try:
        from automation.market_scanner_orchestrator import _orchestrator
        if _orchestrator:
            return _orchestrator.get_stats()
        return {"running": False, "message": "Orchestrator not initialized"}
    except Exception as e:
        return {"running": False, "error": str(e)}


@router.get("/fno/universe")
def get_fno_universe() -> Dict[str, Any]:
    """Get the current FNO stock universe."""
    from automation.fno_universe import FNOUniverseFetcher
    fetcher = FNOUniverseFetcher()
    stocks = fetcher.get_fno_stocks()
    return {
        "count": len(stocks),
        "last_refresh": fetcher.last_refresh.isoformat() if fetcher.last_refresh else None,
        "stocks": [s.to_dict() for s in stocks],
        "nifty50": fetcher.get_nifty50(),
    }


# ── Teaching Cross-Questioning ──

@router.get("/teaching/questions/{session_id}")
def get_teaching_questions(
    session_id: str,
) -> Dict[str, Any]:
    """
    Get cross-examination questions for a teaching session.

    Returns current questions, completion percentage, and next question.
    """
    from strategy_learning.cross_questioner import CrossQuestioner
    cq = CrossQuestioner()
    # In production, load state from DB. For now, return template.
    state = cq.start_questioning(session_id, "Strategy")
    return state.to_dict()
