"""
Workflow Copilot
================
Proactive pre-session pipeline that prepares a full battle plan before
the trader opens the co-planner chat.

Pipeline (all rule-based except the final synthesis):

  Step 1  Behavioral context       CoPlannerContextAssembler (sync SQL)
  Step 2  Last-week mistakes       LastWeekMistakeSummarizer (sync SQL)
  Step 3  Per-symbol scan          readiness_scanner.SetupScanner
                                   → condition-by-condition checklist
                                   → regime alignment score
                                   → R:R estimate
  Step 4  Grade & rank             A/B/C/skip — pure maths, no LLM
  Step 5  Battle plan narrative    One LLM call (synthesis only)

Key difference vs session_briefing_agent.py:
  - Uses intelligence/readiness_scanner.py (full condition checklist)
    instead of automation/setup_scanner.py (pattern detection only)
  - Injects last-week mistake patterns into the LLM prompt
  - Grades use regime_suitability + conditions_met_ratio + R multiple
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Output types ─────────────────────────────────────────────────


@dataclass
class SymbolGrade:
    symbol: str
    grade: str                          # A / B / C / skip
    strategy_name: str
    conditions_met: int
    conditions_total: int
    conditions: list[dict]              # [{name, is_met, proximity, notes}]
    regime: str
    regime_aligned: bool
    readiness_score: float              # 0-1
    estimated_r: Optional[float]
    alert_priority: str                 # critical / high / medium / low
    skip_reason: str = ""

    @property
    def checklist_text(self) -> str:
        """Formatted checklist for LLM prompt injection."""
        if not self.conditions:
            return f"  No conditions available ({self.conditions_met}/{self.conditions_total})"
        lines = []
        for c in self.conditions:
            tick = "YES" if c["is_met"] else "NO "
            prox = (
                f" [{round(c['proximity'] * 100)}% close]"
                if not c["is_met"] and c.get("proximity", 0) > 0.1
                else ""
            )
            note = f" — {c['notes']}" if c.get("notes") else ""
            lines.append(f"    [{tick}] {c['name']}{prox}{note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "grade": self.grade,
            "strategy_name": self.strategy_name,
            "conditions_met": self.conditions_met,
            "conditions_total": self.conditions_total,
            "conditions": self.conditions,
            "regime": self.regime,
            "regime_aligned": self.regime_aligned,
            "readiness_score": round(self.readiness_score, 3),
            "estimated_r": self.estimated_r,
            "alert_priority": self.alert_priority,
            "skip_reason": self.skip_reason,
        }


@dataclass
class CopilotBriefing:
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    trader_id: str = ""

    # Ranked opportunities (A first)
    opportunities: list[SymbolGrade] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    # Behavioral state summary
    behavioral_summary: str = ""
    mistake_patterns: str = ""
    session_limit_reached: bool = False

    # LLM battle plan
    battle_plan: str = ""
    plan_generated: bool = False

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "trader_id": self.trader_id,
            "top_opportunities": [o.to_dict() for o in self.opportunities],
            "skipped_symbols": self.skipped,
            "behavioral_summary": self.behavioral_summary,
            "mistake_patterns": self.mistake_patterns,
            "session_limit_reached": self.session_limit_reached,
            "battle_plan": self.battle_plan,
            "plan_generated": self.plan_generated,
        }


# ── Copilot ───────────────────────────────────────────────────────


class WorkflowCopilot:
    """
    Runs the 5-step pre-session pipeline and returns a CopilotBriefing.

    Usage:
        copilot = WorkflowCopilot(sync_db=sync_db, llm=llm_client)
        briefing = await copilot.prepare(trader_id, symbols=["RELIANCE", "HDFC"])
    """

    # Max symbols to evaluate (keeps runtime under ~10s)
    MAX_SYMBOLS = 12

    def __init__(self, sync_db: Any, llm: Any) -> None:
        self._db = sync_db      # sync SQLAlchemy session
        self._llm = llm

    async def prepare(
        self,
        trader_id: str,
        symbols: list[str],
        timeframe: str = "5m",
    ) -> CopilotBriefing:
        briefing = CopilotBriefing(trader_id=trader_id)

        # If caller passed no symbols, pull matched items from the trader's watchlists
        if not symbols:
            symbols = self._fetch_symbols_from_watchlist(trader_id)
            logger.info(
                "WorkflowCopilot: fetched %d symbols from DB watchlists for trader %s",
                len(symbols), trader_id,
            )

        symbols = symbols[: self.MAX_SYMBOLS]

        # ── Steps 1 & 2: behavioral context + mistakes (sync, fast) ──
        snap, mistake_text = self._behavioral_context(trader_id)
        briefing.mistake_patterns = mistake_text
        briefing.behavioral_summary = self._format_behavioral_summary(snap)

        if snap.todays_trade_count >= snap.max_daily_trades:
            briefing.session_limit_reached = True
        if snap.todays_pnl <= snap.max_daily_loss:
            briefing.session_limit_reached = True

        # ── Step 3: per-symbol scan (parallel async) ──────────────
        if symbols:
            tasks = [
                self._scan_symbol(sym, trader_id, timeframe)
                for sym in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, res in zip(symbols, results):
                if isinstance(res, Exception):
                    logger.warning("Copilot scan failed for %s: %s", sym, res)
                    briefing.skipped.append(
                        {"symbol": sym, "reason": f"Scan error: {type(res).__name__}"}
                    )
                elif res is None or res.grade == "skip":
                    briefing.skipped.append(
                        {
                            "symbol": sym,
                            "reason": res.skip_reason if res else "No data",
                        }
                    )
                else:
                    briefing.opportunities.append(res)

            # ── Step 4: rank (A > B > C, then readiness score) ────
            _order = {"A": 0, "B": 1, "C": 2}
            briefing.opportunities.sort(
                key=lambda o: (_order.get(o.grade, 9), -o.readiness_score)
            )

        # ── Step 5: one LLM call for narrative ────────────────────
        briefing.battle_plan = await self._synthesize(briefing, snap)
        briefing.plan_generated = bool(briefing.battle_plan)

        return briefing

    # ── Watchlist symbol resolution ──────────────────────────────

    def _fetch_symbols_from_watchlist(self, trader_id: str) -> list[str]:
        """
        Pull matched symbols from the trader's strategy watchlists in DB.
        Falls back to master watchlist active items if no strategy watchlists exist.
        Returns deduplicated list ordered by match_score desc.
        """
        try:
            from core.models import Watchlist, WatchlistItem
            from uuid import UUID

            trader_uuid = UUID(trader_id)

            # Prefer strategy watchlists (highest-quality matched items)
            strategy_wls = (
                self._db.query(Watchlist)
                .filter(
                    Watchlist.trader_id == trader_uuid,
                    Watchlist.watchlist_type == "strategy",
                    Watchlist.is_active == True,
                )
                .all()
            )

            if strategy_wls:
                wl_ids = [wl.id for wl in strategy_wls]
                status_filter = ["matched"]
            else:
                # Fallback: pull from master watchlists (not yet distributed)
                master_wls = (
                    self._db.query(Watchlist)
                    .filter(
                        Watchlist.trader_id == trader_uuid,
                        Watchlist.watchlist_type == "master",
                        Watchlist.is_active == True,
                    )
                    .all()
                )
                wl_ids = [wl.id for wl in master_wls]
                status_filter = ["active", "matched"]

            if not wl_ids:
                return []

            items = (
                self._db.query(WatchlistItem)
                .filter(
                    WatchlistItem.watchlist_id.in_(wl_ids),
                    WatchlistItem.status.in_(status_filter),
                )
                .order_by(WatchlistItem.match_score.desc().nullslast())
                .all()
            )

            seen: set[str] = set()
            result: list[str] = []
            for item in items:
                if item.symbol not in seen:
                    seen.add(item.symbol)
                    result.append(item.symbol)

            return result

        except Exception as e:
            logger.warning(
                "Failed to fetch watchlist symbols for trader %s: %s", trader_id, e
            )
            return []

    # ── Step 1 + 2 ───────────────────────────────────────────────

    def _behavioral_context(self, trader_id: str) -> tuple[Any, str]:
        """Returns (TraderSnapshot, mistake_summary_text)."""
        from agents.coplanner_context import CoPlannerContextAssembler, TraderSnapshot

        snap = TraderSnapshot()
        try:
            snap = CoPlannerContextAssembler(self._db).assemble_sync(trader_id)
        except Exception as e:
            logger.warning("Behavioral context failed: %s", e)

        mistake_text = ""
        try:
            from analytics.mistake_summarizer import LastWeekMistakeSummarizer
            mistake_text = LastWeekMistakeSummarizer(self._db).summarize(trader_id)
        except Exception as e:
            logger.debug("Mistake summarizer failed: %s", e)

        return snap, mistake_text

    def _format_behavioral_summary(self, snap: Any) -> str:
        parts = []
        if snap.streak_type == "losing" and abs(snap.current_streak) >= 2:
            parts.append(
                f"Loss streak: {abs(snap.current_streak)} in a row"
            )
        if snap.current_drawdown_pct >= 5:
            parts.append(f"Drawdown: {snap.current_drawdown_pct:.1f}%")
        if snap.discipline_score_avg < 6:
            parts.append(
                f"Discipline score low: {snap.discipline_score_avg:.1f}/10"
            )
        if snap.avg_trades_per_day_7d > 4:
            parts.append(
                f"High trade frequency: avg {snap.avg_trades_per_day_7d:.1f}/day"
            )
        return "; ".join(parts) if parts else "No significant behavioral flags"

    # ── Step 3: scan one symbol ───────────────────────────────────

    async def _scan_symbol(
        self, symbol: str, trader_id: str, timeframe: str
    ) -> Optional[SymbolGrade]:
        """Fetch candles, run SetupScanner, return graded SymbolGrade."""
        try:
            import pandas as pd
            from intelligence.readiness_scanner import SetupScanner as IntelScanner
            from intelligence.market_environment import MarketEnvironmentAnalyzer
            from strategy_engine.strategy_registry import get_registry
            from strategy_engine.conditions.registry import ConditionRegistry

            # Fetch candles
            df = await self._fetch_candles(symbol, timeframe)
            if df is None or len(df) < 20:
                return SymbolGrade(
                    symbol=symbol, grade="skip",
                    strategy_name="", conditions_met=0, conditions_total=0,
                    conditions=[], regime="unknown", regime_aligned=False,
                    readiness_score=0.0, estimated_r=None,
                    alert_priority="low",
                    skip_reason="Insufficient candle data",
                )

            # Build scanner
            registry = get_registry()
            if not registry.is_initialized:
                registry.initialize()

            scanner = IntelScanner(
                strategy_registry=registry,
                condition_registry=ConditionRegistry(),
                env_analyzer=MarketEnvironmentAnalyzer(),
            )

            # Run synchronous scan in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, scanner.scan, df, symbol, timeframe, None, trader_id
            )

            if not result.top_setup:
                return SymbolGrade(
                    symbol=symbol, grade="skip",
                    strategy_name="", conditions_met=0, conditions_total=0,
                    conditions=[], regime=result.environment.regime.value,
                    regime_aligned=False, readiness_score=0.0,
                    estimated_r=None, alert_priority="low",
                    skip_reason="No strategy conditions met",
                )

            top = result.top_setup
            regime_val = result.environment.regime.value if result.environment else "unknown"

            # Serialize conditions
            cond_list = [
                {
                    "name": c.name,
                    "is_met": c.is_met,
                    "proximity": c.proximity,
                    "notes": c.notes,
                }
                for c in top.conditions
            ]

            grade = self._grade(top)

            return SymbolGrade(
                symbol=symbol,
                grade=grade,
                strategy_name=top.strategy_name,
                conditions_met=top.conditions_met,
                conditions_total=top.conditions_total,
                conditions=cond_list,
                regime=regime_val,
                regime_aligned=top.regime_aligned,
                readiness_score=top.readiness_score,
                estimated_r=top.estimated_r_multiple,
                alert_priority=top.alert_priority,
            )

        except Exception as e:
            logger.warning("Symbol scan error %s: %s", symbol, e)
            return None

    async def _fetch_candles(self, symbol: str, timeframe: str):
        """Fetch OHLCV DataFrame via MarketDataService."""
        try:
            import pandas as pd
            from ingestion.market_data_service import get_market_data_service
            from datetime import date

            mds = get_market_data_service()
            if not mds:
                return None

            days = {
                "1m": 1, "5m": 3, "15m": 5, "30m": 10,
                "1h": 30, "1d": 120,
            }.get(timeframe, 5)

            to_date = date.today()
            from_date = to_date - timedelta(days=days)

            candles = []
            if mds.has_historical:
                try:
                    candles = mds._fetch_candles_raw(symbol, timeframe, from_date, to_date)
                except Exception as e:
                    logger.debug("Angel One fetch failed for %s: %s", symbol, e)

            if not candles and mds.has_live and mds._upstox:
                try:
                    from ingestion.broker_adapters.upstox import UpstoxInterval
                    _tf_map = {
                        "1m": "1minute", "5m": "30minute",
                        "15m": "30minute", "30m": "30minute",
                        "1h": "day", "1d": "day",
                    }
                    key = mds._upstox.resolve_instrument_key(symbol)
                    interval = UpstoxInterval(_tf_map.get(timeframe, "30minute"))
                    candles = mds._upstox.get_historical_candles(key, interval, from_date, to_date)
                except Exception as e:
                    logger.debug("Upstox fetch failed for %s: %s", symbol, e)

            if not candles:
                return None

            df = pd.DataFrame(candles)
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
            needed = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in needed):
                return None
            return df[needed].tail(200)

        except Exception as e:
            logger.debug("Candle fetch error for %s: %s", symbol, e)
            return None

    # ── Step 4: grading ──────────────────────────────────────────

    def _grade(self, setup: Any) -> str:
        """
        Grade a SetupReadiness as A / B / C / skip.

        Regime alignment is MANDATORY for A or B.
        A non-aligned setup can never be higher than C — it needs more conditions
        to be met before the regime question even matters, so it can't be actionable.

        A:    regime aligned + ≥75% conditions met + readiness ≥ 0.70
        B:    regime aligned + ≥60% conditions met + readiness ≥ 0.50
        C:    regime aligned + readiness ≥ 0.35  (forming — watch, don't act)
              OR not aligned + ≥80% conditions met + readiness ≥ 0.60  (edge-case: wait for regime shift)
        skip: regime not aligned, OR below all thresholds
        """
        r = setup.readiness_score
        ratio = setup.conditions_met_ratio
        aligned = setup.regime_aligned

        if aligned and ratio >= 0.75 and r >= 0.70:
            return "A"
        if aligned and ratio >= 0.60 and r >= 0.50:
            return "B"
        if aligned and r >= 0.35:
            return "C"
        # Regime mismatch — only show as C if nearly all conditions met
        # (useful for "watch for regime shift" monitoring)
        if not aligned and ratio >= 0.80 and r >= 0.60:
            return "C"
        return "skip"

    # ── Step 5: LLM synthesis ─────────────────────────────────────

    async def _synthesize(self, briefing: CopilotBriefing, snap: Any) -> str:
        """One LLM call — writes the narrative battle plan from structured data."""
        try:
            from agents.llm_client import LLMMessage

            opp_lines = []
            for o in briefing.opportunities[:5]:
                r_str = f"est R {o.estimated_r:.1f}x" if o.estimated_r else "R unknown"
                opp_lines.append(
                    f"  {o.symbol} [{o.grade}] — {o.strategy_name}, "
                    f"{o.conditions_met}/{o.conditions_total} conditions met, "
                    f"regime={o.regime} ({'aligned' if o.regime_aligned else 'NOT aligned'}), "
                    f"{r_str}\n"
                    f"{o.checklist_text}"
                )

            skipped_line = ", ".join(
                s["symbol"] for s in briefing.skipped[:6]
            )

            context = f"""
TRADER STATE:
  Today: {snap.todays_trade_count} trades, P&L ₹{snap.todays_pnl:,.0f}
  Streak: {snap.streak_type} ({abs(snap.current_streak) if snap.current_streak else 0} in a row)
  Drawdown: {snap.current_drawdown_pct:.1f}%
  Discipline: {snap.discipline_score_avg:.1f}/10
  Session limit reached: {briefing.session_limit_reached}

BEHAVIORAL FLAGS:
  {briefing.behavioral_summary or 'None'}

{briefing.mistake_patterns or '(No mistake patterns from last week)'}

WATCHLIST SCAN ({len(briefing.opportunities)} qualified out of {len(briefing.opportunities) + len(briefing.skipped)} symbols):
{chr(10).join(opp_lines) or '  None qualified'}

SKIPPED: {skipped_line or 'none'}
""".strip()

            prompt = (
                "You are this trader's experienced self, preparing their pre-session battle plan.\n"
                "Not an external advisor — you ARE them at their best, talking to their current self.\n\n"
                f"{context}\n\n"
                "CRITICAL RULES:\n"
                "- Any symbol marked 'NOT aligned' = REGIME MISMATCH. Never call it a trade. Say 'watch only — regime mismatch'.\n"
                "- FOCUS LIST: Grade-A and Grade-B ONLY. Grade-C goes in WATCH.\n"
                "- Reference the actual mistake data from 'Last week' — not generic advice.\n"
                "- Never invent statistics not in the context above.\n\n"
                "Output this EXACT format (use markdown — it renders in the UI):\n\n"
                "## Session Posture\n"
                "[1-2 sentences. Direct. Reference today's discipline score and streak.]\n\n"
                "## Focus List\n"
                "**SYMBOL** — Grade [A/B] | [Strategy name]\n"
                "  ✅ [X/Y conditions met] | Entry trigger: [exact price action needed]\n"
                "  ⚠️ Still needs: [one missing condition, if any]\n"
                "(repeat for each A/B grade symbol)\n\n"
                "## Rules for Today\n"
                "1. [Hard rule from last-week mistake — specific, not generic. Sound like YOUR rule, not advice.]\n"
                "2. [Second rule — same]\n\n"
                "## Skip\n"
                "[One sentence on why the skipped symbols aren't worth watching today — reference the specific reason (regime, low readiness, etc.)]\n\n"
                "Voice: Write as the trader's best self. 'Your EMA rule exists because...' not 'It is recommended that...'"
            )

            response = await self._llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.35,
                max_tokens=550,
                caller="workflow_copilot",
            )
            return response.content.strip()

        except Exception as e:
            logger.warning("LLM synthesis failed: %s", e)
            return self._fallback_plan(briefing, snap)

    def _fallback_plan(self, briefing: CopilotBriefing, snap: Any) -> str:
        """Deterministic plan when LLM is unavailable — structured markdown format."""
        lines = []

        # Session Posture
        lines.append("## Session Posture")
        if briefing.session_limit_reached:
            lines.append("Session limit reached — no new trades today. Review, don't trade.")
        elif "loss streak" in briefing.behavioral_summary.lower():
            lines.append("Loss streak active. Your rule: A-grade setups only, half size until streak breaks.")
        else:
            lines.append("Normal session. Trade your plan — no shortcuts, no chasing.")

        # Focus List
        lines.append("\n## Focus List")
        a_opps = [o for o in briefing.opportunities if o.grade == "A"]
        b_opps = [o for o in briefing.opportunities if o.grade == "B"]
        focus = a_opps + b_opps
        if focus:
            for o in focus[:4]:
                missing = [c["name"] for c in o.conditions if not c["is_met"]]
                missing_str = f"  ⚠️ Still needs: {', '.join(missing[:2])}" if missing else "  ✅ All conditions met"
                lines.append(
                    f"**{o.symbol}** — Grade {o.grade} | {o.strategy_name}\n"
                    f"  ✅ {o.conditions_met}/{o.conditions_total} conditions met\n"
                    f"{missing_str}"
                )
        else:
            lines.append("No A or B grade setups today. Consider sitting out or paper-trading.")

        # Rules
        lines.append("\n## Rules for Today")
        if briefing.mistake_patterns:
            pattern_line = briefing.mistake_patterns.strip().split("\n")[0]
            lines.append(f"1. {pattern_line}")
        else:
            lines.append("1. Only enter with confirmed entry trigger — no anticipation trades.")
        lines.append("2. If stop hit, close the trade. No averaging down, no hoping.")

        # Skip
        lines.append("\n## Skip")
        if briefing.skipped:
            skip_syms = ", ".join(s["symbol"] for s in briefing.skipped[:5])
            lines.append(f"{skip_syms} — regime mismatch or insufficient readiness. Not worth watching today.")
        else:
            lines.append("No symbols skipped — full watchlist qualified.")

        return "\n".join(lines)
