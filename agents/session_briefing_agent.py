"""
Session Briefing Agent
======================
Agentic "Prepare My Session" workflow that runs before market open (or on demand)
and produces a fully-synthesized battle plan for the day.

Pipeline (runs in parallel where possible):
  1. Behavioral context   — TraderSnapshot: streak, drawdown, discipline score
  2. Strategy readiness   — Which of the user's strategies are regime-fit today?
  3. Watchlist evaluation — For each symbol: setup detected? which strategy fits?
                            R:R estimate, regime alignment score
  4. LLM synthesis        — Weaves all of the above into a prioritized battle plan:
                            "Focus on RELIANCE ORB (A-grade), skip HDFC (regime mismatch)"

Result is a structured BriefingReport with:
  - top_opportunities   : ranked list of symbol+setup combos to watch
  - strategies_to_use   : which strategies fit today's market conditions
  - behavioral_flags    : any warnings (loss streak, overtrading risk, etc.)
  - battle_plan         : LLM-written narrative battle plan for the session
  - skipped_symbols     : symbols that don't qualify and why
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────


@dataclass
class SymbolOpportunity:
    symbol: str
    grade: str                      # A / B / C / skip
    setup_type: Optional[str]       # "orb_breakout", "vwap_bounce", "ema_pullback", etc.
    strategy_id: Optional[str]      # matched strategy template id
    regime: str                     # current regime for this symbol
    regime_fit: str                 # "strong" / "ok" / "weak" / "mismatch"
    estimated_r: float              # expected R-multiple
    confidence: float               # 0–1
    notes: str                      # human-readable rationale
    raw_data: dict = field(default_factory=dict)


@dataclass
class BriefingReport:
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trader_id: str = ""

    # Ranked opportunities (A first)
    top_opportunities: list[SymbolOpportunity] = field(default_factory=list)
    skipped_symbols: list[dict[str, str]] = field(default_factory=list)

    # Strategy fit for today
    strategies_ready: list[str] = field(default_factory=list)     # regime-aligned
    strategies_avoid: list[str] = field(default_factory=list)     # regime mismatch

    # Behavioral state
    behavioral_flags: list[str] = field(default_factory=list)
    session_limit_reached: bool = False

    # LLM-written narrative
    battle_plan: str = ""
    plan_generated: bool = False

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "trader_id": self.trader_id,
            "top_opportunities": [
                {
                    "symbol": o.symbol,
                    "grade": o.grade,
                    "setup_type": o.setup_type,
                    "strategy_id": o.strategy_id,
                    "regime": o.regime,
                    "regime_fit": o.regime_fit,
                    "estimated_r": o.estimated_r,
                    "confidence": o.confidence,
                    "notes": o.notes,
                }
                for o in self.top_opportunities
            ],
            "skipped_symbols": self.skipped_symbols,
            "strategies_ready": self.strategies_ready,
            "strategies_avoid": self.strategies_avoid,
            "behavioral_flags": self.behavioral_flags,
            "session_limit_reached": self.session_limit_reached,
            "battle_plan": self.battle_plan,
            "plan_generated": self.plan_generated,
        }


# ── Agent ────────────────────────────────────────────────────────


class SessionBriefingAgent:
    """
    Runs the full session-briefing pipeline and returns a BriefingReport.

    Usage:
        agent = SessionBriefingAgent(db=db, llm=get_llm_client())
        report = await agent.brief(trader_id, symbols=["RELIANCE", "HDFC", "NIFTY"])
    """

    def __init__(self, db: Any, llm: Any) -> None:
        self._db = db
        self._llm = llm

    async def brief(
        self,
        trader_id: str,
        symbols: list[str],
        candles_by_symbol: Optional[dict[str, list[dict]]] = None,
    ) -> BriefingReport:
        """
        Run the full session briefing pipeline.

        candles_by_symbol: optional pre-fetched candles {symbol: [bars]}.
          If None, the agent fetches candles via MarketDataService.
        """
        report = BriefingReport(trader_id=trader_id)

        # ── Step 1: Behavioral context (fast, sync DB read) ──────
        snap = await self._get_behavioral_context(trader_id)

        # Behavioral flags
        if snap.streak_type == "losing" and abs(snap.current_streak) >= 3:
            report.behavioral_flags.append(
                f"Loss streak: {abs(snap.current_streak)} in a row — "
                "consider reducing size or skipping marginal setups."
            )
        if snap.todays_trade_count >= snap.max_daily_trades:
            report.behavioral_flags.append(
                f"Daily trade limit reached ({snap.max_daily_trades}) — no new trades."
            )
            report.session_limit_reached = True
        if snap.todays_pnl <= snap.max_daily_loss:
            report.behavioral_flags.append(
                f"Daily loss limit hit (₹{snap.todays_pnl:,.0f}) — stop trading for today."
            )
            report.session_limit_reached = True
        if snap.current_drawdown_pct >= 10:
            report.behavioral_flags.append(
                f"Drawdown at {snap.current_drawdown_pct:.1f}% — trade smaller or pause."
            )
        if snap.discipline_score_avg < 5.0:
            report.behavioral_flags.append(
                "Discipline score is low — review override patterns before trading."
            )

        # ── Step 2: Strategy registry fit ────────────────────────
        regime_for_snapshot = snap.regime or "trending"
        strategies_ready, strategies_avoid = self._score_strategies(regime_for_snapshot)
        report.strategies_ready = strategies_ready
        report.strategies_avoid = strategies_avoid

        # ── Step 3: Parallel symbol evaluation ───────────────────
        if symbols:
            tasks = [
                self._evaluate_symbol(s, candles_by_symbol)
                for s in symbols[:10]   # cap at 10 to avoid LLM overload
            ]
            results: list[SymbolOpportunity | None] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            for sym, res in zip(symbols[:10], results):
                if isinstance(res, Exception):
                    logger.warning("Symbol eval failed for %s: %s", sym, res)
                    report.skipped_symbols.append(
                        {"symbol": sym, "reason": f"Evaluation error: {res}"}
                    )
                elif res is None:
                    report.skipped_symbols.append(
                        {"symbol": sym, "reason": "No qualifying setup detected"}
                    )
                elif res.grade == "skip":
                    report.skipped_symbols.append(
                        {"symbol": sym, "reason": res.notes}
                    )
                else:
                    report.top_opportunities.append(res)

            # Sort: A > B > C, then by confidence desc
            grade_order = {"A": 0, "B": 1, "C": 2}
            report.top_opportunities.sort(
                key=lambda o: (grade_order.get(o.grade, 9), -o.confidence)
            )

        # ── Step 4: LLM battle plan synthesis ────────────────────
        report.battle_plan = await self._synthesize_battle_plan(report, snap)
        report.plan_generated = bool(report.battle_plan)

        return report

    # ── Internal: behavioral context ─────────────────────────────

    async def _get_behavioral_context(self, trader_id: str):
        """Fetch TraderSnapshot — uses sync DB methods (co-planner context assembler)."""
        try:
            from agents.coplanner_context import CoPlannerContextAssembler
            assembler = CoPlannerContextAssembler(self._db)
            return assembler.assemble_sync(trader_id)
        except Exception as e:
            logger.warning("Behavioral context fetch failed: %s", e)
            from agents.coplanner_context import TraderSnapshot
            return TraderSnapshot()

    # ── Internal: strategy scoring ────────────────────────────────

    def _score_strategies(self, regime: str) -> tuple[list[str], list[str]]:
        """
        Check which strategy templates are regime-aligned today.
        Returns (ready_list, avoid_list).
        """
        try:
            from strategy_engine.strategy_registry import get_registry
            reg = get_registry()
            templates = reg.list_templates() if hasattr(reg, "list_templates") else {}

            if not templates:
                # Fallback: use YAML file names
                import os
                strat_dir = "config/strategies"
                if os.path.isdir(strat_dir):
                    templates = {
                        f.replace(".yaml", "").replace(".yml", ""): {}
                        for f in os.listdir(strat_dir)
                        if f.endswith((".yaml", ".yml"))
                    }

            # Regime-to-strategy heuristic map
            regime_good: dict[str, list[str]] = {
                "trending":         ["ema_pullback", "orb", "breakout", "momentum"],
                "breakout":         ["orb", "breakout", "vwap_bounce"],
                "ranging":          ["mean_revert", "vwap_bounce", "range_bound"],
                "high_volatility":  ["orb", "volatility_breakout"],
                "low_volatility":   ["range_bound", "mean_revert"],
                "mean_reverting":   ["vwap_bounce", "mean_revert", "range_bound"],
            }
            regime_bad: dict[str, list[str]] = {
                "trending":         ["range_bound", "mean_revert"],
                "ranging":          ["orb", "breakout", "momentum"],
                "high_volatility":  ["range_bound", "mean_revert"],
                "low_volatility":   ["orb", "volatility_breakout"],
            }

            regime_key = regime.lower().replace(" ", "_")
            good_keywords = regime_good.get(regime_key, [])
            bad_keywords = regime_bad.get(regime_key, [])

            ready, avoid = [], []
            for name in templates:
                nl = name.lower()
                if any(kw in nl for kw in good_keywords):
                    ready.append(name)
                elif any(kw in nl for kw in bad_keywords):
                    avoid.append(name)
                else:
                    ready.append(name)  # neutral — include as candidate

            return ready, avoid
        except Exception as e:
            logger.warning("Strategy scoring failed: %s", e)
            return [], []

    # ── Internal: symbol evaluation ───────────────────────────────

    async def _evaluate_symbol(
        self,
        symbol: str,
        candles_by_symbol: Optional[dict[str, list[dict]]],
    ) -> Optional[SymbolOpportunity]:
        """Evaluate a single symbol for session readiness."""
        try:
            candles = None
            if candles_by_symbol:
                candles = candles_by_symbol.get(symbol)

            if candles is None:
                candles = await self._fetch_candles(symbol)

            if not candles or len(candles) < 20:
                return SymbolOpportunity(
                    symbol=symbol, grade="skip", setup_type=None,
                    strategy_id=None, regime="unknown", regime_fit="unknown",
                    estimated_r=0.0, confidence=0.0,
                    notes="Insufficient candle data",
                )

            # Detect setup via setup scanner detectors
            setup_signal = self._detect_best_setup(symbol, candles)

            # Get regime
            regime = self._detect_regime(candles)

            # Score the opportunity
            return self._score_opportunity(symbol, setup_signal, regime, candles)

        except Exception as e:
            logger.warning("Symbol evaluation error for %s: %s", symbol, e)
            return None

    async def _fetch_candles(self, symbol: str) -> list[dict]:
        """Fetch recent candles from MarketDataService."""
        try:
            from ingestion.market_data_service import get_market_data_service
            from datetime import timedelta
            import datetime as dt
            mds = get_market_data_service()
            if not mds:
                return []
            to_date = dt.date.today()
            from_date = to_date - timedelta(days=3)
            candles = mds._fetch_candles_raw(symbol, "5m", from_date, to_date)
            return candles or []
        except Exception as e:
            logger.debug("Candle fetch for %s: %s", symbol, e)
            return []

    def _detect_best_setup(
        self, symbol: str, candles: list[dict]
    ) -> Optional[dict]:
        """Use setup scanner detectors to find the best setup."""
        try:
            from automation.setup_scanner import SetupScanner
            scanner = SetupScanner.__new__(SetupScanner)
            # ORB check
            orb = scanner.detect_orb_setup(symbol, candles)
            if orb:
                return {"setup_type": orb.setup_type, "direction": orb.direction,
                        "confidence": orb.confidence, "notes": orb.notes}
            # VWAP check
            vwap = scanner.detect_vwap_setup(symbol, candles)
            if vwap:
                return {"setup_type": vwap.setup_type, "direction": vwap.direction,
                        "confidence": vwap.confidence, "notes": vwap.notes}
            # Candle pattern check
            candle_sig = scanner.detect_candle_pattern(symbol, candles)
            if candle_sig:
                return {"setup_type": candle_sig.setup_type, "direction": candle_sig.direction,
                        "confidence": candle_sig.confidence, "notes": candle_sig.notes}
            return None
        except Exception as e:
            logger.debug("Setup detection for %s: %s", symbol, e)
            return None

    def _detect_regime(self, candles: list[dict]) -> str:
        """Lightweight regime detection from candle data."""
        try:
            import pandas as pd
            from strategy_engine.regime_detector import RegimeDetector
            df = pd.DataFrame(candles)
            if df.empty or "close" not in df.columns:
                return "unknown"
            detector = RegimeDetector()
            result = detector.detect(df)
            if hasattr(result, "regime"):
                return result.regime.value if hasattr(result.regime, "value") else str(result.regime)
            return str(result)
        except Exception:
            return "unknown"

    def _score_opportunity(
        self,
        symbol: str,
        setup_signal: Optional[dict],
        regime: str,
        candles: list[dict],
    ) -> SymbolOpportunity:
        """Convert setup signal + regime into a graded opportunity."""
        if not setup_signal:
            return SymbolOpportunity(
                symbol=symbol, grade="skip", setup_type=None,
                strategy_id=None, regime=regime, regime_fit="unknown",
                estimated_r=0.0, confidence=0.0,
                notes="No qualifying setup",
            )

        setup_type = setup_signal.get("setup_type", "")
        confidence = float(setup_signal.get("confidence", 0.5))
        direction = setup_signal.get("direction", "")
        notes = setup_signal.get("notes", "")

        # Map setup → strategy id
        setup_to_strategy = {
            "orb_breakout": "orb", "orb_breakdown": "orb",
            "vwap_bounce": "vwap_bounce", "vwap_rejection": "vwap_bounce",
            "bullish_engulfing": "ema_pullback", "bearish_engulfing": "ema_pullback",
            "pullback": "ema_pullback", "reversal": "mean_revert",
        }
        strategy_id = setup_to_strategy.get(setup_type)

        # Regime fit heuristic
        regime_fit_map: dict[tuple[str, str], str] = {
            ("orb_breakout", "trending"): "strong",
            ("orb_breakout", "breakout"): "strong",
            ("orb_breakout", "ranging"): "weak",
            ("vwap_bounce", "ranging"): "strong",
            ("vwap_bounce", "trending"): "ok",
            ("vwap_bounce", "high_volatility"): "weak",
        }
        regime_fit = regime_fit_map.get(
            (setup_type, regime.lower()), "ok"
        )

        # Estimate R based on setup type + regime fit
        base_r = {"orb_breakout": 2.5, "orb_breakdown": 2.0, "vwap_bounce": 1.8,
                  "vwap_rejection": 1.5, "pullback": 2.0, "reversal": 1.5}.get(setup_type, 1.5)
        fit_multiplier = {"strong": 1.2, "ok": 1.0, "weak": 0.7, "mismatch": 0.4}.get(regime_fit, 1.0)
        estimated_r = round(base_r * fit_multiplier, 1)

        # Grade
        if regime_fit == "strong" and confidence >= 0.7 and estimated_r >= 2.0:
            grade = "A"
        elif regime_fit in ("strong", "ok") and confidence >= 0.5 and estimated_r >= 1.5:
            grade = "B"
        elif regime_fit == "mismatch" or estimated_r < 1.0:
            grade = "skip"
        else:
            grade = "C"

        full_notes = f"{direction.upper()} — {notes}" if direction else notes

        return SymbolOpportunity(
            symbol=symbol,
            grade=grade,
            setup_type=setup_type,
            strategy_id=strategy_id,
            regime=regime,
            regime_fit=regime_fit,
            estimated_r=estimated_r,
            confidence=confidence,
            notes=full_notes,
        )

    # ── Internal: LLM synthesis ───────────────────────────────────

    async def _synthesize_battle_plan(
        self, report: BriefingReport, snap: Any
    ) -> str:
        """Call LLM to generate the narrative battle plan."""
        try:
            from agents.llm_client import LLMMessage

            # Build a concise data dump for the LLM
            opp_lines = []
            for o in report.top_opportunities:
                opp_lines.append(
                    f"  • {o.symbol} [{o.grade}] — {o.setup_type or 'No setup'}, "
                    f"regime={o.regime} ({o.regime_fit}), "
                    f"est R={o.estimated_r}x, confidence={o.confidence:.0%}. {o.notes}"
                )
            skipped_lines = [
                f"  • {s['symbol']}: {s['reason']}"
                for s in report.skipped_symbols[:5]
            ]
            flag_lines = [f"  ⚠ {f}" for f in report.behavioral_flags]

            context = f"""
TRADER STATE:
  Today: {snap.todays_trade_count} trades, P&L ₹{snap.todays_pnl:,.0f},
  Streak: {snap.streak_type} {abs(snap.current_streak) if snap.current_streak else 0},
  Drawdown: {snap.current_drawdown_pct:.1f}%,
  Discipline score: {snap.discipline_score_avg:.1f}/10,
  Preferred strategies: {', '.join(snap.preferred_strategies) or 'not set'}

BEHAVIORAL FLAGS:
{chr(10).join(flag_lines) or '  None'}

STRATEGIES FIT FOR TODAY ({snap.regime or 'unknown'} regime):
  Ready: {', '.join(report.strategies_ready[:6]) or 'none'}
  Avoid: {', '.join(report.strategies_avoid[:4]) or 'none'}

WATCHLIST OPPORTUNITIES ({len(report.top_opportunities)} qualified):
{chr(10).join(opp_lines) or '  None'}

SKIPPED ({len(report.skipped_symbols)} symbols):
{chr(10).join(skipped_lines) or '  None'}
""".strip()

            prompt = f"""You are a professional trading co-pilot preparing a pre-session briefing.

Here is today's situation:
{context}

Write a concise, direct session battle plan (200–300 words). Structure it as:
1. **Session Posture** — How aggressive/conservative should the trader be today? (2 sentences)
2. **Focus List** — Which 1–3 symbols to watch and exactly why (grade A and B only). Be specific about the setup and entry trigger.
3. **Strategy of the Day** — Which strategy template fits best and why.
4. **Rules for Today** — 2–3 hard rules given the behavioral flags (if any).
5. **What to Skip** — Briefly mention regime-mismatched or low-grade symbols to avoid.

Be direct, specific, and actionable. No fluff. Use plain text (no markdown headers in the response)."""

            response = await self._llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.4,
                max_tokens=500,
                caller="session_briefing_agent",
            )
            return response.content.strip()

        except Exception as e:
            logger.warning("LLM battle plan synthesis failed: %s", e)
            # Fallback: rule-based plan
            return self._fallback_plan(report, snap)

    def _fallback_plan(self, report: BriefingReport, snap: Any) -> str:
        """Deterministic fallback when LLM is unavailable."""
        lines = []

        # Posture
        if report.session_limit_reached:
            lines.append("SESSION POSTURE: Limit reached — no new trades today.")
        elif report.behavioral_flags:
            lines.append("SESSION POSTURE: Caution mode — reduce size, only A-grade setups.")
        else:
            lines.append(f"SESSION POSTURE: Normal — {snap.streak_type} streak, trade your plan.")

        # Focus
        a_grade = [o for o in report.top_opportunities if o.grade == "A"]
        if a_grade:
            focus = ", ".join(
                f"{o.symbol} ({o.setup_type or 'setup'}, est R {o.estimated_r}x)"
                for o in a_grade[:3]
            )
            lines.append(f"FOCUS: {focus}")
        else:
            lines.append("FOCUS: No A-grade setups today — consider sitting out or paper trading.")

        # Strategy
        if report.strategies_ready:
            lines.append(f"STRATEGY TODAY: {report.strategies_ready[0]}")

        # Behavioral rules
        for flag in report.behavioral_flags[:2]:
            lines.append(f"RULE: {flag}")

        # Skips
        if report.skipped_symbols:
            skips = ", ".join(s["symbol"] for s in report.skipped_symbols[:4])
            lines.append(f"SKIP: {skips}")

        return "\n".join(lines)
