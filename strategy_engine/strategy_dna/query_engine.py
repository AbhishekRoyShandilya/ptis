# strategy_engine/strategy_dna/query_engine.py
"""
Query engine that AI agents use to ask questions about strategies.
This is the interface between Strategy DNA and the Intelligence Layer.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from strategy_engine.strategy_dna.schema import StrategyDNA
from strategy_engine.strategy_dna.loader import StrategyDNALoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Recommendation for current conditions."""
    strategy_name: str
    display_name: str
    fitness_score: float           # 0-100 how well this fits current conditions
    sizing_recommendation: str     # full, standard, half, quarter, skip
    notes: str
    warnings: List[str]
    alternative: Optional[str]     # suggested alternative if score is low


@dataclass
class PreTradeGuidance:
    """
    Complete guidance packet generated BEFORE a trade.
    Sent to frontend + coaching agent.
    """
    strategy_name: str
    regime_fitness: str            # "excellent", "good", "marginal", "poor"
    sizing_recommendation: str
    checklist: List[str]
    key_warnings: List[str]
    emotional_traps_to_watch: List[str]
    skip_conditions_active: List[str]
    coaching_message: str


class StrategyQueryEngine:
    """
    Central query interface for all strategy knowledge.

    AI agents call this to get strategy-aware intelligence:
    - "What strategies work in this regime?"
    - "Should I take this trade in current conditions?"
    - "What are the traps for this strategy right now?"
    - "What's the sizing for ORB in high volatility?"
    """

    def __init__(self, loader: StrategyDNALoader):
        self.loader = loader
        self._strategies = loader.get_all()

    def get_strategy(self, name: str) -> Optional[StrategyDNA]:
        return self._strategies.get(name)

    # ──────────────────────────────────────
    # REGIME-BASED QUERIES
    # ──────────────────────────────────────

    def get_strategies_for_regime(
        self,
        regime: str,
        session: Optional[str] = None,
    ) -> List[StrategyRecommendation]:
        """
        Rank ALL strategies by fitness for current regime.
        Used by: Regime Agent, Dashboard, Coaching Agent
        """
        recommendations = []

        for name, dna in self._strategies.items():
            fitness, sizing, notes, warnings = self._score_regime_fitness(
                dna, regime, session
            )

            alt = dna.get_fallback_for_regime(regime) if fitness < 40 else None

            recommendations.append(StrategyRecommendation(
                strategy_name=name,
                display_name=dna.display_name,
                fitness_score=fitness,
                sizing_recommendation=sizing,
                notes=notes,
                warnings=warnings,
                alternative=alt,
            ))

        # Sort by fitness score descending
        recommendations.sort(key=lambda r: r.fitness_score, reverse=True)
        return recommendations

    def get_best_strategy_for_regime(
        self,
        regime: str,
        session: Optional[str] = None,
    ) -> Optional[StrategyRecommendation]:
        """Get single best strategy for current conditions."""
        recs = self.get_strategies_for_regime(regime, session)
        return recs[0] if recs else None

    def should_trade_strategy_in_regime(
        self,
        strategy_name: str,
        regime: str,
    ) -> Tuple[bool, str, str]:
        """
        Quick check: should this strategy be traded right now?
        Returns: (should_trade, sizing, reason)
        """
        dna = self._strategies.get(strategy_name)
        if not dna:
            return True, "standard_size", "Strategy not found in DNA — no guidance"

        if dna.should_skip_in_regime(regime):
            alt = dna.get_fallback_for_regime(regime)
            reason = f"{dna.display_name} performs poorly in {regime}."
            if alt:
                reason += f" Consider {alt} instead."
            return False, "skip", reason

        sizing = dna.get_regime_sizing(regime)
        guidance = dna.get_regime_guidance(regime)
        notes = guidance.notes if guidance else "No specific guidance"

        return True, sizing, notes

    # ──────────────────────────────────────
    # PRE-TRADE GUIDANCE
    # ──────────────────────────────────────

    def get_pre_trade_guidance(
        self,
        strategy_name: str,
        current_regime: str,
        current_session: Optional[str] = None,
        trader_emotion: Optional[str] = None,
        consecutive_losses: int = 0,
        trades_today: int = 0,
    ) -> PreTradeGuidance:
        """
        Generate complete pre-trade guidance packet.
        Called when trader selects a strategy in the trade entry form.
        Displayed in frontend + used by coaching agent.
        """
        dna = self._strategies.get(strategy_name)
        if not dna:
            return PreTradeGuidance(
                strategy_name=strategy_name,
                regime_fitness="unknown",
                sizing_recommendation="standard_size",
                checklist=[],
                key_warnings=["Strategy not in DNA — no guidance available"],
                emotional_traps_to_watch=[],
                skip_conditions_active=[],
                coaching_message="Trade carefully — no strategy profile loaded.",
            )

        # Regime fitness
        fitness, sizing, notes, warnings = self._score_regime_fitness(
            dna, current_regime, current_session
        )

        if fitness >= 75:
            regime_fitness = "excellent"
        elif fitness >= 55:
            regime_fitness = "good"
        elif fitness >= 35:
            regime_fitness = "marginal"
        else:
            regime_fitness = "poor"

        # Emotional traps relevant right now
        active_traps = []
        for trap in dna.emotional_traps:
            # Check if current emotion matches known traps
            if trader_emotion:
                emotion_lower = trader_emotion.lower()
                trap_lower = trap.trap.lower()
                if any(word in trap_lower for word in [emotion_lower, "fomo", "revenge", "fear"]):
                    active_traps.append(f"⚠️ {trap.trap}: {trap.solution}")
            if trap.frequency == "happens_often":
                active_traps.append(f"🔄 Common trap: {trap.trap} → {trap.solution}")

        # Check "do not trade when" conditions
        do_not_trade_active = []
        if trader_emotion and trader_emotion.lower() in ["revenge", "frustrated", "fomo", "euphoric"]:
            for condition in dna.do_not_trade_when:
                if any(word in condition.lower() for word in [
                    trader_emotion.lower(), "frustrat", "revenge", "euphori", "fomo"
                ]):
                    do_not_trade_active.append(condition)

        if consecutive_losses >= 2:
            do_not_trade_active.append(
                f"You have {consecutive_losses} consecutive losses. "
                f"Consider stepping away."
            )

        # Active skip conditions
        skip_active = []
        for skip in dna.skip_conditions:
            # Simple keyword matching for regime/session
            skip_lower = skip.lower()
            if current_regime and current_regime.lower() in skip_lower:
                skip_active.append(skip)

        # Generate coaching message
        coaching = self._generate_pre_trade_coaching(
            dna, regime_fitness, sizing, active_traps,
            do_not_trade_active, skip_active, trader_emotion,
            consecutive_losses, trades_today, current_regime,
        )

        return PreTradeGuidance(
            strategy_name=strategy_name,
            regime_fitness=regime_fitness,
            sizing_recommendation=sizing,
            checklist=dna.checklist,
            key_warnings=warnings + [f"⚠️ {s}" for s in skip_active],
            emotional_traps_to_watch=active_traps[:3],
            skip_conditions_active=do_not_trade_active,
            coaching_message=coaching,
        )

    # ──────────────────────────────────────
    # AGENT CONTEXT GENERATION
    # ──────────────────────────────────────

    def get_agent_context(self, strategy_name: str) -> str:
        """
        Get full strategy context for LLM agent prompt injection.
        Used by: Coaching Agent, Strategy Agent, Behavior Agent
        """
        dna = self._strategies.get(strategy_name)
        if not dna:
            return f"No strategy profile found for '{strategy_name}'"
        return dna.to_agent_context()

    def get_regime_context_for_agent(
        self,
        strategy_name: str,
        current_regime: str,
    ) -> str:
        """
        Get regime-specific context for agent prompts.
        Used by: Regime Agent, Performance Agent
        """
        dna = self._strategies.get(strategy_name)
        if not dna:
            return ""
        return dna.to_regime_summary(current_regime)

    def get_all_strategies_regime_matrix(
        self,
        current_regime: str,
    ) -> str:
        """
        Generate a full regime matrix for all strategies.
        Used by: Dashboard, Regime Agent
        """
        lines = [f"STRATEGY FITNESS FOR REGIME: {current_regime.upper()}\n"]
        lines.append(f"{'Strategy':<30} {'Fitness':<10} {'Sizing':<20} {'Notes'}")
        lines.append("─" * 90)

        recs = self.get_strategies_for_regime(current_regime)
        for rec in recs:
            emoji = "✅" if rec.fitness_score >= 60 else "⚠️" if rec.fitness_score >= 35 else "❌"
            lines.append(
                f"{emoji} {rec.display_name:<28} "
                f"{rec.fitness_score:<10.0f} "
                f"{rec.sizing_recommendation:<20} "
                f"{rec.notes[:40]}"
            )

        return "\n".join(lines)

    def get_emotional_guidance(
        self,
        strategy_name: str,
        current_emotion: str,
    ) -> List[str]:
        """
        Get emotion-specific guidance for a strategy.
        Used by: Behavior Agent, Coaching Agent
        """
        dna = self._strategies.get(strategy_name)
        if not dna:
            return []

        guidance = []

        # Check do_not_trade_when
        for condition in dna.do_not_trade_when:
            if current_emotion.lower() in condition.lower():
                guidance.append(f"🚨 YOUR RULE: {condition}")

        # Check traps
        for trap in dna.emotional_traps:
            if current_emotion.lower() in trap.trap.lower() or \
               current_emotion.lower() in trap.description.lower():
                guidance.append(f"⚠️ TRAP: {trap.trap} → SOLUTION: {trap.solution}")

        # General emotional guidance
        if current_emotion.lower() in ["revenge", "frustrated"]:
            guidance.append(
                "🧊 Step away from the screen for 10 minutes. "
                "Your next trade should be your BEST setup, not a reaction."
            )
        elif current_emotion.lower() in ["fomo", "anxious"]:
            guidance.append(
                "⏸️ FOMO trades are statistically your worst performers. "
                "If you have to chase, it's not your entry."
            )
        elif current_emotion.lower() in ["euphoric"]:
            guidance.append(
                "📏 Euphoria leads to oversizing. "
                "Stick to standard position size regardless of how you feel."
            )

        return guidance

    # ──────────────────────────────────────
    # STRATEGY COMPARISON
    # ──────────────────────────────────────

    def compare_strategies(
        self,
        strategy_names: List[str],
        regime: str,
    ) -> Dict:
        """
        Compare multiple strategies for current regime.
        Used by: Dashboard, when trader is deciding what to trade.
        """
        comparison = {}
        for name in strategy_names:
            dna = self._strategies.get(name)
            if not dna:
                continue

            fitness, sizing, notes, warnings = self._score_regime_fitness(dna, regime)
            guidance = dna.get_regime_guidance(regime)

            comparison[name] = {
                "display_name": dna.display_name,
                "fitness": fitness,
                "sizing": sizing,
                "expected_win_rate": guidance.expected_win_rate if guidance else "unknown",
                "notes": notes,
                "warnings": warnings,
                "checklist_items": len(dna.checklist),
                "known_traps": len(dna.emotional_traps),
                "category": dna.category,
            }

        return comparison

    # ──────────────────────────────────────
    # PRIVATE METHODS
    # ──────────────────────────────────────

    def _score_regime_fitness(
        self,
        dna: StrategyDNA,
        regime: str,
        session: Optional[str] = None,
    ) -> Tuple[float, str, str, List[str]]:
        """
        Score how well a strategy fits current regime/session.
        Returns: (fitness_score 0-100, sizing, notes, warnings)
        """
        score = 50  # neutral baseline
        notes_parts = []
        warnings: List[str] = []

        # ───── REGIME FIT ─────
        for r in dna.market_conditions.works_best_regimes:
            if r.regime == regime:
                score += 30
                if r.notes:
                    notes_parts.append(r.notes)
                break

        for r in dna.market_conditions.works_poorly_regimes:
            if r.regime == regime:
                score -= 35
                if r.recommendation:
                    warnings.append(r.recommendation)
                if r.notes:
                    notes_parts.append(f"CAUTION: {r.notes}")
                break

        # ───── SESSION FIT ─────
        if session:
            best_sessions = [
                s.get("name", "") for s in dna.market_conditions.works_best_sessions
            ]
            poor_sessions = [
                s.get("name", "") for s in dna.market_conditions.works_poorly_sessions
            ]

            if session in best_sessions:
                score += 10
                notes_parts.append(f"Performs well during {session}")
            elif session in poor_sessions:
                score -= 15
                warnings.append(f"Avoid trading during {session}")
                notes_parts.append(f"Weak performance in {session}")

        # ───── CLAMP SCORE ─────
        score = max(0, min(100, score))

         # Determine sizing from DNA
        sizing = dna.get_regime_sizing(regime)
        if not sizing or sizing == "":
            # Derive from score
            if score >= 70:
                sizing = "full_size"
            elif score >= 50:
                sizing = "standard_size"
            elif score >= 30:
                sizing = "half_size"
            else:
                sizing = "skip"

        notes = " | ".join(notes_parts) if notes_parts else "No specific regime data"

        return score, sizing, notes, warnings

    def _generate_pre_trade_coaching(
        self,
        dna: StrategyDNA,
        regime_fitness: str,
        sizing: str,
        active_traps: List[str],
        do_not_trade: List[str],
        skip_active: List[str],
        emotion: Optional[str],
        consecutive_losses: int,
        trades_today: int,
        current_regime: str,
    ) -> str:
        """Generate a coaching message before trade entry."""

        parts = []

        # ── HARD STOP conditions ──
        if do_not_trade:
            parts.append(
                "🚨 STOP — Your own rules say DO NOT TRADE right now:\n" +
                "\n".join(f"  • {d}" for d in do_not_trade)
            )
            return "\n\n".join(parts)

        # ── Regime fitness ──
        if regime_fitness == "poor":
            alt = dna.get_fallback_for_regime(current_regime)
            msg = (
                f"⚠️ {dna.display_name} has POOR fitness in {current_regime} regime. "
                f"Your historical data shows this leads to losses."
            )
            if alt:
                msg += f"\n💡 Consider: {alt} works better in this regime."
            parts.append(msg)

        elif regime_fitness == "marginal":
            parts.append(
                f"⚠️ {dna.display_name} is MARGINAL in {current_regime}. "
                f"Recommended sizing: {sizing}. "
                f"Only take A+ setups."
            )

        elif regime_fitness == "excellent":
            parts.append(
                f"✅ {dna.display_name} is EXCELLENT in {current_regime}. "
                f"This is your edge zone. Execute with confidence."
            )

        # ── Sizing guidance ──
        if sizing in ("skip", "skip_or_quarter_size"):
            parts.append("🛑 Sizing recommendation: SKIP this trade.")
        elif sizing == "half_size":
            parts.append("📏 Sizing recommendation: HALF SIZE — conditions not ideal.")
        elif sizing == "quarter_size":
            parts.append("📏 Sizing recommendation: QUARTER SIZE — marginal setup.")

        # ── Emotional traps ──
        if active_traps:
            parts.append(
                "🧠 Watch for these traps:\n" +
                "\n".join(f"  {t}" for t in active_traps[:2])
            )

        # ── Skip conditions ──
        if skip_active:
            parts.append(
                "⏭️ Skip conditions may be active:\n" +
                "\n".join(f"  • {s}" for s in skip_active)
            )

        # ── Loss streak ──
        if consecutive_losses >= 3:
            parts.append(
                f"🔴 You have {consecutive_losses} consecutive losses. "
                f"Your rules say STOP after 3. Take a break."
            )
        elif consecutive_losses == 2:
            parts.append(
                "🟡 2 consecutive losses. Next trade must be an A+ setup only."
            )

        # ── Trade count ──
        max_trades = 6  # configurable
        if trades_today >= max_trades:
            parts.append(
                f"🛑 You've taken {trades_today} trades today. "
                f"Consider stopping — overtrading erodes your edge."
            )
        elif trades_today >= max_trades - 2:
            parts.append(
                f"ℹ️ {trades_today} trades today. Be selective with remaining trades."
            )

        if not parts:
            parts.append(
                f"👍 {dna.display_name} — conditions look reasonable. "
                f"Follow your checklist and execute cleanly."
            )

        return "\n\n".join(parts)      