# behavior_agent.py
# agents/behavior_agent.py
"""
Behavioral Analysis Agent.
Detects FOMO, revenge, emotional patterns using Strategy DNA knowledge.
"""

from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from agents.base_agent import BaseAgent, AgentInsight
from core.models import Trade, JournalEntry
from core.enums import TradeStatus, EmotionalState
from strategy_engine.strategy_dna.query_engine import StrategyQueryEngine
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class BehaviorAgent(BaseAgent):
    """
    2️⃣ Behavioral Analysis Agent

    Uses Strategy DNA to understand STRATEGY-SPECIFIC behavioral traps:
    - ORB → chasing missed breakouts
    - VWAP Bounce → anticipating bounces
    - General → revenge trading, FOMO, overtrading

    DNA tells us what to look for. Data tells us when it's happening.
    """

    def __init__(self, strategy_query: StrategyQueryEngine, db: Session):
        super().__init__("behavior_agent", strategy_query, db)

    def analyze(self, context: Dict) -> List[AgentInsight]:
        insights = []

        trade_id = context.get("trade_id")
        if trade_id:
            insights.extend(self._analyze_trade_behavior(trade_id))

        insights.extend(self._detect_revenge_trading(context))
        insights.extend(self._detect_fomo_pattern(context))
        insights.extend(self._detect_overtrading(context))
        insights.extend(self._detect_emotional_patterns(context))
        insights.extend(self._check_strategy_specific_traps(context))

        return insights

    def _analyze_trade_behavior(self, trade_id: str) -> List[AgentInsight]:
        """Analyze behavioral signals for a specific trade."""
        insights = []

        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return insights

        # ── Check if this is a rapid re-entry after loss ──
        window = timedelta(minutes=settings.REVENGE_TRADE_WINDOW_MINUTES)
        recent_loss = self.db.query(Trade).filter(
            and_(
                Trade.entry_time >= trade.entry_time - window,
                Trade.entry_time < trade.entry_time,
                Trade.status == TradeStatus.CLOSED,
                Trade.symbol == trade.symbol,
            )
        ).all()

        losing_trades = [t for t in recent_loss if t.pnl and t.pnl < 0]
        if losing_trades:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="critical",
                title="Possible Revenge Trade Detected",
                message=(
                    f"You entered {trade.symbol} within "
                    f"{settings.REVENGE_TRADE_WINDOW_MINUTES} minutes of a loss "
                    f"on the same symbol. This matches revenge trading pattern. "
                    f"Your DNA says: Take a 15-min break after any loss."
                ),
                trade_ids=[trade_id] + [t.id for t in losing_trades],
                strategy_name=trade.strategy_name,
                action_suggestion="Wait 15 minutes and reassess with fresh eyes",
            ))

        # ── Check emotional state from journal ──
        if trade.journal_entry:
            journal = trade.journal_entry
            risky_emotions = {
                EmotionalState.REVENGE, EmotionalState.FOMO,
                EmotionalState.FRUSTRATED, EmotionalState.EUPHORIC,
                EmotionalState.ANXIOUS
            }

            if journal.pre_trade_emotion in risky_emotions:
                # Get strategy-specific emotional guidance
                emotion_str = journal.pre_trade_emotion.value
                guidance = []
                if trade.strategy_name:
                    guidance = self.strategy_query.get_emotional_guidance(
                        trade.strategy_name, emotion_str
                    )

                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="behavioral",
                    severity="warning",
                    title=f"Trading Under {emotion_str.title()} State",
                    message=(
                        f"You reported feeling {emotion_str} before this trade. "
                        f"Historical data shows your performance drops significantly "
                        f"when trading in this state.\n" +
                        ("\n".join(guidance) if guidance else "")
                    ),
                    trade_ids=[trade_id],
                    strategy_name=trade.strategy_name,
                ))

        # ── Was this planned? ──
        if trade.journal_entry and trade.journal_entry.was_planned is False:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="warning",
                title="Unplanned Trade",
                message=(
                    "This trade was NOT in your plan. "
                    "Unplanned trades are your #1 source of unnecessary losses. "
                    "What triggered this entry?"
                ),
                trade_ids=[trade_id],
                strategy_name=trade.strategy_name,
                action_suggestion="Only trade setups identified BEFORE the session",
            ))

        return insights

    def _detect_revenge_trading(self, context: Dict) -> List[AgentInsight]:
        """Detect revenge trading patterns across recent trades."""
        insights = []

        recent = self.db.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
        ).order_by(Trade.entry_time.desc()).limit(20).all()

        if len(recent) < 4:
            return insights

        # Look for loss → rapid re-entry → loss → rapid re-entry pattern
        revenge_sequences = 0
        for i in range(len(recent) - 1):
            current = recent[i]
            previous = recent[i + 1]

            if (previous.pnl and previous.pnl < 0 and
                current.entry_time and previous.exit_time and
                (current.entry_time - previous.exit_time).total_seconds() < 900):  # 15 min
                revenge_sequences += 1

        if revenge_sequences >= 2:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="critical",
                title="Revenge Trading Pattern Active",
                message=(
                    f"Detected {revenge_sequences} instances of rapid re-entry "
                    f"after losses in your recent trades. This is a known "
                    f"account-killer pattern. Enforce a mandatory cooldown."
                ),
                action_suggestion="Implement 15-minute mandatory break after any loss",
            ))

        return insights

    def _detect_fomo_pattern(self, context: Dict) -> List[AgentInsight]:
        """Detect FOMO entry patterns."""
        insights = []

        recent = self.db.query(Trade).join(
            JournalEntry, Trade.id == JournalEntry.trade_id, isouter=True
        ).filter(
            Trade.status == TradeStatus.CLOSED,
        ).order_by(Trade.entry_time.desc()).limit(20).all()

        fomo_count = 0
        fomo_results = []

        for trade in recent:
            if trade.journal_entry:
                if trade.journal_entry.pre_trade_emotion == EmotionalState.FOMO:
                    fomo_count += 1
                    if trade.pnl is not None:
                        fomo_results.append(trade.pnl)

            # Also detect FOMO heuristically
            if trade.tags and "fomo_entry" in (trade.tags.get("overrides") or []):
                fomo_count += 1

        if fomo_count >= 3 and fomo_results:
            avg_pnl = sum(fomo_results) / len(fomo_results)
            win_rate = sum(1 for p in fomo_results if p > 0) / len(fomo_results) * 100

            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="warning",
                title="FOMO Trading Pattern",
                message=(
                    f"You've had {fomo_count} FOMO entries in recent trades. "
                    f"FOMO trades: {win_rate:.0f}% win rate, "
                    f"avg PnL: ${avg_pnl:.2f}. "
                    f"Remember: If you have to chase, it's not your entry."
                ),
                action_suggestion="If you feel FOMO, wait for a pullback or skip entirely",
            ))

        return insights

    def _detect_overtrading(self, context: Dict) -> List[AgentInsight]:
        """Detect overtrading patterns."""
        insights = []

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = self.db.query(Trade).filter(
            Trade.entry_time >= today
        ).count()

        if today_trades >= settings.MAX_DAILY_TRADES:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="critical",
                title="Daily Trade Limit Reached",
                message=(
                    f"You've taken {today_trades} trades today "
                    f"(limit: {settings.MAX_DAILY_TRADES}). "
                    f"Overtrading erodes your edge. Stop trading for today."
                ),
                action_suggestion="Close the platform and review your trades",
            ))
        elif today_trades >= settings.MAX_DAILY_TRADES - 2:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="info",
                title="Approaching Trade Limit",
                message=(
                    f"{today_trades} trades today. "
                    f"Be very selective with remaining trades."
                ),
            ))

        return insights

    def _detect_emotional_patterns(self, context: Dict) -> List[AgentInsight]:
        """Detect patterns in emotional states vs outcomes."""
        insights = []

        recent_journals = self.db.query(JournalEntry).filter(
            JournalEntry.trade_id.isnot(None)
        ).order_by(JournalEntry.created_at.desc()).limit(30).all()

        emotion_outcomes = {}
        for journal in recent_journals:
            if journal.pre_trade_emotion and journal.trade:
                emo = journal.pre_trade_emotion.value
                if emo not in emotion_outcomes:
                    emotion_outcomes[emo] = {"trades": 0, "winners": 0, "total_pnl": 0}

                emotion_outcomes[emo]["trades"] += 1
                pnl = journal.trade.pnl
                if pnl is not None:
                    emotion_outcomes[emo]["total_pnl"] += pnl
                    if pnl > 0:
                        emotion_outcomes[emo]["winners"] += 1

        # Find worst-performing emotional state
        worst_emo = None
        worst_avg = 0

        for emo, data in emotion_outcomes.items():
            if data["trades"] >= 3:
                avg = data["total_pnl"] / data["trades"]
                if worst_emo is None or avg < worst_avg:
                    worst_emo = emo
                    worst_avg = avg

        if worst_emo and worst_avg < -20:
            data = emotion_outcomes[worst_emo]
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="behavioral",
                severity="warning",
                title=f"Worst Emotional State: {worst_emo.title()}",
                message=(
                    f"When you trade feeling {worst_emo}, you average "
                    f"${worst_avg:.2f} per trade ({data['trades']} trades). "
                    f"Win rate: {data['winners']/data['trades']*100:.0f}%. "
                    f"This emotion is costing you money."
                ),
                action_suggestion=f"Do NOT trade when feeling {worst_emo}",
            ))

        return insights

    def _check_strategy_specific_traps(self, context: Dict) -> List[AgentInsight]:
        """
        Check for strategy-specific traps defined in DNA.
        This is where DNA knowledge becomes powerful.
        """
        insights = []

        trade_id = context.get("trade_id")
        if not trade_id:
            return insights

        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade or not trade.strategy_name:
            return insights

        dna = self.strategy_query.get_strategy(trade.strategy_name)
        if not dna:
            return insights

        # Check each emotional trap defined in DNA
        for trap in dna.emotional_traps:
            if trap.frequency == "happens_often":
                # For frequent traps, always remind
                insights.append(AgentInsight(
                    agent_name=self.name,
                    insight_type="coaching",
                    severity="info",
                    title=f"Common Trap: {trap.trap}",
                    message=(
                        f"Reminder for {dna.display_name}: {trap.description}\n"
                        f"Solution: {trap.solution}"
                    ),
                    trade_ids=[trade_id],
                    strategy_name=trade.strategy_name,
                ))

        # Check known weaknesses
        for weakness in dna.personal_insights.weaknesses:
            insights.append(AgentInsight(
                agent_name=self.name,
                insight_type="coaching",
                severity="info",
                title="Known Weakness Reminder",
                message=f"Watch out: {weakness}",
                trade_ids=[trade_id],
                strategy_name=trade.strategy_name,
            ))

        return insights