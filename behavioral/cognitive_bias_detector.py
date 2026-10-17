# behavioral/cognitive_bias_detector.py
"""
Detects cognitive biases from trade patterns and behavioral data.
Identifies anchoring, recency bias, disposition effect, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from core.enums import BehavioralEventType, EmotionalState, TradeDirection, TradeStatus
from core.models import BehavioralEvent, OverrideEvent, Trade

logger = logging.getLogger(__name__)


@dataclass
class BiasDetection:
    """A detected cognitive bias instance."""
    bias_type: str
    confidence: float           # 0.0 - 1.0
    severity: str              # low, medium, high
    description: str
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""
    trade_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias_type": self.bias_type,
            "confidence": round(self.confidence, 2),
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "related_trades": len(self.trade_ids),
        }


class CognitiveBiasDetector:
    """
    Analyzes trade history to detect cognitive biases.
    Requires minimum trade count for statistical significance.
    """

    MIN_TRADES = 20

    def __init__(self, lookback_days: int = 60):
        self._lookback_days = lookback_days

    def detect_all(self, db: DBSession) -> list[BiasDetection]:
        """Run all bias detectors and return findings."""
        cutoff = datetime.utcnow() - timedelta(days=self._lookback_days)

        trades = (
            db.query(Trade)
            .filter(
                Trade.entry_time >= cutoff,
                Trade.status == TradeStatus.CLOSED,
            )
            .order_by(Trade.entry_time.asc())
            .all()
        )

        if len(trades) < self.MIN_TRADES:
            logger.info(
                "Only %d trades — need %d for bias detection",
                len(trades), self.MIN_TRADES,
            )
            return []

        detections = []
        detections.extend(self._detect_disposition_effect(trades))
        detections.extend(self._detect_recency_bias(trades))
        detections.extend(self._detect_anchoring(trades))
        detections.extend(self._detect_loss_aversion(trades))
        detections.extend(self._detect_overconfidence(trades, db))
        detections.extend(self._detect_gambler_fallacy(trades))
        detections.extend(self._detect_confirmation_bias(trades, db))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info("Bias detection complete: %d biases found", len(detections))
        return detections

    def _detect_disposition_effect(self, trades: list[Trade]) -> list[BiasDetection]:
        """
        Disposition effect: holding losers too long, cutting winners too short.
        Evidence: avg winner duration < avg loser duration,
                  avg winner R < avg loser R (absolute).
        """
        winners = [t for t in trades if t.pnl and t.pnl > 0 and t.exit_time and t.entry_time]
        losers = [t for t in trades if t.pnl and t.pnl < 0 and t.exit_time and t.entry_time]

        if len(winners) < 5 or len(losers) < 5:
            return []

        avg_win_duration = sum(
            (t.exit_time - t.entry_time).total_seconds() for t in winners
        ) / len(winners) / 60  # minutes

        avg_loss_duration = sum(
            (t.exit_time - t.entry_time).total_seconds() for t in losers
        ) / len(losers) / 60

        avg_win_r = sum(t.r_multiple for t in winners if t.r_multiple) / max(
            1, sum(1 for t in winners if t.r_multiple)
        )
        avg_loss_r = abs(sum(t.r_multiple for t in losers if t.r_multiple) / max(
            1, sum(1 for t in losers if t.r_multiple)
        ))

        signals = []
        confidence = 0.0

        if avg_loss_duration > avg_win_duration * 1.5:
            signals.append(
                f"Avg loser held {avg_loss_duration:.0f}min vs winner {avg_win_duration:.0f}min"
            )
            confidence += 0.35

        if avg_loss_r > avg_win_r * 1.3:
            signals.append(
                f"Avg loss size {avg_loss_r:.1f}R vs avg win {avg_win_r:.1f}R"
            )
            confidence += 0.35

        if avg_win_r < 1.0 and avg_loss_r > 1.0:
            signals.append(
                "Winners cut before 1R while losers exceed 1R on average"
            )
            confidence += 0.3

        if confidence >= 0.5:
            return [BiasDetection(
                bias_type="disposition_effect",
                confidence=min(1.0, confidence),
                severity="high" if confidence > 0.7 else "medium",
                description=(
                    "Disposition Effect: tendency to hold losers longer "
                    "than winners and cut profits too early"
                ),
                evidence=signals,
                recommendation=(
                    "Set target exits BEFORE entering. "
                    "Use shadow trading to compare your exits against strategy targets. "
                    "Review if your 'fear of loss' is creating asymmetric exits."
                ),
                trade_ids=[str(t.id) for t in losers[:5]],
            )]
        return []

    def _detect_recency_bias(self, trades: list[Trade]) -> list[BiasDetection]:
        """
        Recency bias: recent results disproportionately affect behavior.
        Evidence: position sizing changes after wins/losses,
                  strategy switching after short losing streak.
        """
        if len(trades) < 10:
            return []

        signals = []
        confidence = 0.0

        # Check if position size increases after wins, decreases after losses
        size_after_win = []
        size_after_loss = []

        for i in range(1, len(trades)):
            prev_pnl = trades[i - 1].pnl
            curr_size = trades[i].position_size

            if prev_pnl and prev_pnl > 0:
                size_after_win.append(curr_size)
            elif prev_pnl and prev_pnl < 0:
                size_after_loss.append(curr_size)

        if size_after_win and size_after_loss:
            avg_after_win = sum(size_after_win) / len(size_after_win)
            avg_after_loss = sum(size_after_loss) / len(size_after_loss)

            if avg_after_win > avg_after_loss * 1.3:
                signals.append(
                    f"Avg position after win: {avg_after_win:.0f} vs "
                    f"after loss: {avg_after_loss:.0f} ({(avg_after_win/avg_after_loss - 1)*100:.0f}% larger)"
                )
                confidence += 0.4

        # Check strategy switching after losses
        strategy_switches = 0
        for i in range(2, len(trades)):
            if (
                trades[i - 1].pnl and trades[i - 1].pnl < 0
                and trades[i - 2].pnl and trades[i - 2].pnl < 0
                and trades[i].declared_strategy_id != trades[i - 1].declared_strategy_id
                and trades[i].declared_strategy_id is not None
            ):
                strategy_switches += 1

        switch_rate = strategy_switches / max(1, len(trades) - 2)
        if switch_rate > 0.15:
            signals.append(
                f"Strategy switches after consecutive losses: "
                f"{strategy_switches} times ({switch_rate*100:.0f}%)"
            )
            confidence += 0.4

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="recency_bias",
                confidence=min(1.0, confidence),
                severity="medium",
                description=(
                    "Recency Bias: recent results are disproportionately "
                    "influencing sizing and strategy decisions"
                ),
                evidence=signals,
                recommendation=(
                    "Keep position sizing rule-based, not emotion-based. "
                    "Review your strategy's edge over 50+ trades, not the last 3. "
                    "A losing streak doesn't invalidate a positive expectancy strategy."
                ),
            )]
        return []

    def _detect_anchoring(self, trades: list[Trade]) -> list[BiasDetection]:
        """
        Anchoring: fixating on entry price or round numbers for exits.
        Evidence: exits clustering around round numbers or entry price.
        """
        signals = []
        confidence = 0.0

        # Check breakeven exits
        breakeven_exits = sum(
            1 for t in trades
            if t.pnl is not None and abs(t.pnl) < (t.risk_amount or 1) * 0.1
            and t.exit_price is not None
        )
        be_rate = breakeven_exits / len(trades)

        if be_rate > 0.15:
            signals.append(
                f"{be_rate*100:.0f}% of trades exited near breakeven — "
                f"possible anchoring to entry price"
            )
            confidence += 0.4

        # Check exits at round numbers
        if any(t.exit_price for t in trades):
            round_exits = sum(
                1 for t in trades
                if t.exit_price and (t.exit_price % 1.0 < 0.05 or t.exit_price % 1.0 > 0.95)
            )
            total_with_exit = sum(1 for t in trades if t.exit_price)
            if total_with_exit > 0:
                round_rate = round_exits / total_with_exit
                if round_rate > 0.3:
                    signals.append(
                        f"{round_rate*100:.0f}% of exits at round numbers — "
                        f"possible price anchoring"
                    )
                    confidence += 0.3

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="anchoring",
                confidence=min(1.0, confidence),
                severity="low",
                description=(
                    "Anchoring Bias: exit decisions may be influenced by "
                    "entry price or round numbers rather than market structure"
                ),
                evidence=signals,
                recommendation=(
                    "Set exit levels based on market structure (S/R, VWAP, ATR) "
                    "rather than entry price. Use the shadow engine to compare "
                    "structure-based exits vs your actual exits."
                ),
            )]
        return []

    def _detect_loss_aversion(self, trades: list[Trade]) -> list[BiasDetection]:
        """
        Loss aversion: risk-reward asymmetry showing fear of losses
        dominating decision making.
        """
        winners = [t for t in trades if t.r_multiple and t.r_multiple > 0]
        losers = [t for t in trades if t.r_multiple and t.r_multiple < 0]

        if len(winners) < 5 or len(losers) < 5:
            return []

        avg_win_r = sum(t.r_multiple for t in winners) / len(winners)
        avg_loss_r = abs(sum(t.r_multiple for t in losers) / len(losers))
        win_rate = len(winners) / (len(winners) + len(losers))

        signals = []
        confidence = 0.0

        # High win rate but tiny winners (fear of giving back)
        if win_rate > 0.6 and avg_win_r < 0.8:
            signals.append(
                f"High win rate ({win_rate*100:.0f}%) but small avg winner "
                f"({avg_win_r:.1f}R) suggests cutting profits from fear"
            )
            confidence += 0.4

        # Negative expectancy despite high win rate
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
        if win_rate > 0.5 and expectancy < 0:
            signals.append(
                f"Negative expectancy ({expectancy:.2f}R) despite "
                f"{win_rate*100:.0f}% win rate — winners too small"
            )
            confidence += 0.4

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="loss_aversion",
                confidence=min(1.0, confidence),
                severity="high" if expectancy < 0 else "medium",
                description=(
                    "Loss Aversion: fear of losing is causing premature "
                    "profit-taking, limiting upside"
                ),
                evidence=signals,
                recommendation=(
                    "Focus on R-multiple distribution, not win rate. "
                    "Practice letting winners run to at least 1.5R. "
                    "Compare your exits vs the trailing stop shadow variant."
                ),
            )]
        return []

    def _detect_overconfidence(
        self, trades: list[Trade], db: DBSession
    ) -> list[BiasDetection]:
        """
        Overconfidence: increased sizing or reduced caution after winning streak.
        """
        signals = []
        confidence = 0.0

        # Find winning streaks and check behavior after
        streak = 0
        post_streak_trades = []

        for i, trade in enumerate(trades):
            if trade.pnl and trade.pnl > 0:
                streak += 1
            else:
                if streak >= 3 and i < len(trades) - 1:
                    post_streak_trades.append(trades[i])
                streak = 0

        if post_streak_trades and len(post_streak_trades) >= 3:
            # Check if post-streak trades have higher risk
            avg_normal_risk = sum(
                t.risk_pct for t in trades if t.risk_pct
            ) / max(1, sum(1 for t in trades if t.risk_pct))

            avg_post_streak_risk = sum(
                t.risk_pct for t in post_streak_trades if t.risk_pct
            ) / max(1, sum(1 for t in post_streak_trades if t.risk_pct))

            if avg_post_streak_risk > avg_normal_risk * 1.3:
                signals.append(
                    f"Risk increases {((avg_post_streak_risk/avg_normal_risk - 1)*100):.0f}% "
                    f"after winning streaks"
                )
                confidence += 0.5

        # Check conviction scores during streaks
        high_conviction_losses = sum(
            1 for t in trades
            if t.conviction_score and t.conviction_score >= 8
            and t.pnl and t.pnl < 0
        )
        high_conviction_total = sum(
            1 for t in trades
            if t.conviction_score and t.conviction_score >= 8
        )

        if high_conviction_total >= 5:
            hc_loss_rate = high_conviction_losses / high_conviction_total
            if hc_loss_rate > 0.5:
                signals.append(
                    f"High conviction trades (8+) lose {hc_loss_rate*100:.0f}% of the time"
                )
                confidence += 0.4

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="overconfidence",
                confidence=min(1.0, confidence),
                severity="medium",
                description=(
                    "Overconfidence Bias: winning streaks or high conviction "
                    "leading to increased risk-taking"
                ),
                evidence=signals,
                recommendation=(
                    "Keep position sizing mechanical regardless of recent results. "
                    "Your conviction score doesn't correlate with outcomes — "
                    "trust the setup, not the feeling."
                ),
            )]
        return []

    def _detect_gambler_fallacy(self, trades: list[Trade]) -> list[BiasDetection]:
        """
        Gambler's fallacy: believing a losing streak makes a win 'due'.
        Evidence: increasing size during losing streaks.
        """
        signals = []
        confidence = 0.0

        # Track sizing during consecutive losses
        streak_sizes = []
        current_streak = 0

        for trade in trades:
            if trade.pnl and trade.pnl < 0:
                current_streak += 1
                if current_streak >= 2 and trade.risk_pct:
                    streak_sizes.append((current_streak, trade.risk_pct))
            else:
                current_streak = 0

        if len(streak_sizes) >= 5:
            # Correlation between streak length and size
            early_streak = [s[1] for s in streak_sizes if s[0] <= 3]
            late_streak = [s[1] for s in streak_sizes if s[0] > 3]

            if early_streak and late_streak:
                avg_early = sum(early_streak) / len(early_streak)
                avg_late = sum(late_streak) / len(late_streak)

                if avg_late > avg_early * 1.2:
                    signals.append(
                        f"Risk increases during extended losing streaks: "
                        f"early={avg_early:.2f}% vs late={avg_late:.2f}%"
                    )
                    confidence += 0.5

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="gamblers_fallacy",
                confidence=min(1.0, confidence),
                severity="high",
                description=(
                    "Gambler's Fallacy: increasing risk during losing streaks, "
                    "believing a win is 'due'"
                ),
                evidence=signals,
                recommendation=(
                    "Each trade is independent. Reduce size during losing streaks, "
                    "don't increase it. Consider a hard rule: "
                    "after 3 consecutive losses, reduce size by 50%."
                ),
            )]
        return []

    def _detect_confirmation_bias(
        self, trades: list[Trade], db: DBSession
    ) -> list[BiasDetection]:
        """
        Confirmation bias: ignoring invalidation signals,
        only seeing confirming evidence.
        """
        from core.enums import OverrideType

        trade_ids = [t.id for t in trades]

        invalidation_ignored = (
            db.query(func.count(OverrideEvent.id))
            .filter(
                OverrideEvent.trade_id.in_(trade_ids),
                OverrideEvent.override_type == OverrideType.INVALIDATION_IGNORED,
            )
            .scalar()
        )

        missing_conf = (
            db.query(func.count(OverrideEvent.id))
            .filter(
                OverrideEvent.trade_id.in_(trade_ids),
                OverrideEvent.override_type == OverrideType.MISSING_CONFIRMATION,
            )
            .scalar()
        )

        signals = []
        confidence = 0.0

        inv_rate = invalidation_ignored / len(trades)
        miss_rate = missing_conf / len(trades)

        if inv_rate > 0.1:
            signals.append(
                f"Invalidation signals ignored on {inv_rate*100:.0f}% of trades"
            )
            confidence += 0.4

        if miss_rate > 0.2:
            signals.append(
                f"Missing required confirmations on {miss_rate*100:.0f}% of trades"
            )
            confidence += 0.3

        if confidence >= 0.4:
            return [BiasDetection(
                bias_type="confirmation_bias",
                confidence=min(1.0, confidence),
                severity="high" if inv_rate > 0.2 else "medium",
                description=(
                    "Confirmation Bias: tendency to ignore disconfirming evidence "
                    "and enter trades without full confirmation"
                ),
                evidence=signals,
                recommendation=(
                    "Before every entry, explicitly state what would invalidate "
                    "the setup. Use the strategy checklist to ensure all required "
                    "confirmations are actually met before entering."
                ),
            )]
        return []