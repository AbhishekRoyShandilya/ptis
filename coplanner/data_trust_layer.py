# coplanner/data_trust_layer.py
"""
THE DATA TRUST LAYER

Solves the CRITICAL problem:
"User's raw trade data is noisy because of emotional trading"

Provides CLEAN data to Strategy-Regime Matrix by:
1. Prioritizing AVT autonomous trade data (cleanest)
2. Filtering user trades by execution quality
3. Blending with confidence-based weights

THREE DATA TIERS:
  Tier 1: AVT Clean Data      → "What the strategy CAN do"
  Tier 2: User Quality Trades  → "What the user does WHEN disciplined"
  Tier 3: All User Trades      → "What the user actually does" (behavioral only)

Architecture:
  AVT Bridge → Tier 1 (cleanest)
  Trade Quality Classifier → Tier 2 (filtered) / Tier 3 (all)
  DataTrustLayer blends tiers → StrategyRegimeMatrix
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from coplanner.models import ConfidenceLevel, RegimePerformanceStats, TradeRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS & DATA MODELS
# ─────────────────────────────────────────────

class DataTier(Enum):
    """Which data source a trade came from."""
    AVT_AUTONOMOUS = "avt_autonomous"       # Fully AVT-executed
    AVT_GUIDED = "avt_guided"               # AVT signal, user executed
    USER_CLEAN = "user_clean"               # User trade, clean execution
    USER_PARTIAL = "user_partial"           # User trade, minor deviations
    USER_EMOTIONAL = "user_emotional"       # User trade, emotional execution
    USER_UNKNOWN = "user_unknown"           # Not classified yet


class TradeQualityTag(Enum):
    """Quality classification of a trade's execution."""
    PRISTINE = "pristine"           # Perfect execution
    CLEAN = "clean"                 # Minor imperfections but followed plan
    ACCEPTABLE = "acceptable"       # Some deviation but reasonable
    DEVIATED = "deviated"           # Significant deviation from plan
    EMOTIONAL = "emotional"         # Clearly emotional decision
    RANDOM = "random"               # No real setup, gamble


@dataclass
class TieredTradeRecord:
    """Trade record enriched with quality and trust data."""
    trade: TradeRecord
    tier: DataTier = DataTier.USER_UNKNOWN
    quality: TradeQualityTag = TradeQualityTag.CLEAN
    trust_weight: float = 0.5
    quality_reasons: List[str] = field(default_factory=list)


@dataclass
class TierWeights:
    """
    Configurable weights for each data tier.
    These control how much each tier influences the matrix.
    """
    avt_autonomous: float = 1.0
    avt_guided: float = 0.85
    user_clean: float = 0.70
    user_partial: float = 0.40
    user_emotional: float = 0.05
    user_unknown: float = 0.30

    def get_weight(self, tier: DataTier) -> float:
        return getattr(self, tier.value, 0.3)


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class DataTrustLayer:
    """
    Core trust layer that provides CLEAN data to Strategy-Regime Matrix.

    Instead of: matrix.build(all_raw_trades)  <-- DIRTY
    Now:        matrix.build(trust_layer.get_trusted_data())  <-- CLEAN

    Usage:
        trust = DataTrustLayer(db, avt_bridge, quality_classifier)
        trusted_data = trust.get_trusted_trades()
        stats = trust.get_blended_performance('breakout', 'trending_up')
    """

    def __init__(
        self,
        db,
        avt_data_source=None,
        quality_classifier=None,
        tier_weights: Optional[TierWeights] = None,
        config: Optional[dict] = None,
    ):
        self.db = db
        self.avt_source = avt_data_source
        self.classifier = quality_classifier
        self.weights = tier_weights or TierWeights()
        self.config = config or {}

        # Cache
        self._tiered_cache: Optional[List[TieredTradeRecord]] = None
        self._cache_time: Optional[datetime] = None

        self.MIN_AVT_TRADES = self.config.get("min_avt_trades", 10)
        self.QUALITY_SHIFT_THRESHOLD = self.config.get("quality_shift_threshold", 0.70)

    # ─────────────────────────────────────────────
    # MAIN API — Get Trusted Data
    # ─────────────────────────────────────────────

    def get_trusted_trades(self, force_refresh: bool = False) -> List[TieredTradeRecord]:
        """
        Returns ALL trades enriched with tier and quality data.
        Filtered and weighted for strategy evaluation.
        """
        if self._tiered_cache and not force_refresh:
            return self._tiered_cache

        all_tiered: List[TieredTradeRecord] = []

        # Tier 1: AVT Data
        avt_trades = self._get_avt_trades()
        for trade in avt_trades:
            all_tiered.append(TieredTradeRecord(
                trade=trade,
                tier=DataTier.AVT_AUTONOMOUS,
                quality=TradeQualityTag.PRISTINE,
                trust_weight=self.weights.avt_autonomous,
                quality_reasons=["AVT autonomous execution"],
            ))

        # Tier 2 & 3: User Data (classified)
        user_trades = self.db.get_all_completed_trades()
        for trade in user_trades:
            if self._is_avt_trade(trade):
                continue

            tier, quality, reasons = self._classify_trade(trade)
            weight = self.weights.get_weight(tier)

            all_tiered.append(TieredTradeRecord(
                trade=trade,
                tier=tier,
                quality=quality,
                trust_weight=weight,
                quality_reasons=reasons,
            ))

        self._tiered_cache = all_tiered
        self._cache_time = datetime.now(timezone.utc)

        logger.info(
            "Data trust layer: %d trades classified. AVT: %d, Clean: %d, Emotional: %d",
            len(all_tiered),
            sum(1 for t in all_tiered if t.tier == DataTier.AVT_AUTONOMOUS),
            sum(1 for t in all_tiered if t.quality in (TradeQualityTag.PRISTINE, TradeQualityTag.CLEAN)),
            sum(1 for t in all_tiered if t.quality == TradeQualityTag.EMOTIONAL),
        )
        return all_tiered

    def get_strategy_trusted_trades(
        self, strategy_name: str, regime: str,
    ) -> List[TieredTradeRecord]:
        """Get trusted trades for specific strategy x regime. Excludes emotional/random."""
        all_tiered = self.get_trusted_trades()
        return [
            t for t in all_tiered
            if (t.trade.strategy_name.lower().strip() == strategy_name.lower().strip()
                and t.trade.regime_at_entry.lower().strip() == regime.lower().strip()
                and t.quality not in (TradeQualityTag.EMOTIONAL, TradeQualityTag.RANDOM))
        ]

    def get_behavioral_trades(self) -> List[TieredTradeRecord]:
        """
        Get ALL trades including emotional ones.
        Behavioral patterns REQUIRE noisy data because
        the noise IS the pattern we're looking for.
        """
        return self.get_trusted_trades()

    # ─────────────────────────────────────────────
    # BLENDED PERFORMANCE — Weighted stats
    # ─────────────────────────────────────────────

    def get_blended_performance(
        self, strategy_name: str, regime: str,
    ) -> RegimePerformanceStats:
        """
        Compute WEIGHTED performance stats for strategy x regime.
        Higher-trust trades count more. Emotional trades excluded.
        Replaces simple win_rate = wins/total.
        """
        trades = self.get_strategy_trusted_trades(strategy_name, regime)

        if not trades:
            return RegimePerformanceStats(
                strategy_name=strategy_name,
                regime=regime,
                win_rate=0.5,
                sample_confidence=ConfidenceLevel.INSUFFICIENT,
            )

        total_weight = 0.0
        weighted_wins = 0.0
        weighted_rr = 0.0
        raw_wins = 0
        raw_total = 0
        max_win_rr = 0.0
        max_loss_rr = 0.0
        total_holding = 0.0

        for tiered in trades:
            trade = tiered.trade
            weight = tiered.trust_weight

            total_weight += weight
            raw_total += 1

            if trade.outcome == "win":
                weighted_wins += weight
                raw_wins += 1
                max_win_rr = max(max_win_rr, trade.actual_rr)
            else:
                max_loss_rr = min(max_loss_rr, trade.actual_rr)

            weighted_rr += trade.actual_rr * weight

            if trade.entry_time and trade.exit_time:
                holding = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                total_holding += holding

        weighted_wr = weighted_wins / total_weight if total_weight > 0 else 0.5
        weighted_avg_rr = weighted_rr / total_weight if total_weight > 0 else 0
        effective_n = total_weight

        return RegimePerformanceStats(
            strategy_name=strategy_name,
            regime=regime,
            wins=raw_wins,
            losses=raw_total - raw_wins,
            total_trades=raw_total,
            win_rate=round(weighted_wr, 4),
            avg_rr=round(weighted_avg_rr, 4),
            total_rr=weighted_rr,
            max_win_rr=max_win_rr,
            max_loss_rr=max_loss_rr,
            avg_holding_period=total_holding / raw_total if raw_total > 0 else 0,
            sample_confidence=self._effective_confidence(effective_n),
            last_updated=datetime.now(timezone.utc),
        )

    def _effective_confidence(self, effective_n: float) -> ConfidenceLevel:
        if effective_n >= 25:
            return ConfidenceLevel.HIGH
        elif effective_n >= 12:
            return ConfidenceLevel.MODERATE
        elif effective_n >= 4:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INSUFFICIENT

    # ─────────────────────────────────────────────
    # DATA QUALITY REPORT
    # ─────────────────────────────────────────────

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Report on overall data quality. Helps user understand execution quality."""
        all_tiered = self.get_trusted_trades()

        if not all_tiered:
            return {"total_trades": 0, "message": "No trade data available."}

        tier_counts: Dict[str, int] = {}
        quality_counts: Dict[str, int] = {}
        for t in all_tiered:
            tier_counts[t.tier.value] = tier_counts.get(t.tier.value, 0) + 1
            quality_counts[t.quality.value] = quality_counts.get(t.quality.value, 0) + 1

        total = len(all_tiered)
        clean_count = sum(
            1 for t in all_tiered
            if t.quality in (TradeQualityTag.PRISTINE, TradeQualityTag.CLEAN)
        )
        emotional_count = sum(
            1 for t in all_tiered
            if t.quality in (TradeQualityTag.EMOTIONAL, TradeQualityTag.RANDOM)
        )

        clean_ratio = clean_count / total
        emotional_ratio = emotional_count / total

        if clean_ratio >= 0.80:
            grade, message = "A", "Excellent execution discipline. Data is highly trustworthy."
        elif clean_ratio >= 0.65:
            grade, message = "B", "Good execution. Most data is usable."
        elif clean_ratio >= 0.50:
            grade, message = "C", "Mixed execution. AVT data being prioritized to compensate."
        elif clean_ratio >= 0.35:
            grade, message = "D", "Poor execution discipline. Heavily relying on AVT data."
        else:
            grade, message = "F", "Most trades are emotional. System using AVT data almost exclusively."

        return {
            "total_trades": total,
            "tier_breakdown": tier_counts,
            "quality_breakdown": quality_counts,
            "clean_ratio": round(clean_ratio, 2),
            "emotional_ratio": round(emotional_ratio, 2),
            "execution_grade": grade,
            "message": message,
            "effective_sample_size": round(sum(t.trust_weight for t in all_tiered), 1),
            "recommendation": self._quality_recommendation(
                clean_ratio, emotional_ratio, tier_counts
            ),
        }

    def _quality_recommendation(
        self, clean_ratio: float, emotional_ratio: float, tier_counts: Dict[str, int],
    ) -> str:
        avt_count = tier_counts.get("avt_autonomous", 0)
        if avt_count < self.MIN_AVT_TRADES:
            return (
                f"Only {avt_count} AVT trades. Need at least {self.MIN_AVT_TRADES} "
                f"for reliable strategy evaluation. Run AVT more frequently."
            )
        if emotional_ratio > 0.40:
            return (
                "Over 40% of trades are emotional. System is using AVT data "
                "to protect you from noisy self-data. Focus on execution discipline."
            )
        if clean_ratio >= 0.70:
            return (
                "Good data quality. System is blending your clean trades "
                "with AVT data for optimal accuracy."
            )
        return (
            "Mixed data quality. Increasing your clean execution ratio "
            "will make the system more personalized to your style."
        )

    # ─────────────────────────────────────────────
    # ADAPTIVE WEIGHTING
    # ─────────────────────────────────────────────

    def compute_adaptive_weights(self) -> TierWeights:
        """
        Dynamically adjust tier weights based on user's execution quality.
        If user executes well -> increase user data weight.
        If user executes poorly -> increase AVT weight.
        """
        report = self.get_data_quality_report()
        clean_ratio = report.get("clean_ratio", 0)

        if clean_ratio >= 0.80:
            return TierWeights(
                avt_autonomous=0.85, avt_guided=0.80,
                user_clean=0.80, user_partial=0.50,
                user_emotional=0.05, user_unknown=0.35,
            )
        elif clean_ratio >= 0.60:
            return TierWeights()  # defaults
        elif clean_ratio >= 0.40:
            return TierWeights(
                avt_autonomous=1.0, avt_guided=0.90,
                user_clean=0.55, user_partial=0.25,
                user_emotional=0.02, user_unknown=0.20,
            )
        else:
            return TierWeights(
                avt_autonomous=1.0, avt_guided=0.95,
                user_clean=0.40, user_partial=0.15,
                user_emotional=0.01, user_unknown=0.10,
            )

    def refresh_weights(self) -> None:
        """Recompute and apply adaptive weights."""
        new_weights = self.compute_adaptive_weights()
        self.weights = new_weights
        self._tiered_cache = None
        logger.info(
            "Adaptive weights updated: AVT %.2f, User_clean %.2f, User_emotional %.2f",
            new_weights.avt_autonomous, new_weights.user_clean, new_weights.user_emotional,
        )

    # ─────────────────────────────────────────────
    # AVT DATA ACCESS
    # ─────────────────────────────────────────────

    def _get_avt_trades(self) -> List[TradeRecord]:
        """Get trades from AVT autonomous system."""
        if not self.avt_source:
            return []
        try:
            if hasattr(self.avt_source, "get_completed_trades"):
                return self.avt_source.get_completed_trades()
            if hasattr(self.avt_source, "get_trade_history"):
                raw = self.avt_source.get_trade_history()
                return [self._convert_avt_trade(t) for t in raw]
            if hasattr(self.db, "get_avt_trades"):
                return self.db.get_avt_trades()
            return []
        except Exception as e:
            logger.warning("Failed to get AVT trades: %s", e)
            return []

    def _is_avt_trade(self, trade: TradeRecord) -> bool:
        """Check if a trade is from AVT (to avoid double counting)."""
        if not self.avt_source:
            return False
        if trade.trade_id.startswith("avt_"):
            return True
        return False

    def _convert_avt_trade(self, raw: Any) -> TradeRecord:
        """Convert raw AVT trade data to TradeRecord."""
        if isinstance(raw, TradeRecord):
            return raw
        if isinstance(raw, dict):
            return TradeRecord(
                trade_id=str(raw.get("id", raw.get("trade_id", ""))),
                symbol=raw.get("symbol", ""),
                strategy_name=raw.get("strategy", raw.get("strategy_name", "")),
                timeframe=raw.get("timeframe", ""),
                regime_at_entry=raw.get("regime", raw.get("regime_at_entry", "unknown")),
                entry_price=float(raw.get("entry", raw.get("entry_price", 0))),
                exit_price=float(raw.get("exit", raw.get("exit_price", 0))),
                stop_loss=float(raw.get("sl", raw.get("stop_loss", 0))),
                target_price=float(raw.get("target", raw.get("target_price", 0))),
                entry_time=raw.get("entry_time", datetime.now(timezone.utc)),
                exit_time=raw.get("exit_time", datetime.now(timezone.utc)),
                outcome=raw.get("outcome", "unknown"),
                actual_rr=float(raw.get("rr", raw.get("actual_rr", 0))),
            )
        raise ValueError(f"Cannot convert AVT trade: {type(raw)}")

    # ─────────────────────────────────────────────
    # TRADE QUALITY CLASSIFICATION
    # ─────────────────────────────────────────────

    def _classify_trade(
        self, trade: TradeRecord,
    ) -> Tuple[DataTier, TradeQualityTag, List[str]]:
        """
        Classify a user trade's execution quality.
        Uses TradeQualityClassifier if available, else heuristic rules.
        """
        if self.classifier:
            return self.classifier.classify(trade)
        return self._heuristic_classify(trade)

    def _heuristic_classify(
        self, trade: TradeRecord,
    ) -> Tuple[DataTier, TradeQualityTag, List[str]]:
        """
        Heuristic-based quality classification.
        6 checks: SL respected, early exit, short hold, bad RR, emotional notes, missing plan.
        """
        reasons: List[str] = []
        demerits = 0

        # Check 1: Stop loss respected
        if trade.outcome == "loss":
            expected_loss = abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0
            actual_loss = abs(trade.entry_price - trade.exit_price)
            if expected_loss > 0 and actual_loss > expected_loss * 1.3:
                reasons.append("Stop loss widened -- loss exceeds planned SL by >30%")
                demerits += 2

        # Check 2: Early exit on winners
        if trade.outcome == "win" and trade.target_price:
            potential_reward = abs(trade.target_price - trade.entry_price)
            actual_reward = abs(trade.exit_price - trade.entry_price)
            if potential_reward > 0 and actual_reward < potential_reward * 0.3:
                reasons.append("Exited far too early -- captured <30% of target")
                demerits += 1

        # Check 3: Very short holding time
        if trade.entry_time and trade.exit_time:
            holding_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
            if holding_minutes < 2:
                reasons.append("Extremely short hold (<2 min) -- likely impulse")
                demerits += 2

        # Check 4: RR at entry was terrible
        if trade.entry_price and trade.stop_loss and trade.target_price:
            risk = abs(trade.entry_price - trade.stop_loss)
            reward = abs(trade.target_price - trade.entry_price)
            if risk > 0:
                planned_rr = reward / risk
                if planned_rr < 0.8:
                    reasons.append(f"Entered with poor planned RR ({planned_rr:.1f})")
                    demerits += 1

        # Check 5: Notes contain emotional keywords
        emotional_keywords = [
            "fomo", "revenge", "angry", "frustrated", "impulsive",
            "panic", "scared", "hope", "gut feel", "just felt",
            "no setup", "random", "bored", "overtraded",
        ]
        notes_lower = getattr(trade, "notes", "") or ""
        notes_lower = notes_lower.lower()
        for keyword in emotional_keywords:
            if keyword in notes_lower:
                reasons.append(f"Notes indicate emotional trading: '{keyword}'")
                demerits += 2
                break

        # Check 6: Missing critical data
        if not trade.stop_loss or trade.stop_loss == 0:
            reasons.append("No stop loss set -- undisciplined entry")
            demerits += 2
        if not trade.target_price or trade.target_price == 0:
            reasons.append("No target set -- no exit plan")
            demerits += 1

        if demerits == 0:
            if not reasons:
                reasons.append("Clean execution -- all checks passed")
            return DataTier.USER_CLEAN, TradeQualityTag.CLEAN, reasons
        elif demerits <= 1:
            return DataTier.USER_PARTIAL, TradeQualityTag.ACCEPTABLE, reasons
        elif demerits <= 2:
            return DataTier.USER_PARTIAL, TradeQualityTag.DEVIATED, reasons
        else:
            return DataTier.USER_EMOTIONAL, TradeQualityTag.EMOTIONAL, reasons

    # ─────────────────────────────────────────────
    # COLD START HANDLING
    # ─────────────────────────────────────────────

    def get_data_availability(self) -> Dict[str, Any]:
        """Check what data is available. Handles cold start gracefully."""
        avt_trades = self._get_avt_trades()
        user_trades = self.db.get_all_completed_trades()
        avt_count = len(avt_trades)
        user_count = len(user_trades)

        if avt_count == 0 and user_count == 0:
            mode = "COLD_START"
            message = (
                "No trade data available. System will use neutral assumptions. "
                "Start trading or run AVT to build data."
            )
        elif avt_count < self.MIN_AVT_TRADES and user_count < 20:
            mode = "WARMING_UP"
            message = (
                f"Limited data: {avt_count} AVT trades, {user_count} user trades. "
                f"Need at least {self.MIN_AVT_TRADES} AVT trades for reliable evaluation."
            )
        elif avt_count >= self.MIN_AVT_TRADES:
            mode = "AVT_READY"
            message = f"AVT data sufficient ({avt_count} trades). Strategy evaluation is reliable."
        else:
            mode = "USER_ONLY"
            message = (
                f"No AVT data. Using user trades only ({user_count}). "
                f"Quality filtering applied but results may be noisy."
            )

        return {
            "mode": mode,
            "avt_trades": avt_count,
            "user_trades": user_count,
            "total_usable": avt_count + user_count,
            "message": message,
            "avt_sufficient": avt_count >= self.MIN_AVT_TRADES,
        }

    def invalidate_cache(self) -> None:
        """Clear cached data -- call after new trades."""
        self._tiered_cache = None
        self._cache_time = None
