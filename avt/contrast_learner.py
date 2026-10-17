"""
============================================================
CONTRAST LEARNING ENGINE (CLE)
avt/contrast_learner.py

Learns WHY similar setups had different outcomes
by comparing WIN vs LOSS feature distributions.

Produces SOFT insights, NOT hard rules.

┌─────────────────────────────────────────────────────────────┐
│  ARCHITECTURAL RULE: AVT EDGE DOMAIN ONLY                   │
│                                                             │
│  This module lives in AVT because it uses trade OUTCOMES    │
│  (wins/losses) to discover what drives performance.         │
│                                                             │
│  It NEVER modifies StrategyProfile (user intent).           │
│  Its outputs feed:                                          │
│   → self_improver.py (hypothesis candidates)                │
│   → competition.py  (blind spot analysis for user)          │
│   → edge_monitor.py (stability detection)                   │
│                                                             │
│  Weight adjustments go through self_improver's              │
│  prove-it cycle — NEVER applied directly.                   │
└─────────────────────────────────────────────────────────────┘
============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger("PTIS.ContrastLearner")


# ═══════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════

@dataclass
class OutcomeExample:
    """Single trade with features and outcome"""
    features: Dict[str, float]
    outcome: float                          # R-multiple (+1.5, -0.8, etc.)
    label: str = ""                         # "win" | "loss" — auto-set from outcome
    regime: Optional[str] = None            # "bearish", "bullish", "sideways"
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            self.label = "win" if self.outcome > 0 else "loss"


@dataclass
class FeatureContrast:
    """How a single feature differs between wins and losses"""
    feature_name: str
    win_mean: float
    win_std: float
    loss_mean: float
    loss_std: float
    difference: float                   # win_mean - loss_mean
    effect_size: float                  # Cohen's d
    importance: float                   # 0-1 normalized importance score
    direction: str                      # "higher_better" | "lower_better" | "neutral"
    p_value: float = 1.0               # Statistical significance
    n_win: int = 0
    n_loss: int = 0
    reliable: bool = False             # Enough data to trust this?

    def to_human(self) -> str:
        """One-line human-readable description"""
        if self.direction == "higher_better":
            return (
                f"{self.feature_name}: winners avg {self.win_mean:.3f} "
                f"vs losers {self.loss_mean:.3f} (higher = better, "
                f"effect: {self.effect_size:.2f})"
            )
        elif self.direction == "lower_better":
            return (
                f"{self.feature_name}: winners avg {self.win_mean:.3f} "
                f"vs losers {self.loss_mean:.3f} (lower = better, "
                f"effect: {self.effect_size:.2f})"
            )
        return (
            f"{self.feature_name}: winners {self.win_mean:.3f} "
            f"vs losers {self.loss_mean:.3f} (unclear direction)"
        )


@dataclass
class ContrastInsight:
    """Complete contrast analysis result"""
    summary: str
    top_features: List[FeatureContrast]
    all_features: List[FeatureContrast]
    confidence: float                   # 0-1 how much to trust this
    n_wins: int = 0
    n_losses: int = 0
    n_total: int = 0
    regime: Optional[str] = None        # None = all regimes combined
    data_sufficient: bool = False

    def to_display(self) -> str:
        """Human-readable display"""
        lines = []
        lines.append(f"{'─' * 55}")

        if self.regime:
            lines.append(f"  CONTRAST INSIGHT (regime: {self.regime})")
        else:
            lines.append(f"  CONTRAST INSIGHT (all trades)")

        lines.append(f"{'─' * 55}")
        lines.append(f"  Trades analyzed: {self.n_wins}W / {self.n_losses}L")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        lines.append(f"")
        lines.append(f"  {self.summary}")
        lines.append(f"")

        if self.top_features:
            lines.append(f"  Key differentiators:")
            for i, fc in enumerate(self.top_features[:6], 1):
                reliability = "✅" if fc.reliable else "⚠️"
                lines.append(f"    {i}. {reliability} {fc.to_human()}")
        else:
            lines.append(f"  No significant differentiators found yet.")

        if not self.data_sufficient:
            lines.append(f"")
            lines.append(f"  ⚠️  Need more trades for reliable insights.")

        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


@dataclass
class WeightAdjustment:
    """Suggested weight change for AVT scoring"""
    feature_name: str
    current_weight: float
    suggested_weight: float
    reason: str
    confidence: float               # How confident in this suggestion
    is_suggestion: bool = True      # Always True — never forced


@dataclass
class ImprovementSuggestion:
    """Soft suggestion for strategy improvement"""
    description: str
    evidence: str
    feature_involved: str
    direction: str                  # "increase_attention" | "decrease_attention"
    strength: str                   # "strong" | "moderate" | "weak"
    confidence: float


# ═══════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════

class ContrastLearner:
    """
    Compares WIN vs LOSS trades to discover what subtle
    differences drive outcomes.

    PHILOSOPHY:
    ───────────
    This does NOT produce rules.
    It produces SOFT INSIGHTS that the scoring system
    can use to adjust weights probabilistically.

    Think of it as:
    "Hey, I noticed your winning trades tend to have
     tighter compression and better location.
     Maybe pay more attention to those."

    NOT:
    "Only trade when R² > 0.82"

    USAGE:
    ──────
    learner = ContrastLearner()

    # Feed trades as they complete
    learner.add_example(OutcomeExample(features={...}, outcome=1.5))
    learner.add_example(OutcomeExample(features={...}, outcome=-0.8))

    # When you have enough data
    insight = learner.generate_insight()
    print(insight.to_display())

    # Get AVT weight adjustments
    adjustments = learner.suggest_weight_adjustments(current_weights)
    """

    # Minimum samples for any analysis
    MIN_TOTAL = 20
    MIN_PER_GROUP = 8

    # Effect size thresholds
    STRONG_EFFECT = 0.8
    MODERATE_EFFECT = 0.5
    WEAK_EFFECT = 0.3

    def __init__(self, strategy_id: str = "default"):
        self.strategy_id = strategy_id
        self.examples: List[OutcomeExample] = []
        self._last_insight: Optional[ContrastInsight] = None
        self._insight_stale = True

    # ───────────────────────────────────────────────────
    # DATA COLLECTION
    # ───────────────────────────────────────────────────

    def add_example(self, example: OutcomeExample):
        """Add a completed trade with features and outcome"""
        self.examples.append(example)
        self._insight_stale = True

        logger.debug(
            f"Added {example.label} example "
            f"(outcome: {example.outcome:+.2f}R). "
            f"Total: {len(self.examples)}"
        )

    def add_examples_batch(self, examples: List[OutcomeExample]):
        """Add multiple examples at once"""
        self.examples.extend(examples)
        self._insight_stale = True
        logger.info(f"Added {len(examples)} examples. Total: {len(self.examples)}")

    @property
    def n_wins(self) -> int:
        return sum(1 for e in self.examples if e.label == "win")

    @property
    def n_losses(self) -> int:
        return sum(1 for e in self.examples if e.label == "loss")

    @property
    def has_sufficient_data(self) -> bool:
        return (
            len(self.examples) >= self.MIN_TOTAL and
            self.n_wins >= self.MIN_PER_GROUP and
            self.n_losses >= self.MIN_PER_GROUP
        )

    # ───────────────────────────────────────────────────
    # CORE ANALYSIS
    # ───────────────────────────────────────────────────

    def compute_contrast(
        self,
        examples: Optional[List[OutcomeExample]] = None
    ) -> List[FeatureContrast]:
        """
        Compare win vs loss groups across all features.

        For each feature:
        1. Compute mean/std for winners and losers
        2. Compute effect size (Cohen's d)
        3. Simple significance test
        4. Determine direction
        5. Rank by importance

        Returns sorted list (most important first).
        """
        examples = examples or self.examples

        wins = [e for e in examples if e.label == "win"]
        losses = [e for e in examples if e.label == "loss"]

        if len(wins) < 3 or len(losses) < 3:
            logger.warning(
                f"Insufficient data: {len(wins)} wins, {len(losses)} losses. "
                f"Need at least 3 each."
            )
            return []

        # Get all feature names present in both groups
        win_features = set()
        loss_features = set()
        for w in wins:
            win_features.update(w.features.keys())
        for l in losses:
            loss_features.update(l.features.keys())

        common_features = win_features & loss_features

        contrasts = []
        for fname in common_features:
            contrast = self._compute_single_contrast(fname, wins, losses)
            if contrast is not None:
                contrasts.append(contrast)

        # Sort by importance (effect size × reliability factor)
        contrasts.sort(key=lambda c: c.importance, reverse=True)

        return contrasts

    def _compute_single_contrast(
        self,
        feature_name: str,
        wins: List[OutcomeExample],
        losses: List[OutcomeExample]
    ) -> Optional[FeatureContrast]:
        """Compute contrast for a single feature"""

        # Gather values
        win_vals = [
            w.features[feature_name]
            for w in wins
            if feature_name in w.features
            and np.isfinite(w.features[feature_name])
        ]
        loss_vals = [
            l.features[feature_name]
            for l in losses
            if feature_name in l.features
            and np.isfinite(l.features[feature_name])
        ]

        if len(win_vals) < 3 or len(loss_vals) < 3:
            return None

        win_arr = np.array(win_vals)
        loss_arr = np.array(loss_vals)

        win_mean = float(np.mean(win_arr))
        win_std = float(np.std(win_arr, ddof=1)) if len(win_arr) > 1 else 0.01
        loss_mean = float(np.mean(loss_arr))
        loss_std = float(np.std(loss_arr, ddof=1)) if len(loss_arr) > 1 else 0.01

        # Cohen's d
        pooled_std = np.sqrt((win_std**2 + loss_std**2) / 2)
        if pooled_std < 1e-10:
            effect_size = 0.0
        else:
            effect_size = abs(win_mean - loss_mean) / pooled_std

        # Simple t-test p-value (Welch's)
        p_value = self._welch_ttest_p(win_arr, loss_arr)

        # Direction
        diff = win_mean - loss_mean
        if abs(effect_size) < self.WEAK_EFFECT:
            direction = "neutral"
        elif diff > 0:
            direction = "higher_better"
        else:
            direction = "lower_better"

        # Reliability: enough samples AND significant
        reliable = (
            len(win_vals) >= self.MIN_PER_GROUP and
            len(loss_vals) >= self.MIN_PER_GROUP and
            p_value < 0.1
        )

        # Importance score: effect size weighted by reliability
        reliability_factor = 1.0 if reliable else 0.5
        sample_factor = min(1.0, (len(win_vals) + len(loss_vals)) / 40)
        importance = effect_size * reliability_factor * sample_factor

        return FeatureContrast(
            feature_name=feature_name,
            win_mean=win_mean,
            win_std=win_std,
            loss_mean=loss_mean,
            loss_std=loss_std,
            difference=float(diff),
            effect_size=float(effect_size),
            importance=float(importance),
            direction=direction,
            p_value=float(p_value),
            n_win=len(win_vals),
            n_loss=len(loss_vals),
            reliable=reliable
        )

    @staticmethod
    def _welch_ttest_p(a: np.ndarray, b: np.ndarray) -> float:
        """Simple Welch's t-test. Returns p-value."""
        n1, n2 = len(a), len(b)
        m1, m2 = np.mean(a), np.mean(b)
        v1 = np.var(a, ddof=1) if n1 > 1 else 0
        v2 = np.var(b, ddof=1) if n2 > 1 else 0

        se = np.sqrt(v1/n1 + v2/n2)
        if se < 1e-10:
            return 1.0

        t_stat = (m1 - m2) / se

        # Degrees of freedom (Welch-Satterthwaite)
        num = (v1/n1 + v2/n2) ** 2
        denom = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
        if denom < 1e-10:
            return 1.0
        df = num / denom

        # Approximate p-value using normal distribution for simplicity
        # (avoids scipy dependency for MVP)
        # For df > 30 this is very close to t-distribution
        z = abs(t_stat)
        if z > 4:
            return 0.0001
        # Simple sigmoid approximation of 2-tailed p
        p = 2.0 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        # Clamp
        return float(min(1.0, max(0.0001, p)))

    # ───────────────────────────────────────────────────
    # INSIGHT GENERATION
    # ───────────────────────────────────────────────────

    def generate_insight(
        self,
        regime: Optional[str] = None,
        top_n: int = 6
    ) -> ContrastInsight:
        """
        Generate human-readable contrast insight.

        Args:
            regime: If set, only analyze trades in this regime
            top_n: Number of top features to highlight
        """
        # Filter by regime if specified
        if regime:
            filtered = [e for e in self.examples if e.regime == regime]
        else:
            filtered = self.examples

        n_wins = sum(1 for e in filtered if e.label == "win")
        n_losses = sum(1 for e in filtered if e.label == "loss")
        data_sufficient = (
            len(filtered) >= self.MIN_TOTAL and
            n_wins >= self.MIN_PER_GROUP and
            n_losses >= self.MIN_PER_GROUP
        )

        # Compute contrasts
        all_contrasts = self.compute_contrast(filtered)

        # Filter to meaningful contrasts
        meaningful = [
            c for c in all_contrasts
            if c.effect_size >= self.WEAK_EFFECT
        ]
        top_features = meaningful[:top_n]

        # Compute confidence
        confidence = self._compute_insight_confidence(
            n_total=len(filtered),
            n_wins=n_wins,
            n_losses=n_losses,
            n_significant=len([c for c in meaningful if c.reliable])
        )

        # Generate summary
        summary = self._generate_summary(top_features, data_sufficient)

        insight = ContrastInsight(
            summary=summary,
            top_features=top_features,
            all_features=all_contrasts,
            confidence=confidence,
            n_wins=n_wins,
            n_losses=n_losses,
            n_total=len(filtered),
            regime=regime,
            data_sufficient=data_sufficient
        )

        self._last_insight = insight
        self._insight_stale = False

        return insight

    def _generate_summary(
        self,
        top_features: List[FeatureContrast],
        data_sufficient: bool
    ) -> str:
        """Generate natural language summary"""

        if not data_sufficient:
            return (
                "Not enough trades yet to identify reliable patterns. "
                "Keep trading and I'll analyze as data grows."
            )

        if not top_features:
            return (
                "No significant differences found between winning and "
                "losing setups. Outcomes may be driven by factors outside "
                "the current feature set, or the strategy has consistent "
                "execution with random outcome variance."
            )

        # Build summary from top features
        strong = [f for f in top_features if f.effect_size >= self.STRONG_EFFECT and f.reliable]
        moderate = [f for f in top_features if self.MODERATE_EFFECT <= f.effect_size < self.STRONG_EFFECT]

        parts = []

        if strong:
            feature_descriptions = []
            for f in strong[:3]:
                feature_descriptions.append(self._feature_to_plain_english(f))
            parts.append(
                "Strong differentiators: " + "; ".join(feature_descriptions)
            )

        if moderate:
            feature_descriptions = []
            for f in moderate[:3]:
                feature_descriptions.append(self._feature_to_plain_english(f))
            parts.append(
                "Also notable: " + "; ".join(feature_descriptions)
            )

        if not parts:
            # Only weak effects
            parts.append(
                "Subtle differences found but none are strong individually. "
                "Multiple small factors may combine to affect outcomes."
            )

        return " ".join(parts)

    @staticmethod
    def _feature_to_plain_english(fc: FeatureContrast) -> str:
        """Convert a feature contrast to plain English"""

        # Human-friendly feature names
        readable = {
            "wick_ratio_mean": "wick ratio",
            "body_consistency": "candle body consistency",
            "body_size_variance": "body size variance",
            "range_contraction_r2": "compression quality (R²)",
            "squeeze_score": "squeeze score",
            "volume_trend_slope": "volume trend",
            "volume_ratio_last5": "recent volume ratio",
            "volume_spike_count": "volume spikes",
            "volume_consistency": "volume consistency",
            "direction_entropy": "direction randomness",
            "rsi_at_setup": "RSI",
            "momentum_score": "momentum",
            "distance_to_nearest_support": "distance to support",
            "distance_to_nearest_resistance": "distance to resistance",
            "price_percentile_in_day_range": "price position in day range",
            "trend_direction_score": "trend direction",
            "adx": "trend strength (ADX)",
            "compression_candle_count": "compression duration",
            "range_volatility": "range volatility",
        }

        name = readable.get(fc.feature_name, fc.feature_name.replace("_", " "))

        if fc.direction == "higher_better":
            return f"winners had higher {name} ({fc.win_mean:.2f} vs {fc.loss_mean:.2f})"
        elif fc.direction == "lower_better":
            return f"winners had lower {name} ({fc.win_mean:.2f} vs {fc.loss_mean:.2f})"
        else:
            return f"{name} showed some difference but direction unclear"

    def _compute_insight_confidence(
        self,
        n_total: int,
        n_wins: int,
        n_losses: int,
        n_significant: int
    ) -> float:
        """
        How much should we trust this insight?

        Factors:
        - Sample size
        - Balance between wins and losses
        - Number of significant features found
        """
        # Sample size factor (saturates at ~100 trades)
        size_factor = min(1.0, n_total / 80)

        # Balance factor (worst: all wins or all losses)
        if n_total > 0:
            minority = min(n_wins, n_losses)
            balance = minority / (n_total / 2)  # 1.0 = perfect balance
            balance_factor = min(1.0, balance)
        else:
            balance_factor = 0.0

        # Signal factor (did we find anything?)
        signal_factor = min(1.0, n_significant / 3)

        confidence = (
            size_factor * 0.4 +
            balance_factor * 0.3 +
            signal_factor * 0.3
        )

        return round(confidence, 3)

    # ───────────────────────────────────────────────────
    # REGIME-WISE CONTRAST
    # ───────────────────────────────────────────────────

    def generate_regime_insights(self) -> Dict[str, ContrastInsight]:
        """
        Generate separate contrast insights per regime.

        This reveals if outcome drivers CHANGE by market condition.

        Example:
        - In bearish regime: compression quality matters most
        - In sideways regime: location matters most
        """
        regimes = set(e.regime for e in self.examples if e.regime)

        insights = {}

        # Overall (no regime filter)
        insights["all"] = self.generate_insight(regime=None)

        # Per regime
        for regime in regimes:
            regime_examples = [e for e in self.examples if e.regime == regime]
            n_wins = sum(1 for e in regime_examples if e.label == "win")
            n_losses = sum(1 for e in regime_examples if e.label == "loss")

            # Only analyze if enough data in this regime
            if n_wins >= 5 and n_losses >= 5:
                insights[regime] = self.generate_insight(regime=regime)
            else:
                logger.debug(
                    f"Skipping regime '{regime}': {n_wins}W/{n_losses}L "
                    f"(need 5 each)"
                )

        return insights

    # ───────────────────────────────────────────────────
    # PAIR-WISE CONTRAST (specific pairs)
    # ───────────────────────────────────────────────────

    def contrast_pair(
        self,
        win_example: OutcomeExample,
        loss_example: OutcomeExample,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compare two specific trades: one win, one loss.

        Useful for user asking:
        "Why did THIS trade win but THAT trade lose?"

        Returns the features where they differ most,
        contextualized by what we know from overall contrast.
        """
        common_features = (
            set(win_example.features.keys()) &
            set(loss_example.features.keys())
        )

        differences = []
        for fname in common_features:
            win_val = win_example.features[fname]
            loss_val = loss_example.features[fname]
            diff = win_val - loss_val

            if abs(diff) < 1e-10:
                continue

            # Contextualize with overall contrast knowledge
            overall_contrast = None
            if self._last_insight:
                for fc in self._last_insight.all_features:
                    if fc.feature_name == fname:
                        overall_contrast = fc
                        break

            # Score this difference
            # Higher score if: large difference AND aligns with overall pattern
            raw_importance = abs(diff)

            alignment = 0.0
            if overall_contrast and overall_contrast.effect_size > self.WEAK_EFFECT:
                # Does this pair's difference align with overall pattern?
                if (overall_contrast.direction == "higher_better" and diff > 0) or \
                   (overall_contrast.direction == "lower_better" and diff < 0):
                    alignment = overall_contrast.effect_size
                elif overall_contrast.direction != "neutral":
                    alignment = -overall_contrast.effect_size * 0.5  # Contradicts

            importance = raw_importance + alignment

            differences.append({
                "feature": fname,
                "win_value": float(win_val),
                "loss_value": float(loss_val),
                "difference": float(diff),
                "importance": float(importance),
                "aligns_with_overall": alignment > 0,
                "overall_direction": (
                    overall_contrast.direction
                    if overall_contrast else "unknown"
                ),
                "overall_effect_size": (
                    overall_contrast.effect_size
                    if overall_contrast else 0
                )
            })

        # Sort by importance
        differences.sort(key=lambda d: abs(d["importance"]), reverse=True)
        top = differences[:top_n]

        # Generate explanation
        explanations = []
        for d in top:
            name = d["feature"].replace("_", " ")
            if d["difference"] > 0:
                explanations.append(
                    f"Winner had higher {name} "
                    f"({d['win_value']:.3f} vs {d['loss_value']:.3f})"
                )
            else:
                explanations.append(
                    f"Winner had lower {name} "
                    f"({d['win_value']:.3f} vs {d['loss_value']:.3f})"
                )

        return {
            "top_differences": top,
            "explanation": (
                "Key differences between these two trades:\n• " +
                "\n• ".join(explanations)
                if explanations
                else "No significant differences found in measured features."
            ),
            "n_features_compared": len(common_features),
            "caveat": (
                "This is a single pair comparison. "
                "Individual differences may not generalize."
            )
        }

    # ───────────────────────────────────────────────────
    # AVT INTEGRATION
    # ───────────────────────────────────────────────────

    def suggest_weight_adjustments(
        self,
        current_weights: Dict[str, float],
        max_adjustment: float = 0.15
    ) -> List[WeightAdjustment]:
        """
        Suggest SOFT weight adjustments for AVT scoring
        based on what features actually drive outcomes.

        These are SUGGESTIONS, not commands.
        AVT decides whether to apply them.

        ARCHITECTURAL NOTE: These suggestions feed into
        self_improver.py's prove-it cycle. They are
        NEVER applied directly to AVT scoring config.

        Args:
            current_weights: Current feature weights in AVT
            max_adjustment: Maximum weight change allowed (caps aggression)
        """
        if not self.has_sufficient_data:
            return []

        insight = self._get_current_insight()
        if insight is None:
            return []

        adjustments = []

        for fc in insight.top_features:
            if fc.feature_name not in current_weights:
                continue

            if not fc.reliable:
                continue

            current = current_weights[fc.feature_name]

            # Compute suggested adjustment based on effect size
            # Strong effect → larger adjustment, but capped
            raw_adjustment = fc.effect_size * 0.1  # Scale factor
            capped_adjustment = min(raw_adjustment, max_adjustment)

            # Direction: increase weight if feature matters for outcome
            if fc.direction != "neutral":
                suggested = current + capped_adjustment
            else:
                continue  # Don't adjust neutral features

            # Confidence in this adjustment
            adjustment_confidence = (
                insight.confidence *
                (1.0 if fc.reliable else 0.5) *
                min(1.0, fc.effect_size / self.STRONG_EFFECT)
            )

            adjustments.append(WeightAdjustment(
                feature_name=fc.feature_name,
                current_weight=current,
                suggested_weight=round(suggested, 4),
                reason=(
                    f"{fc.feature_name} shows {fc.direction} pattern "
                    f"(effect: {fc.effect_size:.2f}, p={fc.p_value:.3f}). "
                    f"Winners: {fc.win_mean:.3f}, Losers: {fc.loss_mean:.3f}"
                ),
                confidence=round(adjustment_confidence, 3),
                is_suggestion=True  # Always a suggestion, never forced
            ))

        # Sort by confidence
        adjustments.sort(key=lambda a: a.confidence, reverse=True)
        return adjustments

    # ───────────────────────────────────────────────────
    # JUDGMENT COMPARISON
    # ───────────────────────────────────────────────────

    def compare_with_judgment(
        self,
        judgment_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare what USER thinks matters (judgment importance)
        vs what ACTUALLY matters (outcome-based contrast).

        This reveals blind spots:
        "You focus on compression quality, but location
         matters more for actual outcomes."

        NOTE: This method READS user intent (judgment_importance)
        but NEVER WRITES back to it. One-way data flow.

        Args:
            judgment_importance: Feature importance from onboarding
                               (what user thinks matters)
        """
        insight = self._get_current_insight()
        if insight is None or not insight.data_sufficient:
            return {
                "status": "insufficient_data",
                "message": "Need more outcome data to compare."
            }

        # Build outcome importance from contrast
        outcome_importance = {
            fc.feature_name: fc.importance
            for fc in insight.all_features
            if fc.effect_size >= self.WEAK_EFFECT
        }

        # Normalize both to sum to 1
        j_total = sum(judgment_importance.values()) or 1
        o_total = sum(outcome_importance.values()) or 1

        j_normalized = {k: v/j_total for k, v in judgment_importance.items()}
        o_normalized = {k: v/o_total for k, v in outcome_importance.items()}

        all_features = set(j_normalized.keys()) | set(o_normalized.keys())

        alignments = []
        blind_spots = []
        over_weighted = []

        for fname in all_features:
            j_weight = j_normalized.get(fname, 0)
            o_weight = o_normalized.get(fname, 0)
            gap = o_weight - j_weight

            item = {
                "feature": fname,
                "judgment_weight": round(j_weight, 4),
                "outcome_weight": round(o_weight, 4),
                "gap": round(gap, 4)
            }

            if abs(gap) < 0.03:
                alignments.append(item)
            elif gap > 0.03:
                # Outcome says important, user doesn't focus on it
                blind_spots.append(item)
            else:
                # User focuses on it, but outcomes don't support
                over_weighted.append(item)

        # Sort by gap magnitude
        blind_spots.sort(key=lambda x: x["gap"], reverse=True)
        over_weighted.sort(key=lambda x: abs(x["gap"]), reverse=True)

        return {
            "status": "complete",
            "aligned": alignments,
            "blind_spots": blind_spots[:5],
            "over_weighted": over_weighted[:5],
            "summary": self._generate_judgment_comparison_summary(
                blind_spots, over_weighted, alignments
            ),
            "confidence": insight.confidence
        }

    def _generate_judgment_comparison_summary(
        self,
        blind_spots: List[Dict],
        over_weighted: List[Dict],
        aligned: List[Dict]
    ) -> str:
        """Generate summary of judgment vs outcome comparison"""
        parts = []

        if aligned:
            parts.append(
                f"{len(aligned)} features are well-aligned between "
                f"your judgment and actual outcomes."
            )

        if blind_spots:
            names = [b["feature"].replace("_", " ") for b in blind_spots[:3]]
            parts.append(
                f"Potential blind spots (matter for outcomes but you may "
                f"under-weight): {', '.join(names)}"
            )

        if over_weighted:
            names = [o["feature"].replace("_", " ") for o in over_weighted[:3]]
            parts.append(
                f"Possibly over-weighted (you focus on these but outcomes "
                f"don't strongly support): {', '.join(names)}"
            )

        if not blind_spots and not over_weighted:
            parts.append(
                "Your judgment aligns well with outcome data. "
                "No major blind spots detected."
            )

        return " ".join(parts)

    # ───────────────────────────────────────────────────
    # IMPROVEMENT SUGGESTIONS
    # ───────────────────────────────────────────────────

    def generate_improvement_suggestions(
        self, max_suggestions: int = 5
    ) -> List[ImprovementSuggestion]:
        """
        Generate SOFT improvement suggestions.

        These are presented as insights, not rules.

        Example:
        "Consider paying more attention to distance_to_resistance.
         Winning trades averaged 2.1 ATR away while losing trades
         averaged 0.9 ATR. (Confidence: 71%)"
        """
        insight = self._get_current_insight()
        if insight is None or not insight.data_sufficient:
            return []

        suggestions = []

        for fc in insight.top_features:
            if fc.effect_size < self.MODERATE_EFFECT:
                continue

            # Determine suggestion type
            if fc.reliable and fc.effect_size >= self.STRONG_EFFECT:
                strength = "strong"
            elif fc.reliable:
                strength = "moderate"
            else:
                strength = "weak"

            readable_name = fc.feature_name.replace("_", " ")

            if fc.direction == "higher_better":
                description = (
                    f"Consider favoring setups with higher {readable_name}. "
                    f"Your winners averaged {fc.win_mean:.3f} while losers "
                    f"averaged {fc.loss_mean:.3f}."
                )
                direction = "increase_attention"
            elif fc.direction == "lower_better":
                description = (
                    f"Consider favoring setups with lower {readable_name}. "
                    f"Your winners averaged {fc.win_mean:.3f} while losers "
                    f"averaged {fc.loss_mean:.3f}."
                )
                direction = "increase_attention"
            else:
                continue

            evidence = (
                f"Based on {fc.n_win} wins and {fc.n_loss} losses. "
                f"Effect size: {fc.effect_size:.2f} "
                f"({'strong' if fc.effect_size >= self.STRONG_EFFECT else 'moderate'}). "
                f"p-value: {fc.p_value:.3f}."
            )

            suggestions.append(ImprovementSuggestion(
                description=description,
                evidence=evidence,
                feature_involved=fc.feature_name,
                direction=direction,
                strength=strength,
                confidence=round(
                    insight.confidence * (1.0 if fc.reliable else 0.6),
                    3
                )
            ))

        # Sort by confidence × strength
        strength_map = {"strong": 3, "moderate": 2, "weak": 1}
        suggestions.sort(
            key=lambda s: s.confidence * strength_map.get(s.strength, 1),
            reverse=True
        )

        return suggestions[:max_suggestions]

    # ───────────────────────────────────────────────────
    # TEMPORAL ANALYSIS (is the pattern changing?)
    # ───────────────────────────────────────────────────

    def check_pattern_stability(
        self, recent_window: int = 30
    ) -> Dict[str, Any]:
        """
        Check if outcome drivers are STABLE or SHIFTING.

        Compares recent trades vs older trades.
        If key features change importance, it means
        market regime may be shifting.
        """
        if len(self.examples) < recent_window * 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {recent_window * 2} trades."
            }

        # Sort by timestamp if available, otherwise by insertion order
        sorted_examples = sorted(
            self.examples,
            key=lambda e: e.timestamp or datetime.min
        )

        older = sorted_examples[:-recent_window]
        recent = sorted_examples[-recent_window:]

        older_contrasts = {
            c.feature_name: c
            for c in self.compute_contrast(older)
        }
        recent_contrasts = {
            c.feature_name: c
            for c in self.compute_contrast(recent)
        }

        shifts = []
        all_features = set(older_contrasts.keys()) | set(recent_contrasts.keys())

        for fname in all_features:
            old_c = older_contrasts.get(fname)
            new_c = recent_contrasts.get(fname)

            if old_c is None or new_c is None:
                continue

            # Check if importance changed significantly
            importance_shift = abs(new_c.importance - old_c.importance)
            direction_changed = old_c.direction != new_c.direction

            if importance_shift > 0.3 or direction_changed:
                shifts.append({
                    "feature": fname,
                    "old_importance": old_c.importance,
                    "new_importance": new_c.importance,
                    "old_direction": old_c.direction,
                    "new_direction": new_c.direction,
                    "direction_changed": direction_changed,
                    "importance_shift": importance_shift
                })

        shifts.sort(key=lambda s: s["importance_shift"], reverse=True)

        is_stable = len(shifts) <= 2

        return {
            "status": "analyzed",
            "stable": is_stable,
            "n_shifts": len(shifts),
            "shifts": shifts[:5],
            "summary": (
                "Outcome drivers are stable. Current insights remain valid."
                if is_stable
                else (
                    f"Detected {len(shifts)} shifting features. "
                    f"Market dynamics may be changing. "
                    f"Recent data should be weighted more heavily."
                )
            ),
            "recommendation": (
                "maintain_weights"
                if is_stable
                else "increase_recency_weight"
            )
        }

    # ───────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ───────────────────────────────────────────────────

    def _get_current_insight(self) -> Optional[ContrastInsight]:
        """Get current insight, recomputing if stale"""
        if self._insight_stale or self._last_insight is None:
            if self.has_sufficient_data:
                return self.generate_insight()
        return self._last_insight

    def get_status(self) -> Dict[str, Any]:
        """Current state of the contrast learner"""
        return {
            "strategy_id": self.strategy_id,
            "total_examples": len(self.examples),
            "wins": self.n_wins,
            "losses": self.n_losses,
            "sufficient_data": self.has_sufficient_data,
            "regimes_covered": list(set(
                e.regime for e in self.examples if e.regime
            )),
            "has_insight": self._last_insight is not None,
            "insight_stale": self._insight_stale
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence"""
        insight = self._last_insight
        return {
            "strategy_id": self.strategy_id,
            "n_examples": len(self.examples),
            "examples": [
                {
                    "features": e.features,
                    "outcome": e.outcome,
                    "label": e.label,
                    "regime": e.regime,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "symbol": e.symbol
                }
                for e in self.examples
            ],
            "last_insight": {
                "summary": insight.summary,
                "confidence": insight.confidence,
                "n_top_features": len(insight.top_features),
                "top_feature_names": [
                    f.feature_name for f in insight.top_features
                ]
            } if insight else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContrastLearner':
        """Restore from serialized state"""
        learner = cls(strategy_id=data.get("strategy_id", "default"))
        for ex_data in data.get("examples", []):
            learner.add_example(OutcomeExample(
                features=ex_data["features"],
                outcome=ex_data["outcome"],
                label=ex_data.get("label", ""),
                regime=ex_data.get("regime"),
                timestamp=(
                    datetime.fromisoformat(ex_data["timestamp"])
                    if ex_data.get("timestamp") else None
                ),
                symbol=ex_data.get("symbol")
            ))
        return learner
