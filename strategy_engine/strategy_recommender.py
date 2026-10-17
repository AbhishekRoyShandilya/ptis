# strategy_engine/strategy_recommender.py
"""
Strategy Discovery Engine.

Given your trader DNA profile, finds strategies that:
1. Match your risk style and thinking style
2. Cover regimes where your current strategies don't work
3. Have similar or better risk-adjusted returns
4. Align with your decision patterns

This is NOT random suggestions.
This is style-aligned, regime-complementary strategy discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from core.enums import MarketRegime
from behavioral.trader_profile import TraderDNA

logger = logging.getLogger(__name__)


@dataclass
class StrategyCandidate:
    """A strategy suggestion with reasoning."""
    name: str
    description: str
    strategy_type: str         # breakout, pullback, reversal, etc.

    # Why it fits you
    style_match_score: float = 0.0     # 0-1
    regime_coverage_score: float = 0.0 # 0-1
    risk_compatibility: float = 0.0    # 0-1
    overall_score: float = 0.0        # 0-1

    # What it covers
    target_regimes: list[str] = field(default_factory=list)
    fills_gap_for: list[str] = field(default_factory=list)

    # How it compares
    expected_win_rate: Optional[float] = None
    expected_r_target: Optional[float] = None
    expected_risk_pct: Optional[float] = None

    # Why suggested
    reasons: list[str] = field(default_factory=list)

    # Implementation hints
    key_rules: list[str] = field(default_factory=list)
    similar_to: Optional[str] = None  # existing strategy it's most like

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.strategy_type,
            "scores": {
                "style_match": round(self.style_match_score, 2),
                "regime_coverage": round(self.regime_coverage_score, 2),
                "risk_compatibility": round(self.risk_compatibility, 2),
                "overall": round(self.overall_score, 2),
            },
            "target_regimes": self.target_regimes,
            "fills_gap_for": self.fills_gap_for,
            "expected_metrics": {
                "win_rate": self.expected_win_rate,
                "r_target": self.expected_r_target,
                "risk_pct": self.expected_risk_pct,
            },
            "reasons": self.reasons,
            "key_rules": self.key_rules,
            "similar_to": self.similar_to,
        }


# ══════════════════════════════════════════════
#  STRATEGY KNOWLEDGE BASE
# ══════════════════════════════════════════════

# strategy_engine/strategy_recommender.py (continued from STRATEGY_TEMPLATES)

STRATEGY_TEMPLATES = [
    {
        "name": "Momentum Continuation",
        "description": "Enter strong momentum after brief consolidation within trend. Wait for flag/pennant pattern to form, enter on breakout of consolidation with volume.",
        "type": "momentum",
        "thinking_style": ["momentum"],
        "timing_style": ["confirmation", "momentum"],
        "risk_style": ["moderate", "aggressive"],
        "regimes": ["trending_up", "trending_down"],
        "avoids": ["range", "low_vol"],
        "expected_win_rate": 0.45,
        "expected_r_target": 2.5,
        "expected_risk_pct": 1.0,
        "key_rules": [
            "Identify strong impulse move (3+ candles in direction)",
            "Wait for 3-8 bar consolidation (flag/pennant)",
            "Enter on breakout of consolidation range",
            "Volume must increase on breakout bar",
            "Stop below consolidation low (long) / above high (short)",
            "Target: 1:1 measured move of impulse leg",
        ],
    },
    {
        "name": "VWAP Bounce Trend Continuation",
        "description": "In trending markets, enter when price pulls back to VWAP and shows rejection. VWAP acts as dynamic support/resistance in trend.",
        "type": "pullback",
        "thinking_style": ["structure", "probability"],
        "timing_style": ["confirmation", "patient"],
        "risk_style": ["conservative", "moderate"],
        "regimes": ["trending_up", "trending_down"],
        "avoids": ["volatile", "low_vol"],
        "expected_win_rate": 0.55,
        "expected_r_target": 2.0,
        "expected_risk_pct": 0.75,
        "key_rules": [
            "Confirm intraday trend (price above VWAP for long, below for short)",
            "Wait for pullback to touch or wick through VWAP",
            "Rejection candle at VWAP (hammer, pin bar, engulfing)",
            "EMA 9 still sloping in trend direction",
            "Stop below VWAP wick (long) with ATR buffer",
            "T1: previous high/low, T2: 2R",
        ],
    },
    {
        "name": "Range Fade at Extremes",
        "description": "In range-bound markets, fade moves to range extremes with confirmation. Profit from mean reversion within established range.",
        "type": "mean_reversion",
        "thinking_style": ["probability", "structure"],
        "timing_style": ["patient", "confirmation"],
        "risk_style": ["conservative", "moderate"],
        "regimes": ["range", "low_vol"],
        "avoids": ["trending_up", "trending_down", "volatile"],
        "expected_win_rate": 0.60,
        "expected_r_target": 1.5,
        "expected_risk_pct": 0.5,
        "key_rules": [
            "Identify clear range (3+ touches on support/resistance)",
            "Wait for price to reach range extreme (top 10% or bottom 10%)",
            "Rejection candle at extreme (wick rejection, engulfing)",
            "RSI divergence or overbought/oversold at extreme",
            "ADX below 20 confirming range condition",
            "Stop beyond range extreme with buffer",
            "Target: opposite range extreme or VWAP",
        ],
    },
    {
        "name": "Volatility Expansion Breakout",
        "description": "After period of compression (Bollinger squeeze, narrow range), enter on the expansion move. Works when volatility returns to compressed markets.",
        "type": "breakout",
        "thinking_style": ["volatility", "momentum"],
        "timing_style": ["anticipatory", "momentum"],
        "risk_style": ["moderate", "aggressive"],
        "regimes": ["low_vol", "range"],  # enter in compression, profit in expansion
        "avoids": [],
        "expected_win_rate": 0.40,
        "expected_r_target": 3.0,
        "expected_risk_pct": 1.0,
        "key_rules": [
            "Identify Bollinger Band squeeze (bands narrowing, BW below 20-period low)",
            "Wait for first expansion candle breaking upper/lower band",
            "Volume confirms expansion (1.5x+ relative volume)",
            "Enter on close of first expansion candle",
            "Stop at midline of Bollinger Bands",
            "Trail with 2x ATR from highs",
        ],
    },
    {
        "name": "Gap Fill Reversal",
        "description": "When gap opening fails to follow through, trade the fill back toward previous close. Best with gaps into resistance/support.",
        "type": "mean_reversion",
        "thinking_style": ["probability", "structure"],
        "timing_style": ["early_bird", "reactive"],
        "risk_style": ["moderate"],
        "regimes": ["range", "volatile"],
        "avoids": ["trending_up", "trending_down"],
        "expected_win_rate": 0.55,
        "expected_r_target": 1.5,
        "expected_risk_pct": 0.75,
        "key_rules": [
            "Gap open > 0.5% from previous close",
            "Gap into known resistance (for gap up) or support (for gap down)",
            "First 5-10 min shows rejection (failed follow-through)",
            "Volume declining after initial gap candle",
            "Enter on break below opening 5-min candle low (for gap up fade)",
            "Stop above opening range high",
            "Target: previous day close or 50% gap fill",
        ],
    },
    {
        "name": "Supply/Demand Zone Rejection",
        "description": "Trade rejections at previously identified supply and demand zones. Structure-based approach using order flow concepts.",
        "type": "structure",
        "thinking_style": ["structure"],
        "timing_style": ["patient", "confirmation"],
        "risk_style": ["conservative", "moderate"],
        "regimes": ["trending_up", "trending_down", "range"],
        "avoids": ["volatile"],
        "expected_win_rate": 0.50,
        "expected_r_target": 2.0,
        "expected_risk_pct": 0.75,
        "key_rules": [
            "Identify fresh supply/demand zones from strong impulsive moves",
            "Wait for price to return to the zone (first or second revisit only)",
            "Confirmation candle showing rejection at zone boundary",
            "Volume spike at zone entry followed by absorption",
            "Stop beyond the zone with small buffer",
            "Target: origin of the move that created the zone",
        ],
    },
    {
        "name": "Trend Exhaustion Reversal",
        "description": "Identify overextended trends using RSI divergence and structure breaks. Counter-trend entry after trend shows clear exhaustion.",
        "type": "reversal",
        "thinking_style": ["probability", "structure"],
        "timing_style": ["patient", "confirmation"],
        "risk_style": ["conservative"],
        "regimes": ["trending_up", "trending_down"],
        "avoids": ["range", "low_vol"],
        "expected_win_rate": 0.40,
        "expected_r_target": 3.0,
        "expected_risk_pct": 0.5,
        "key_rules": [
            "Price makes new high/low but RSI makes lower high/higher low (divergence)",
            "ADX declining from above 30 (trend losing strength)",
            "Structure break: lower low after higher highs (for uptrend exhaustion)",
            "Volume declining on final push",
            "Wait for pullback to broken structure before entry",
            "Stop beyond the exhaustion extreme",
            "Target: previous consolidation zone or key structure level",
        ],
    },
    {
        "name": "Intraday Scalp — Order Flow",
        "description": "Quick scalps based on order flow imbalances at key levels. High frequency, small targets, tight stops.",
        "type": "scalp",
        "thinking_style": ["momentum", "volatility"],
        "timing_style": ["reactive", "momentum"],
        "risk_style": ["aggressive", "adaptive"],
        "regimes": ["volatile", "trending_up", "trending_down"],
        "avoids": ["low_vol"],
        "expected_win_rate": 0.55,
        "expected_r_target": 1.2,
        "expected_risk_pct": 0.25,
        "key_rules": [
            "Identify key intraday levels (VWAP, ORH/ORL, PDH/PDL)",
            "Watch for aggressive buying/selling (tape reading or L2)",
            "Enter on confirmed level break or rejection",
            "Very tight stop: 0.5x ATR",
            "Quick target: 1-1.5R",
            "Max 3-5 minute hold time",
            "Only in first 90 minutes or last 30 minutes",
        ],
    },
]


# ══════════════════════════════════════════════
#  DISCOVERY ENGINE
# ══════════════════════════════════════════════

class StrategyDiscoveryEngine:
    """
    Discovers complementary strategies based on your trader DNA.
    
    Priority logic:
    1. Fill regime gaps (where you have no strategy)
    2. Match your thinking/timing/risk style
    3. Offer better risk-adjusted alternatives for weak regimes
    4. Avoid suggesting strategies incompatible with your personality
    """

    def discover(
        self,
        dna: TraderDNA,
        max_suggestions: int = 5,
    ) -> list[StrategyCandidate]:
        """Find strategies that complement your existing setup."""
        candidates = []

        for template in STRATEGY_TEMPLATES:
            candidate = self._evaluate_candidate(template, dna)
            if candidate.overall_score > 0.3:  # minimum relevance threshold
                candidates.append(candidate)

        # Sort by overall score descending
        candidates.sort(key=lambda c: c.overall_score, reverse=True)

        return candidates[:max_suggestions]

    def _evaluate_candidate(
        self, template: dict, dna: TraderDNA
    ) -> StrategyCandidate:
        """Score a strategy template against trader DNA."""
        candidate = StrategyCandidate(
            name=template["name"],
            description=template["description"],
            strategy_type=template["type"],
            target_regimes=template.get("regimes", []),
            expected_win_rate=template.get("expected_win_rate"),
            expected_r_target=template.get("expected_r_target"),
            expected_risk_pct=template.get("expected_risk_pct"),
            key_rules=template.get("key_rules", []),
        )

        # ── 1. Style Match Score (40% weight) ──
        style_score = self._score_style_match(template, dna)
        candidate.style_match_score = style_score

        # ── 2. Regime Coverage Score (35% weight) ──
        regime_score, gaps_filled = self._score_regime_coverage(template, dna)
        candidate.regime_coverage_score = regime_score
        candidate.fills_gap_for = gaps_filled

        # ── 3. Risk Compatibility (25% weight) ──
        risk_score = self._score_risk_compatibility(template, dna)
        candidate.risk_compatibility = risk_score

        # ── Overall Score ──
        candidate.overall_score = (
            style_score * 0.40
            + regime_score * 0.35
            + risk_score * 0.25
        )

        # ── Generate Reasons ──
        candidate.reasons = self._generate_reasons(candidate, template, dna)

        # ── Find Similar Existing Strategy ──
        candidate.similar_to = self._find_most_similar(template, dna)

        return candidate

    def _score_style_match(self, template: dict, dna: TraderDNA) -> float:
        """How well does this strategy match your thinking/timing style."""
        score = 0.0
        checks = 0

        # Thinking style match
        template_thinking = template.get("thinking_style", [])
        if dna.thinking_profile.primary_style in template_thinking:
            score += 1.0
            checks += 1
        elif dna.thinking_profile.secondary_style in template_thinking:
            score += 0.7
            checks += 1
        else:
            checks += 1

        # Timing style match
        template_timing = template.get("timing_style", [])
        if dna.timing_profile.style in template_timing:
            score += 1.0
            checks += 1
        else:
            # Check if any timing style is close
            for ts in template_timing:
                if ts == dna.timing_profile.style:
                    score += 1.0
                    break
            checks += 1

        # Trader type compatibility
        template_type = template.get("type", "")
        if dna.trader_type == "scalper" and template_type == "scalp":
            score += 1.0
        elif dna.trader_type == "day_trader" and template_type in ("breakout", "momentum", "pullback", "mean_reversion"):
            score += 0.8
        elif dna.trader_type == "swing_trader" and template_type in ("structure", "reversal"):
            score += 0.8
        else:
            score += 0.3
        checks += 1

        return score / checks if checks > 0 else 0

    def _score_regime_coverage(
        self, template: dict, dna: TraderDNA
    ) -> tuple[float, list[str]]:
        """How well does this strategy fill regime gaps."""
        template_regimes = set(template.get("regimes", []))
        missing_regimes = set(dna.regime_adaptability.missing_regime_coverage)
        underperforming = set(dna.regime_adaptability.underperforming_regimes)

        gaps_filled = []

        # Covers missing regimes
        fills_missing = template_regimes & missing_regimes
        if fills_missing:
            gaps_filled.extend([f"Covers missing regime: {r}" for r in fills_missing])

        # Covers underperforming regimes
        fills_underperforming = template_regimes & underperforming
        if fills_underperforming:
            gaps_filled.extend([f"Alternative for weak regime: {r}" for r in fills_underperforming])

        # Score
        if not missing_regimes and not underperforming:
            return 0.3, gaps_filled  # no gaps to fill, but still relevant

        total_gaps = len(missing_regimes) + len(underperforming)
        filled = len(fills_missing) + len(fills_underperforming)

        if total_gaps > 0:
            return min(1.0, filled / total_gaps + 0.1), gaps_filled
        return 0.3, gaps_filled

    def _score_risk_compatibility(self, template: dict, dna: TraderDNA) -> float:
        """How well does this strategy's risk profile match yours."""
        template_risk_styles = template.get("risk_style", [])
        your_risk = dna.risk_profile.style

        if your_risk in template_risk_styles:
            score = 1.0
        elif your_risk == "adaptive" and len(template_risk_styles) > 1:
            score = 0.8  # adaptive traders can handle multiple risk styles
        elif your_risk == "moderate" and "conservative" in template_risk_styles:
            score = 0.7  # moderate can do conservative
        elif your_risk == "moderate" and "aggressive" in template_risk_styles:
            score = 0.5  # might be a stretch
        elif your_risk == "conservative" and "aggressive" in template_risk_styles:
            score = 0.2  # bad match
        else:
            score = 0.4

        # Check if expected risk is within your tolerance
        expected_risk = template.get("expected_risk_pct", 1.0)
        max_risk = dna.risk_profile.max_risk_pct_observed
        if max_risk > 0 and expected_risk <= max_risk:
            score = min(1.0, score + 0.1)
        elif expected_risk > max_risk * 1.5:
            score = max(0.1, score - 0.2)

        return score

    def _generate_reasons(
        self, candidate: StrategyCandidate, template: dict, dna: TraderDNA
    ) -> list[str]:
        """Generate human-readable reasons for the suggestion."""
        reasons = []

        # Style match
        if candidate.style_match_score >= 0.7:
            reasons.append(
                f"✅ Matches your {dna.thinking_profile.primary_style} thinking style"
            )

        # Regime gap
        if candidate.fills_gap_for:
            for gap in candidate.fills_gap_for[:2]:
                reasons.append(f"📊 {gap}")

        # Risk compatibility
        if candidate.risk_compatibility >= 0.7:
            reasons.append(
                f"⚖️ Compatible with your {dna.risk_profile.style} risk style"
            )

        # Performance potential
        if template.get("expected_r_target", 0) > dna.risk_profile.avg_risk_pct:
            reasons.append(
                f"📈 Higher R-target ({template['expected_r_target']}R) than your average"
            )

        # Timing fit
        template_timing = template.get("timing_style", [])
        if dna.timing_profile.style in template_timing:
            reasons.append(
                f"⏰ Fits your {dna.timing_profile.style} timing preference"
            )

        # Diversification
        template_thinking = template.get("thinking_style", [])
        if dna.thinking_profile.primary_style not in template_thinking:
            reasons.append(
                "🔄 Adds thinking style diversification to your playbook"
            )

        return reasons

    def _find_most_similar(self, template: dict, dna: TraderDNA) -> Optional[str]:
        """Find which of your learned strategy profiles is most similar."""
        from core.database import get_db
        from core.models import StrategyProfile

        try:
            db = next(get_db())
            profiles = db.query(StrategyProfile).filter(StrategyProfile.status == "active").all()
        except Exception:
            return None

        if not profiles:
            return None

        best_match = None
        best_score = 0.0
        template_regimes = set(template.get("regimes", []))

        for profile in profiles:
            score = 0.0

            # Regime overlap
            profile_regimes = set((profile.regime_preferences or {}).get("preferred", []))
            overlap = profile_regimes & template_regimes
            if overlap:
                score += 0.5 * (len(overlap) / max(len(profile_regimes), len(template_regimes), 1))

            # Risk similarity
            risk_params = profile.risk_parameters or {}
            if abs(risk_params.get("max_risk_pct", 1.0) - template.get("expected_risk_pct", 1.0)) < 0.5:
                score += 0.5

            if score > best_score:
                best_score = score
                best_match = profile.short_code

        return best_match