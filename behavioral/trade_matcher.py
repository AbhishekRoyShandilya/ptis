"""
Per-Trade Strategy Auto-Matcher
=================================

For a given closed trade + its replay data, score every active strategy
template and pick the best match.

Result is written to `StrategyAttribution.tier2_best_match_id` and
`tier2_matches` (JSONB list of all candidates with scores).

Why this exists:
  - The Trade table has `declared_strategy_id` (Tier 1 — what the user said
    they were trading) but no auto-classification of what their entry
    conditions ACTUALLY matched.
  - When user said "ema_pullback" but entered far from the 20 EMA, this
    matcher catches the discrepancy → flagged for the CausalLinker.

Reuses extensively:
  - `strategy_engine/strategy_registry.py:get_registry().get_all_active()` for templates
  - `strategy_engine/templates/loader.py:ConditionDef` for the condition spec
  - The replay's IndicatorSnapshot values to evaluate conditions

Scoring:
  Each condition_type has a small evaluator that returns 0-1 match score.
  Strategy total_score = weighted average of its conditions' scores.
  Best match = highest score, capped to those ≥ MATCH_MIN_THRESHOLD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from behavioral.replay_models import IndicatorSnapshot, ReplayedTrade

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────


MATCH_MIN_THRESHOLD = 0.35       # below this, "no clear match"
MATCH_HIGH_CONFIDENCE = 0.70     # above this, mark as confident match


# ── Output types ────────────────────────────────────────────────────


@dataclass
class ConditionScore:
    """How well one condition matched the trade's entry context."""
    condition_type: str
    description: str
    required: bool
    weight: float
    score: float                 # 0-1
    detail: str = ""             # human-readable why


@dataclass
class StrategyMatchScore:
    """Score for a single strategy candidate."""
    strategy_id: str
    strategy_name: str
    total_score: float           # weighted avg, 0-1
    required_score: float        # weighted avg of required-only conditions
    condition_scores: List[ConditionScore] = field(default_factory=list)


@dataclass
class TradeMatchResult:
    """
    Top-level matcher output for a single trade.

    `best_match_id` is None when no strategy clears MATCH_MIN_THRESHOLD —
    means "this trade doesn't fit any of your configured strategies".
    """
    trade_id: str
    best_match_id: Optional[str]
    best_match_name: Optional[str]
    best_match_score: float
    confidence: str              # "high" | "moderate" | "low" | "none"
    declared_match: Optional[bool] = None  # did Tier 2 match Tier 1 (user-declared)?
    candidates: List[StrategyMatchScore] = field(default_factory=list)


# ── Matcher ─────────────────────────────────────────────────────────


class TradeStrategyMatcher:
    """
    Score each active strategy against a replayed trade's entry context.

    Usage:
        matcher = TradeStrategyMatcher()
        result = matcher.match(replay, declared_strategy_id="ema_pullback")
    """

    def __init__(self, registry: Any = None):
        if registry is None:
            try:
                from strategy_engine.strategy_registry import get_registry
                registry = get_registry()
                if hasattr(registry, "is_initialized") and not registry.is_initialized:
                    try:
                        registry.initialize()
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("StrategyRegistry not available: %s", e)
                registry = None
        self._registry = registry

    def match(
        self,
        replay: ReplayedTrade,
        declared_strategy_id: Optional[str] = None,
        allowed_strategy_ids: Optional[List[str]] = None,
    ) -> TradeMatchResult:
        """
        Score the user's active strategies and return the best match.

        allowed_strategy_ids: when provided (the trade owner's strategy
        short_codes), only those strategies are considered — so a user's trade
        is never matched against another user's strategy (multi-user correctness).
        """
        templates = []
        try:
            if self._registry is not None:
                templates = list(self._registry.get_all_active())
        except Exception as e:
            logger.warning("Failed to load templates: %s", e)

        # Scope to the trade owner's own strategies.
        if allowed_strategy_ids:
            allowed = {str(a).lower() for a in allowed_strategy_ids if a}
            templates = [
                t for t in templates
                if str(getattr(t, "id", "")).lower() in allowed
                or str(getattr(t, "name", "")).lower() in allowed
            ]

        result = TradeMatchResult(
            trade_id=replay.trade_id,
            best_match_id=None,
            best_match_name=None,
            best_match_score=0.0,
            confidence="none",
            candidates=[],
        )

        if not templates:
            logger.debug("No active strategies — cannot match trade %s", replay.trade_id)
            return result

        # Score each template
        scores: List[StrategyMatchScore] = []
        for tpl in templates:
            try:
                s = self._score_template(tpl, replay)
                scores.append(s)
            except Exception as e:
                logger.debug("Score failed for %s: %s", getattr(tpl, "id", "?"), e)
                continue

        if not scores:
            return result

        # Sort by total_score; tie-break by required_score
        scores.sort(key=lambda s: (s.total_score, s.required_score), reverse=True)
        result.candidates = scores

        best = scores[0]
        if best.total_score >= MATCH_MIN_THRESHOLD:
            result.best_match_id = best.strategy_id
            result.best_match_name = best.strategy_name
            result.best_match_score = best.total_score
            if best.total_score >= MATCH_HIGH_CONFIDENCE:
                result.confidence = "high"
            elif best.total_score >= 0.55:
                result.confidence = "moderate"
            else:
                result.confidence = "low"
        else:
            result.confidence = "none"

        # Compare to user-declared strategy
        if declared_strategy_id:
            result.declared_match = (result.best_match_id == declared_strategy_id)

        return result

    # ── Scoring per template ──

    def _score_template(self, template: Any, replay: ReplayedTrade) -> StrategyMatchScore:
        confs = list(getattr(template, "entry_confirmations", []) or [])
        condition_scores: List[ConditionScore] = []
        total_weight = 0.0
        weighted_sum = 0.0
        required_weight = 0.0
        required_sum = 0.0

        for cond in confs:
            score, detail = self._score_condition(cond, replay)
            cs = ConditionScore(
                condition_type=cond.type,
                description=getattr(cond, "description", "") or cond.type,
                required=bool(getattr(cond, "required", True)),
                weight=float(getattr(cond, "weight", 1.0) or 1.0),
                score=round(score, 4),
                detail=detail,
            )
            condition_scores.append(cs)
            w = cs.weight
            total_weight += w
            weighted_sum += score * w
            if cs.required:
                required_weight += w
                required_sum += score * w

        # Regime preference contributes a small bonus
        regime_bonus = self._score_regime_match(template, replay)

        if total_weight > 0:
            base_score = weighted_sum / total_weight
        else:
            base_score = 0.0
        if required_weight > 0:
            req_score = required_sum / required_weight
        else:
            req_score = base_score

        # Required conditions weighted heavier in final score
        final_score = 0.7 * req_score + 0.2 * base_score + 0.1 * regime_bonus
        final_score = max(0.0, min(1.0, final_score))

        return StrategyMatchScore(
            strategy_id=getattr(template, "id", "") or "",
            strategy_name=getattr(template, "name", "") or "",
            total_score=round(final_score, 4),
            required_score=round(req_score, 4),
            condition_scores=condition_scores,
        )

    # ── Per-condition scoring ──
    # Each condition type has its own evaluator. Returns (score 0-1, human-readable detail).

    def _score_condition(self, cond: Any, replay: ReplayedTrade) -> Tuple[float, str]:
        ctype = (getattr(cond, "type", "") or "").lower()
        params = getattr(cond, "params", {}) or {}
        snap = replay.snapshot_at_entry

        # Route to type-specific scorer
        scorer = _CONDITION_SCORERS.get(ctype, _score_unknown)
        return scorer(params, replay, snap)

    @staticmethod
    def _score_regime_match(template: Any, replay: ReplayedTrade) -> float:
        """0-1 bonus based on whether trade's regime matches template preferences."""
        if not replay.observations.regime_at_entry:
            return 0.5  # neutral when unknown
        regime_at_entry = replay.observations.regime_at_entry
        preferred = [
            getattr(r, "value", str(r)).lower()
            for r in getattr(template, "preferred_regimes", []) or []
        ]
        avoid = [
            getattr(r, "value", str(r)).lower()
            for r in getattr(template, "avoid_regimes", []) or []
        ]
        if regime_at_entry in preferred:
            return 1.0
        if regime_at_entry in avoid:
            return 0.0
        return 0.5  # neutral / unknown


# ── Condition scorers ──────────────────────────────────────────────


def _score_unknown(params, replay, snap):
    """Conditions we don't have an explicit scorer for — neutral score."""
    return 0.5, "unknown condition type — neutral score"


def _score_ema_proximity(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """How close was price to the target EMA at entry?"""
    if snap is None:
        return 0.0, "no entry snapshot"
    period = int(params.get("ema_period", 20))
    max_dist_pct = float(params.get("max_distance_pct", 0.3))  # in %

    ema_value = None
    if period <= 9 and snap.ema_9 is not None:
        ema_value = snap.ema_9
    elif period <= 21 and snap.ema_20 is not None:
        ema_value = snap.ema_20
    elif snap.ema_50 is not None:
        ema_value = snap.ema_50

    if ema_value is None or snap.close <= 0:
        return 0.0, f"EMA{period} not available"

    dist_pct = abs(snap.close - ema_value) / snap.close * 100
    if dist_pct <= max_dist_pct:
        score = 1.0
    elif dist_pct <= max_dist_pct * 2:
        # Partial credit: 1.0 → 0.0 linearly over 2x max distance
        score = 1.0 - ((dist_pct - max_dist_pct) / max_dist_pct)
    else:
        score = 0.0
    return max(0.0, min(1.0, score)), f"price {dist_pct:.2f}% from EMA{period} (limit {max_dist_pct}%)"


def _score_trend_alignment(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """Is price on the right side of the trend EMA?"""
    if snap is None:
        return 0.0, "no entry snapshot"
    period = int(params.get("ema_period", 50))
    is_long = (replay.direction or "long").lower() == "long"

    ema_value = snap.ema_50 if period >= 50 else snap.ema_20
    if ema_value is None or snap.close <= 0:
        return 0.5, "trend EMA not available — neutral"

    aligned = (snap.close > ema_value) if is_long else (snap.close < ema_value)
    return (1.0 if aligned else 0.0), (
        f"close {snap.close:.2f} vs EMA{period} {ema_value:.2f} — "
        f"{'aligned' if aligned else 'against trend'}"
    )


def _score_momentum_reset(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """RSI in the reset zone (typically 35-65)?"""
    if snap is None or snap.rsi_14 is None:
        return 0.5, "RSI not available"
    rsi_min = float(params.get("rsi_min", 35))
    rsi_max = float(params.get("rsi_max", 65))
    rsi = float(snap.rsi_14)
    if rsi_min <= rsi <= rsi_max:
        return 1.0, f"RSI {rsi:.1f} in reset zone [{rsi_min}-{rsi_max}]"
    # Partial credit for being close
    if rsi < rsi_min:
        gap = rsi_min - rsi
    else:
        gap = rsi - rsi_max
    score = max(0.0, 1.0 - gap / 15.0)  # full credit lost ~15 RSI points away
    return score, f"RSI {rsi:.1f} outside reset zone [{rsi_min}-{rsi_max}]"


def _score_above_vwap(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """Bullish: price > VWAP. Bearish: price < VWAP."""
    if snap is None or snap.vwap is None:
        return 0.5, "VWAP not available"
    is_long = (replay.direction or "long").lower() == "long"
    aligned = (snap.close > snap.vwap) if is_long else (snap.close < snap.vwap)
    return (1.0 if aligned else 0.0), (
        f"close {snap.close:.2f} vs VWAP {snap.vwap:.2f} — "
        f"{'aligned' if aligned else 'against'}"
    )


def _score_volume_surge(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """Volume on entry candle ≥ N times the recent avg?"""
    if snap is None or snap.volume_ratio is None:
        return 0.5, "volume ratio not available"
    threshold = float(params.get("ratio", 1.5))
    ratio = float(snap.volume_ratio)
    if ratio >= threshold:
        return 1.0, f"volume {ratio:.2f}x avg ≥ {threshold}x"
    # Partial credit
    score = max(0.0, ratio / threshold)
    return score, f"volume {ratio:.2f}x avg < {threshold}x"


def _score_compression(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """BB width below threshold — squeeze present?"""
    if snap is None or snap.bb_width_pct is None:
        return 0.5, "BB width not available"
    max_width = float(params.get("max_width_pct", 4.0))
    width = float(snap.bb_width_pct)
    if width <= max_width:
        return 1.0, f"BB width {width:.2f}% ≤ {max_width}% (compressed)"
    # Linear partial credit up to 2x threshold
    score = max(0.0, 1.0 - (width - max_width) / max_width)
    return score, f"BB width {width:.2f}% > {max_width}% (not compressed)"


def _score_breakout(
    params: Dict[str, Any],
    replay: ReplayedTrade,
    snap: Optional[IndicatorSnapshot],
) -> Tuple[float, str]:
    """For ORB-style: price broke out of opening range. Approximated via
    move_pct_complete_at_entry — high value = entered AFTER significant move."""
    pct = replay.observations.move_pct_complete_at_entry
    if pct is None:
        return 0.5, "move-pct not computable"
    # Sweet spot: 30-70% completed (entered after confirmation but not late)
    if 0.30 <= pct <= 0.70:
        return 1.0, f"move {int(pct*100)}% complete — entered in confirmation zone"
    if pct < 0.30:
        return 0.6, f"move {int(pct*100)}% complete — early (pre-confirmation)"
    return max(0.0, 1.0 - (pct - 0.70) / 0.30), (
        f"move {int(pct*100)}% complete — late entry"
    )


# Map condition types → scorers (lowercase keys)
_CONDITION_SCORERS = {
    "ema_proximity": _score_ema_proximity,
    "trend_alignment": _score_trend_alignment,
    "momentum_reset": _score_momentum_reset,
    "above_vwap": _score_above_vwap,
    "vwap_position": _score_above_vwap,
    "volume_surge": _score_volume_surge,
    "volume_confirmation": _score_volume_surge,
    "compression": _score_compression,
    "bb_squeeze": _score_compression,
    "breakout": _score_breakout,
    "orb_breakout": _score_breakout,
}
