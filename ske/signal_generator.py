# ske/signal_generator.py
"""
Signal Generator — evaluates market data against an SKO's fuzzy scoring
config to produce real-time trade signals with conviction scores, entry
zones, and sizing recommendations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from core.enums import ConvictionBand
from ske.feature_computer import FeatureComputer
from ske.models import (
    ConvictionResult,
    EntryZone,
    FeatureScore,
    RegimeFingerprint,
    SignalOutput,
    ScoringParameter,
    VariantInfo,
)
from ske.regime_fingerprint import RegimeFingerprintComputer
from ske.scoring import compute_conviction, compute_feature_score
from ske.variant_engine import VariantEngine

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates trade signals by running market data through an SKO's
    fuzzy scoring pipeline.

    Pipeline:
        1. Compute feature vector from candles
        2. Score each feature against SKO's scoring_config
        3. Aggregate → conviction score → conviction band
        4. Check regime match
        5. Classify into variant
        6. Compute entry zone, stop, targets
        7. Return SignalOutput
    """

    def __init__(
        self,
        feature_computer: Optional[FeatureComputer] = None,
        regime_computer: Optional[RegimeFingerprintComputer] = None,
        variant_engine: Optional[VariantEngine] = None,
    ):
        self._feature_computer = feature_computer or FeatureComputer()
        self._regime_computer = regime_computer or RegimeFingerprintComputer()
        self._variant_engine = variant_engine or VariantEngine()

    def evaluate(
        self,
        sko: Dict[str, Any],
        candles: List[Dict[str, Any]],
        symbol: str,
        variants: Optional[List[VariantInfo]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalOutput]:
        """
        Evaluate candles against an SKO and produce a signal (or None if skip).

        Args:
            sko: Dict with SKO fields (scoring_config, conviction_map, etc.)
            candles: OHLCV bars (oldest first), minimum 30 bars.
            symbol: Instrument symbol.
            variants: Pre-loaded variants for classification.
            market_data: External market data for regime fingerprint.

        Returns:
            SignalOutput or None if conviction_band == SKIP.
        """
        if len(candles) < 5:
            logger.warning("Too few candles for signal evaluation: %d", len(candles))
            return None

        scoring_config = sko.get("scoring_config", {})
        conviction_map = sko.get("conviction_map", {})
        sko_id = sko.get("id")

        # 1. Compute feature vector
        feature_names = list(scoring_config.keys())
        computer = FeatureComputer(feature_list=feature_names) if feature_names else self._feature_computer
        features = computer.compute_features(candles, symbol)

        if not features:
            return None

        # 2. Score each feature (base scoring)
        feature_scores: List[FeatureScore] = []
        for feat_name, param_dict in scoring_config.items():
            raw_value = features.get(feat_name)
            if raw_value is None:
                continue

            param = ScoringParameter(**param_dict) if isinstance(param_dict, dict) else param_dict
            score = compute_feature_score(raw_value, param)

            feature_scores.append(FeatureScore(
                feature_name=feat_name,
                raw_value=raw_value,
                normalized_score=score,
                weight=param.weight if isinstance(param, ScoringParameter) else 1.0,
            ))

        # 3. Aggregate conviction (base)
        conviction = compute_conviction(feature_scores, conviction_map or None)

        # 4. Skip if conviction is too low
        if conviction.conviction_band == ConvictionBand.SKIP:
            return None

        # 5. Regime fingerprint
        regime_fp = self._regime_computer.compute(candles, market_data)
        regime_prefs = sko.get("regime_fingerprint", {}).get("preferred", [])
        regime_match = self._regime_computer.match_score(regime_fp, regime_prefs)

        # 6. Variant classification
        matched_variant = None
        variant_sim = 0.0
        variant_provenance = None
        variant_conviction_boost = 0.0
        variant_regime_match = 0.0
        variant_narrative = None

        if variants:
            label, sim = self._variant_engine.classify_trade(features, variants)
            matched_variant = label
            variant_sim = sim

            # Find the matched variant object for enrichment
            matched_obj = None
            if label:
                matched_obj = next(
                    (v for v in variants if v.label == label), None
                )

            if matched_obj:
                variant_provenance = matched_obj.provenance
                variant_narrative = matched_obj.narrative or None

                # 6a. Per-variant scoring overrides
                if matched_obj.scoring_overrides and variant_sim > 0.5:
                    variant_scores = self._apply_variant_scoring(
                        features, scoring_config, matched_obj.scoring_overrides
                    )
                    if variant_scores:
                        variant_conviction = compute_conviction(
                            variant_scores, conviction_map or None
                        )
                        # Blend: higher similarity → more weight to variant scoring
                        blend = min(variant_sim, 0.8)
                        blended_score = (
                            conviction.weighted_score * (1 - blend)
                            + variant_conviction.weighted_score * blend
                        )
                        variant_conviction_boost = blended_score - conviction.weighted_score

                        # Update conviction with blended score
                        conviction = ConvictionResult(
                            weighted_score=max(0.0, min(1.0, blended_score)),
                            conviction_band=conviction.conviction_band,
                            individual_scores=variant_scores,
                            sizing_recommendation=conviction.sizing_recommendation,
                            size_multiplier=conviction.size_multiplier,
                        )
                        feature_scores = variant_scores

                # 6b. Variant regime affinity matching
                if matched_obj.regime_affinity and regime_fp:
                    variant_regime_match = self._compute_regime_affinity_match(
                        regime_fp, matched_obj.regime_affinity
                    )
                    # Boost/dampen conviction based on regime fit (up to ±15%)
                    regime_adjustment = (variant_regime_match - 0.5) * 0.30
                    adjusted_score = conviction.weighted_score + regime_adjustment
                    adjusted_score = max(0.0, min(1.0, adjusted_score))
                    variant_conviction_boost += (adjusted_score - conviction.weighted_score)

                    conviction = ConvictionResult(
                        weighted_score=adjusted_score,
                        conviction_band=conviction.conviction_band,
                        individual_scores=conviction.individual_scores,
                        sizing_recommendation=conviction.sizing_recommendation,
                        size_multiplier=conviction.size_multiplier,
                    )

                # 6c. Anti-variant penalty
                if matched_obj.is_anti_variant and variant_sim > 0.6:
                    logger.info(
                        "Anti-variant match for %s: %s (sim=%.2f) — applying penalty",
                        symbol, label, variant_sim,
                    )
                    penalty = variant_sim * 0.4  # up to 40% penalty
                    penalized = max(0.0, conviction.weighted_score - penalty)
                    variant_conviction_boost += (penalized - conviction.weighted_score)
                    conviction = ConvictionResult(
                        weighted_score=penalized,
                        conviction_band=conviction.conviction_band,
                        individual_scores=conviction.individual_scores,
                        sizing_recommendation="reduce" if penalty > 0.2 else conviction.sizing_recommendation,
                        size_multiplier=max(0.3, conviction.size_multiplier - penalty),
                    )

        # 7. Entry zone, stop, targets
        entry_zone = self._compute_entry_zone(sko, candles, features)
        stop_level = self._compute_stop(sko, candles, features)
        target_levels = self._compute_targets(sko, candles, features, stop_level)

        return SignalOutput(
            sko_id=sko_id,
            sko_name=sko.get("name", ""),
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            conviction_score=conviction.weighted_score,
            conviction_band=conviction.conviction_band,
            individual_scores=feature_scores,
            sizing_recommendation=conviction.sizing_recommendation,
            size_multiplier=conviction.size_multiplier,
            entry_zone=entry_zone,
            stop_level=stop_level,
            target_levels=target_levels,
            regime_fingerprint=regime_fp,
            regime_match_score=regime_match,
            matched_variant=matched_variant,
            variant_similarity=variant_sim,
            variant_provenance=variant_provenance,
            variant_conviction_boost=round(variant_conviction_boost, 4),
            variant_regime_match=round(variant_regime_match, 4),
            variant_narrative=variant_narrative,
        )

    def evaluate_batch(
        self,
        sko: Dict[str, Any],
        symbols_candles: Dict[str, List[Dict[str, Any]]],
        variants: Optional[List[VariantInfo]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[SignalOutput]:
        """Evaluate multiple symbols against one SKO."""
        signals = []
        for symbol, candles in symbols_candles.items():
            signal = self.evaluate(sko, candles, symbol, variants, market_data)
            if signal is not None:
                signals.append(signal)
        return signals

    # ──────────────────────────────────────────
    #  Variant-aware scoring helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _apply_variant_scoring(
        features: Dict[str, float],
        base_scoring_config: Dict[str, Any],
        scoring_overrides: Dict[str, Any],
    ) -> List[FeatureScore]:
        """
        Merge variant scoring overrides with base scoring config
        and re-score the features.
        """
        merged = {}
        for feat_name, param_dict in base_scoring_config.items():
            merged_params = dict(param_dict) if isinstance(param_dict, dict) else {}
            if feat_name in scoring_overrides:
                override = scoring_overrides[feat_name]
                if isinstance(override, dict):
                    merged_params.update(override)
            merged[feat_name] = merged_params

        # Also add any entirely new features from overrides
        for feat_name, override in scoring_overrides.items():
            if feat_name not in merged and isinstance(override, dict):
                merged[feat_name] = override

        scores: List[FeatureScore] = []
        for feat_name, param_dict in merged.items():
            raw_value = features.get(feat_name)
            if raw_value is None:
                continue

            param = ScoringParameter(**param_dict)
            score = compute_feature_score(raw_value, param)
            scores.append(FeatureScore(
                feature_name=feat_name,
                raw_value=raw_value,
                normalized_score=score,
                weight=param.weight,
            ))

        return scores

    @staticmethod
    def _compute_regime_affinity_match(
        current_regime: RegimeFingerprint,
        variant_affinity: RegimeFingerprint,
    ) -> float:
        """
        Compute similarity between current market regime and variant's
        preferred regime using cosine similarity (0.0 to 1.0).
        """
        import numpy as np
        current_vec = np.array(current_regime.to_vector())
        affinity_vec = np.array(variant_affinity.to_vector())

        norm_c = np.linalg.norm(current_vec)
        norm_a = np.linalg.norm(affinity_vec)

        if norm_c == 0 or norm_a == 0:
            return 0.5  # no information → neutral

        similarity = float(np.dot(current_vec, affinity_vec) / (norm_c * norm_a))
        # Map from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    # ──────────────────────────────────────────
    #  Entry / Stop / Target computation
    # ──────────────────────────────────────────

    @staticmethod
    def _compute_entry_zone(
        sko: Dict[str, Any],
        candles: List[Dict[str, Any]],
        features: Dict[str, float],
    ) -> Optional[EntryZone]:
        """Compute entry zone from SKO config + current price."""
        last = candles[-1]
        close = last["close"]
        atr = features.get("atr_percentile", 0.5)

        # Use last bar's range as a proxy for zone width
        bar_range = last["high"] - last["low"]
        half_zone = bar_range * 0.5

        entry_zones_config = sko.get("entry_zones", [])
        if entry_zones_config:
            # Use configured zone (first matching)
            zone = entry_zones_config[0]
            return EntryZone(
                lower=zone.get("lower_bound", close - half_zone),
                upper=zone.get("upper_bound", close + half_zone),
                sweet_spot=zone.get("sweet_spot", close),
            )

        return EntryZone(
            lower=round(close - half_zone, 2),
            upper=round(close + half_zone, 2),
            sweet_spot=round(close, 2),
        )

    @staticmethod
    def _compute_stop(
        sko: Dict[str, Any],
        candles: List[Dict[str, Any]],
        features: Dict[str, float],
    ) -> Optional[float]:
        """Compute stop loss from exit rules + current ATR."""
        exit_rules = sko.get("exit_rules", {})
        atr_multiple = exit_rules.get("atr_multiple", 1.5)

        # Estimate ATR from recent candles
        if len(candles) < 15:
            return None

        trs = []
        for i in range(-14, 0):
            tr = max(
                candles[i]["high"] - candles[i]["low"],
                abs(candles[i]["high"] - candles[i - 1]["close"]),
                abs(candles[i]["low"] - candles[i - 1]["close"]),
            )
            trs.append(tr)

        atr = sum(trs) / len(trs)
        close = candles[-1]["close"]

        return round(close - atr * atr_multiple, 2)

    @staticmethod
    def _compute_targets(
        sko: Dict[str, Any],
        candles: List[Dict[str, Any]],
        features: Dict[str, float],
        stop_level: Optional[float],
    ) -> List[float]:
        """Compute target levels based on risk parameters."""
        close = candles[-1]["close"]
        if stop_level is None:
            return []

        risk = close - stop_level
        if risk <= 0:
            return []

        risk_params = sko.get("risk_parameters", {})
        rr_min = risk_params.get("rr_minimum", 1.5)

        # Generate targets at 1:1, RR minimum, and 2x RR minimum
        targets = [
            round(close + risk * 1.0, 2),
            round(close + risk * rr_min, 2),
            round(close + risk * rr_min * 1.5, 2),
        ]
        return targets
