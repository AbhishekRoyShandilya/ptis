# ske/builder.py
"""
SKO Builder — constructs or updates a Strategy Knowledge Object from trade history.

Build pipeline:
1. Compute feature vectors for each trade's entry candles
2. Estimate scoring parameters by fitting distributions to winning trades
3. Compute conviction map from outcome distribution
4. Discover variants via clustering
5. Run DQAL validation
6. Determine tier from trade count + DQAL score
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from core.enums import ConvictionBand, SKOStatus, ScoringFunctionType
from ske.dqal import DQALValidator
from ske.feature_computer import FeatureComputer
from ske.models import ScoringParameter, VariantInfo
from ske.quality_gate import TradeQualityGate
from ske.tier_engine import TierProgressionEngine
from ske.variant_engine import VariantEngine

logger = logging.getLogger(__name__)


class SKOBuilder:
    """
    Builds and updates Strategy Knowledge Objects from trade data.
    """

    def __init__(
        self,
        feature_computer: Optional[FeatureComputer] = None,
        variant_engine: Optional[VariantEngine] = None,
        dqal_validator: Optional[DQALValidator] = None,
        quality_gate: Optional[TradeQualityGate] = None,
        tier_engine: Optional[TierProgressionEngine] = None,
        min_trades_developing: int = 10,
        min_trades_validated: int = 20,
        min_trades_proven: int = 50,
    ):
        self._feature_computer = feature_computer or FeatureComputer()
        self._variant_engine = variant_engine or VariantEngine()
        self._dqal = dqal_validator or DQALValidator(
            min_trades_developing, min_trades_validated, min_trades_proven,
        )
        self._quality_gate = quality_gate or TradeQualityGate()
        self._tier_engine = tier_engine or TierProgressionEngine()
        self._tier_thresholds = [
            (min_trades_proven, SKOStatus.PROVEN),
            (min_trades_validated, SKOStatus.VALIDATED),
            (min_trades_developing, SKOStatus.DEVELOPING),
            (0, SKOStatus.HYPOTHESIS),
        ]

    def _merge_variants(
        self,
        discovered: List[VariantInfo],
        existing: List[VariantInfo],
    ) -> List[VariantInfo]:
        """
        Merge discovered (system) variants with existing user-taught/hybrid variants.
        System variants are replaced by fresh discovery; user-taught and hybrid are preserved.
        """
        preserved = [v for v in existing if v.provenance in ("user_taught", "hybrid")]
        if preserved:
            logger.info(
                "Preserving %d user-taught/hybrid variants during rebuild",
                len(preserved),
            )
        return list(discovered) + preserved

    def build_from_trades(
        self,
        trades: List[Dict[str, Any]],
        name: str,
        short_code: str,
        description: str = "",
        candles_by_trade: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        apply_quality_gate: bool = True,
        sko_context: Optional[Dict[str, Any]] = None,
        existing_variants: Optional[List[VariantInfo]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete SKO from a list of trades.

        Args:
            trades: List of trade dicts with at least:
                - symbol, entry_price, exit_price, direction, r_multiple (or pnl)
                - Optional: stop_loss, take_profit, entry_time
                - Optional (for quality gate): discipline_score, decision_quality_score,
                  journal_quality_score, emotion_pre, override_count
            name: SKO display name.
            short_code: Unique short code.
            description: Strategy description.
            candles_by_trade: Optional pre-fetched candles keyed by trade ID.
            apply_quality_gate: Whether to run the quality gate pipeline.
            sko_context: Context for alignment checking (variant_centroids, etc.)

        Returns:
            Dict with all SKO fields ready for DB insertion.
        """
        if not trades:
            return self._empty_sko(name, short_code, description)

        # Extract outcomes (R-multiples)
        outcomes = []
        for t in trades:
            r = t.get("r_multiple")
            if r is None:
                entry = t.get("entry_price", 0)
                exit_p = t.get("exit_price", 0)
                stop = t.get("stop_loss")
                if entry and exit_p:
                    if stop and stop != entry:
                        risk = abs(entry - stop)
                        r = (exit_p - entry) / risk if t.get("direction", "long") == "long" else (entry - exit_p) / risk
                    else:
                        r = (exit_p - entry) / entry * 100 if entry > 0 else 0.0
                else:
                    r = 0.0
            outcomes.append(float(r))

        # Compute feature vectors
        feature_vectors = []
        if candles_by_trade:
            for t in trades:
                trade_id = str(t.get("id", ""))
                candles = candles_by_trade.get(trade_id, [])
                if candles:
                    fv = self._feature_computer.compute_features(candles, t.get("symbol", ""))
                    feature_vectors.append(fv)
                else:
                    feature_vectors.append({})
        else:
            # Use trade-level features as proxy
            for t in trades:
                feature_vectors.append(self._trade_to_feature_proxy(t))

        # ── Quality Gate (if enabled) ──
        learning_weights = [1.0] * len(trades)  # default: all trades weighted equally
        gate_results = []

        if apply_quality_gate:
            # Attach feature vectors to trades for alignment checking
            for t, fv in zip(trades, feature_vectors):
                if fv and "feature_vector" not in t:
                    t["feature_vector"] = fv

            gate_results = self._quality_gate.evaluate_batch(trades, sko_context)
            learning_weights = [gr.learning_weight for gr in gate_results]

        # Use weighted stats for filtered model
        n = len(outcomes)
        wins_weighted = sum(w for o, w in zip(outcomes, learning_weights) if o > 0 and w > 0)
        total_weighted = sum(w for w in learning_weights if w > 0)
        win_rate = wins_weighted / total_weighted if total_weighted > 0 else 0.0

        weighted_exp = (
            sum(o * w for o, w in zip(outcomes, learning_weights) if w > 0) / total_weighted
            if total_weighted > 0 else 0.0
        )

        # Profit factor (weighted)
        weighted_profit = sum(o * w for o, w in zip(outcomes, learning_weights) if o > 0 and w > 0)
        weighted_loss = abs(sum(o * w for o, w in zip(outcomes, learning_weights) if o <= 0 and w > 0))
        profit_factor = weighted_profit / weighted_loss if weighted_loss > 0 else (
            float("inf") if weighted_profit > 0 else 0.0
        )

        # Estimate scoring parameters from winning trades (weighted)
        scoring_config = self._estimate_scoring_params(feature_vectors, outcomes)

        # Conviction map
        conviction_map = self._compute_conviction_map(outcomes)

        # Discover variants (pass learning weights for weighted clustering)
        valid_fvs = [fv for fv in feature_vectors if fv]
        valid_outcomes = outcomes[:len(valid_fvs)]
        valid_weights = learning_weights[:len(valid_fvs)]
        variants = (
            self._variant_engine.discover_variants(
                valid_fvs, valid_outcomes, learning_weights=valid_weights,
            )
            if len(valid_fvs) >= 5 else []
        )

        # Preserve user-taught/hybrid variants from previous build
        if existing_variants:
            variants = self._merge_variants(variants, existing_variants)

        # Exit rules from trade data
        exit_rules = self._estimate_exit_rules(trades)

        # Risk parameters
        risk_params = self._estimate_risk_params(trades, outcomes)

        # Tier via effective sample size
        tier_eval = self._tier_engine.evaluate(
            learning_weights=learning_weights,
            dqal_score=0.0,  # DQAL not yet computed
            win_rate=win_rate,
        )

        # DQAL (with effective sample size)
        sko_data = {
            "name": name,
            "short_code": short_code,
            "scoring_config": scoring_config,
            "conviction_map": conviction_map,
            "exit_rules": exit_rules,
            "risk_parameters": risk_params,
            "total_trades": n,
            "status": tier_eval.eligible_tier,
            "effective_sample_size": tier_eval.effective_count,
        }
        dqal_result = self._dqal.validate(sko_data, outcomes)

        # Re-evaluate tier with DQAL score
        tier_eval = self._tier_engine.evaluate(
            learning_weights=learning_weights,
            dqal_score=dqal_result.overall_score,
            win_rate=win_rate,
        )

        result = {
            "name": name,
            "short_code": short_code,
            "description": description,
            "status": tier_eval.eligible_tier,
            "tier": n,
            "total_trades": n,
            "win_rate": round(win_rate, 4),
            "expectancy": round(weighted_exp, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 99.0,
            "scoring_config": scoring_config,
            "conviction_map": conviction_map,
            "entry_zones": [],
            "exit_rules": exit_rules,
            "risk_parameters": risk_params,
            "regime_fingerprint": {},
            "time_preferences": {},
            "dqal_score": dqal_result.overall_score,
            "dqal_grade": dqal_result.grade,
            "tags": [],
            "variants": [v.model_dump() for v in variants],
            "effective_sample_size": tier_eval.effective_count,
            "learning_weights": learning_weights,
            "gate_results": [gr.model_dump() for gr in gate_results] if gate_results else [],
        }

        return result

    def update_from_trade(
        self,
        sko: Dict[str, Any],
        trade: Dict[str, Any],
        candles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally update an SKO with a new trade.
        Returns updated fields (not a full SKO — caller merges).
        """
        r = trade.get("r_multiple", 0.0)
        old_n = sko.get("total_trades", 0)
        new_n = old_n + 1

        # Running win rate
        old_wr = sko.get("win_rate", 0.0)
        new_wins = old_wr * old_n + (1 if r > 0 else 0)
        new_win_rate = new_wins / new_n

        # Running expectancy
        old_exp = sko.get("expectancy", 0.0)
        new_exp = (old_exp * old_n + r) / new_n

        # Tier check
        dqal_score = sko.get("dqal_score", 0.0)
        new_tier = self._compute_tier(new_n, dqal_score)

        updates = {
            "total_trades": new_n,
            "tier": new_n,
            "win_rate": round(new_win_rate, 4),
            "expectancy": round(new_exp, 4),
            "status": new_tier.value,
        }

        # Check for tier promotion
        old_status = sko.get("status", SKOStatus.HYPOTHESIS.value)
        if new_tier.value != old_status:
            logger.info(
                "SKO '%s' promoted: %s → %s",
                sko.get("short_code"), old_status, new_tier.value,
            )

        return updates

    # ──────────────────────────────────────────
    #  Scoring Parameter Estimation
    # ──────────────────────────────────────────

    def _estimate_scoring_params(
        self,
        feature_vectors: List[Dict[str, float]],
        outcomes: List[float],
    ) -> Dict[str, Any]:
        """
        Fit Gaussian scoring parameters from winning-trade feature distributions.
        For each feature, center = mean of winning trades, spread = std.
        """
        if not feature_vectors or not any(fv for fv in feature_vectors):
            return {}

        # Collect all feature names
        all_features: set = set()
        for fv in feature_vectors:
            all_features.update(fv.keys())

        # Split into winners/losers
        winners = []
        for fv, r in zip(feature_vectors, outcomes):
            if r > 0 and fv:
                winners.append(fv)

        if len(winners) < 3:
            # Not enough winners — use all trades
            winners = [fv for fv in feature_vectors if fv]

        if not winners:
            return {}

        config: Dict[str, Any] = {}
        for feat in sorted(all_features):
            values = [fv.get(feat) for fv in winners if feat in fv]
            values = [v for v in values if v is not None]
            if len(values) < 2:
                continue

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 0.1

            config[feat] = {
                "type": ScoringFunctionType.GAUSSIAN.value,
                "center": round(mean, 6),
                "spread": round(max(std, 0.01), 6),
                "weight": 1.0,
                "invert": False,
            }

        return config

    def _compute_conviction_map(self, outcomes: List[float]) -> Dict[str, List[float]]:
        """Default conviction map — can be refined with outcome distribution."""
        return {
            "skip": [0.0, 0.30],
            "small": [0.30, 0.50],
            "standard": [0.50, 0.70],
            "aggressive": [0.70, 0.85],
            "max": [0.85, 1.0],
        }

    def _estimate_exit_rules(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate exit rules from trade stop/target patterns."""
        atr_multiples = []
        for t in trades:
            entry = t.get("entry_price", 0)
            stop = t.get("stop_loss")
            if entry and stop and entry > 0:
                risk_pct = abs(entry - stop) / entry
                # Rough ATR multiple estimate (assume ATR ~ 1-2% of price)
                atr_mult = risk_pct / 0.015 if risk_pct > 0 else 1.5
                atr_multiples.append(atr_mult)

        avg_atr_mult = sum(atr_multiples) / len(atr_multiples) if atr_multiples else 1.5

        return {
            "stop_method": "atr",
            "atr_multiple": round(avg_atr_mult, 2),
            "trailing": {"enabled": False, "atr_multiple": round(avg_atr_mult * 0.75, 2)},
        }

    def _estimate_risk_params(
        self, trades: List[Dict[str, Any]], outcomes: List[float],
    ) -> Dict[str, Any]:
        """Estimate risk parameters from trade data."""
        risk_pcts = []
        for t in trades:
            rp = t.get("risk_pct")
            if rp and rp > 0:
                risk_pcts.append(rp)

        base_risk = sum(risk_pcts) / len(risk_pcts) if risk_pcts else 1.0

        # RR from outcomes
        wins = [r for r in outcomes if r > 0]
        avg_r = sum(wins) / len(wins) if wins else 1.5

        return {
            "base_risk_pct": round(base_risk, 2),
            "max_risk_pct": round(base_risk * 2, 2),
            "rr_minimum": round(max(1.0, avg_r * 0.5), 2),
        }

    def _compute_tier(self, total_trades: int, dqal_score: float) -> SKOStatus:
        """Determine SKO status from trade count and DQAL score."""
        for threshold, status in self._tier_thresholds:
            if total_trades >= threshold:
                # Proven requires DQAL >= 60
                if status == SKOStatus.PROVEN and dqal_score < 60:
                    return SKOStatus.VALIDATED
                return status
        return SKOStatus.HYPOTHESIS

    @staticmethod
    def _empty_sko(name: str, short_code: str, description: str) -> Dict[str, Any]:
        """Return an empty/skeleton SKO."""
        return {
            "name": name,
            "short_code": short_code,
            "description": description,
            "status": SKOStatus.HYPOTHESIS.value,
            "tier": 0,
            "total_trades": 0,
            "win_rate": None,
            "expectancy": None,
            "profit_factor": None,
            "scoring_config": {},
            "conviction_map": {
                "skip": [0.0, 0.30],
                "small": [0.30, 0.50],
                "standard": [0.50, 0.70],
                "aggressive": [0.70, 0.85],
                "max": [0.85, 1.0],
            },
            "entry_zones": [],
            "exit_rules": {},
            "risk_parameters": {"base_risk_pct": 1.0, "max_risk_pct": 2.0, "rr_minimum": 1.5},
            "regime_fingerprint": {},
            "time_preferences": {},
            "dqal_score": None,
            "dqal_grade": None,
            "tags": [],
            "variants": [],
        }

    @staticmethod
    def _trade_to_feature_proxy(trade: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract proxy features directly from trade metadata when candle data
        isn't available. Limited but better than empty.
        """
        features: Dict[str, float] = {}

        entry = trade.get("entry_price", 0)
        exit_p = trade.get("exit_price", 0)
        stop = trade.get("stop_loss")

        if entry and exit_p:
            # Price change as momentum proxy
            features["momentum_5"] = (exit_p - entry) / entry if entry > 0 else 0.0

        if entry and stop:
            risk_pct = abs(entry - stop) / entry if entry > 0 else 0.0
            features["atr_percentile"] = min(risk_pct / 0.03, 1.0)  # normalize to ~3% max

        conviction = trade.get("conviction_score")
        if conviction is not None:
            features["conviction_proxy"] = conviction / 10.0  # normalize 1-10 to 0-1

        return features
