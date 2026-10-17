# improvement_lab/explainer.py
"""
Explanation layer for the Improvement Lab.

Wraps SHAP for tree-based models (RF, GBM) and generates
pseudo-explanations for rule-based detections. Every improvement
candidate gets an explanation so the trader understands WHY.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_shap():
    """Lazy-import shap to avoid hard dependency."""
    try:
        import shap
        return shap
    except ImportError:
        logger.info("shap not installed — using fallback feature importance")
        return None


class ImprovementExplainer:
    """
    Generates human-readable explanations for improvement detections.

    - Tree models → SHAP TreeExplainer
    - Fallback → model.feature_importances_ (sklearn built-in)
    - Rule-based → pseudo-SHAP from conditional stats
    """

    # ── Tree model explanations ──────────────────────────────────

    def explain_tree_model(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Explain a trained tree model using SHAP or fallback.

        Returns:
            {
                "method": "shap" | "feature_importances",
                "global_importance": {feature: importance},
                "top_features": [(feature, importance, direction)],
                "interaction_effects": [...],
            }
        """
        shap = _try_import_shap()

        if shap is not None:
            return self._shap_explain(shap, model, X, feature_names, top_k)
        else:
            return self._fallback_explain(model, feature_names, top_k)

    def _shap_explain(
        self,
        shap_module: Any,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        """Full SHAP explanation with TreeExplainer."""
        try:
            explainer = shap_module.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # For binary classification, shap_values may be a list [neg, pos]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class

            # Global importance: mean absolute SHAP
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            global_imp = {
                feature_names[i]: round(float(mean_abs[i]), 4)
                for i in range(len(feature_names))
            }

            # Top features sorted by importance
            sorted_idx = np.argsort(mean_abs)[::-1][:top_k]
            top_features = []
            for idx in sorted_idx:
                mean_signed = float(np.mean(shap_values[:, idx]))
                top_features.append({
                    "feature": feature_names[idx],
                    "importance": round(float(mean_abs[idx]), 4),
                    "direction": "positive" if mean_signed > 0 else "negative",
                    "mean_shap": round(mean_signed, 4),
                })

            # Interaction effects (top 3 pairs)
            interactions = self._find_interactions(
                shap_values, feature_names, top_k=3,
            )

            return {
                "method": "shap",
                "global_importance": global_imp,
                "top_features": top_features,
                "interaction_effects": interactions,
            }
        except Exception as exc:
            logger.warning("SHAP failed, using fallback: %s", exc)
            return self._fallback_explain(model, feature_names, top_k)

    def _fallback_explain(
        self,
        model: Any,
        feature_names: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        """Fallback using sklearn's feature_importances_."""
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return {
                "method": "unavailable",
                "global_importance": {},
                "top_features": [],
                "interaction_effects": [],
            }

        global_imp = {
            feature_names[i]: round(float(importances[i]), 4)
            for i in range(len(feature_names))
        }

        sorted_idx = np.argsort(importances)[::-1][:top_k]
        top_features = [
            {
                "feature": feature_names[idx],
                "importance": round(float(importances[idx]), 4),
                "direction": "unknown",  # can't determine without SHAP
                "mean_shap": None,
            }
            for idx in sorted_idx
        ]

        return {
            "method": "feature_importances",
            "global_importance": global_imp,
            "top_features": top_features,
            "interaction_effects": [],
        }

    @staticmethod
    def _find_interactions(
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find top feature interaction pairs via SHAP correlation."""
        n_features = shap_values.shape[1]
        if n_features < 2:
            return []

        interactions = []
        for i in range(min(n_features, 20)):
            for j in range(i + 1, min(n_features, 20)):
                # Correlation between SHAP values = interaction strength
                corr = float(np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1])
                if not np.isnan(corr) and abs(corr) > 0.3:
                    interactions.append({
                        "feature_a": feature_names[i],
                        "feature_b": feature_names[j],
                        "correlation": round(corr, 3),
                        "type": "amplifying" if corr > 0 else "dampening",
                    })

        interactions.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return interactions[:top_k]

    # ── Rule-based explanations ──────────────────────────────────

    def explain_rule_based(
        self,
        condition: Dict[str, Any],
        segment_stats: Dict[str, Any],
        baseline_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate pseudo-SHAP explanation for rule-based detection.

        Computes impact as the difference between segment and baseline
        performance, attributed to each condition axis.
        """
        baseline_wr = baseline_stats.get("win_rate", 0.5)
        segment_wr = segment_stats.get("win_rate", 0.5)
        impact = segment_wr - baseline_wr  # negative = bad segment

        top_features = []
        for key, value in condition.items():
            top_features.append({
                "feature": key,
                "importance": round(abs(impact), 4),
                "direction": "negative" if impact < 0 else "positive",
                "value": value,
                "mean_shap": round(impact / max(len(condition), 1), 4),
            })

        return {
            "method": "rule_based",
            "global_importance": {k: round(abs(impact), 4) for k in condition},
            "top_features": top_features,
            "interaction_effects": [],
            "baseline_win_rate": round(baseline_wr, 3),
            "segment_win_rate": round(segment_wr, 3),
            "lift": round(impact, 3),
        }

    # ── Human-readable formatting ────────────────────────────────

    @staticmethod
    def format_for_display(explanation: Dict[str, Any]) -> str:
        """Convert explanation dict to human-readable text."""
        if not explanation or not explanation.get("top_features"):
            return "No explanation available."

        lines = []
        method = explanation.get("method", "unknown")

        if method == "shap":
            lines.append("Key factors (SHAP analysis):")
        elif method == "feature_importances":
            lines.append("Key factors (feature importance):")
        else:
            lines.append("Key factors:")

        for feat in explanation["top_features"][:5]:
            name = feat["feature"].replace("_", " ").title()
            imp = feat["importance"]
            direction = feat.get("direction", "")
            arrow = "↑" if direction == "positive" else "↓" if direction == "negative" else "•"
            value_str = f" = {feat['value']}" if "value" in feat else ""
            lines.append(f"  {arrow} {name}{value_str} (impact: {imp:.3f})")

        interactions = explanation.get("interaction_effects", [])
        if interactions:
            lines.append("")
            lines.append("Feature interactions:")
            for ix in interactions[:2]:
                a = ix["feature_a"].replace("_", " ").title()
                b = ix["feature_b"].replace("_", " ").title()
                lines.append(f"  {a} × {b} ({ix['type']}, r={ix['correlation']:.2f})")

        return "\n".join(lines)
