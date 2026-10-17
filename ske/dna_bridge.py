# ske/dna_bridge.py
"""
DNA Bridge — converts a Strategy Knowledge Object (SKO) to a StrategyDNA
YAML representation, ensuring backward compatibility with all 16 coaching
agents and existing DNA consumers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class DNABridge:
    """
    Converts SKO fields to StrategyDNA-compatible YAML.
    """

    def sko_to_dna_dict(self, sko: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert SKO data to a StrategyDNA-compatible dict.

        Args:
            sko: Dict with all SKO fields.

        Returns:
            Dict matching StrategyDNA schema (suitable for YAML dump).
        """
        scoring_config = sko.get("scoring_config", {})
        exit_rules = sko.get("exit_rules", {})
        risk_params = sko.get("risk_parameters", {})
        regime_fp = sko.get("regime_fingerprint", {})
        time_prefs = sko.get("time_preferences", {})

        # Map scoring config → setup_conditions
        setup_conditions = []
        for feat_name, params in scoring_config.items():
            condition = {
                "type": feat_name,
                "required": params.get("weight", 1.0) >= 0.8,
                "weight": params.get("weight", 1.0),
                "description": f"Feature {feat_name}: centered at {params.get('center', 'N/A')}",
            }
            setup_conditions.append(condition)

        # Map exit_rules → DNA exit fields
        stop_config = {
            "type": exit_rules.get("stop_method", "atr"),
            "atr_multiplier": exit_rules.get("atr_multiple", 1.5),
        }

        trailing = exit_rules.get("trailing", {})
        trailing_config = {
            "enabled": trailing.get("enabled", False),
            "atr_multiplier": trailing.get("atr_multiple", 1.0),
        }

        # Target
        rr_min = risk_params.get("rr_minimum", 1.5)
        target_config = {
            "type": "rr_based",
            "rr_ratio": rr_min,
            "partial_exits": [
                {"at_rr": 1.0, "exit_pct": 0.33},
                {"at_rr": rr_min, "exit_pct": 0.50},
            ],
        }

        # Regime guidance
        preferred = regime_fp.get("preferred", [])
        avoid = regime_fp.get("avoid", [])
        regime_guidance = {}
        if preferred:
            regime_guidance["preferred_regimes"] = [
                p.get("regime", "trending_up") if isinstance(p, dict) else str(p)
                for p in preferred
            ]
        if avoid:
            regime_guidance["avoid_regimes"] = [
                a.get("regime", "unknown") if isinstance(a, dict) else str(a)
                for a in avoid
            ]

        # Session guidance
        session_guidance = {}
        if time_prefs:
            best_sessions = sorted(
                time_prefs.items(),
                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                reverse=True,
            )
            session_guidance["preferred_sessions"] = [s[0] for s in best_sessions[:3]]

        dna = {
            "strategy_id": sko.get("short_code", "sko_unknown"),
            "name": sko.get("name", "Unnamed SKO Strategy"),
            "version": "1.0",
            "description": sko.get("description", sko.get("narrative", "")),
            "source": "ske_auto_generated",
            "setup_conditions": setup_conditions,
            "stop_loss": stop_config,
            "trailing_stop": trailing_config,
            "target": target_config,
            "risk": {
                "max_risk_pct": risk_params.get("base_risk_pct", 1.0),
                "max_position_pct": risk_params.get("max_risk_pct", 2.0),
            },
            "regime_guidance": regime_guidance,
            "session_guidance": session_guidance,
            "metadata": {
                "tier": sko.get("status", "hypothesis"),
                "total_trades": sko.get("total_trades", 0),
                "win_rate": sko.get("win_rate"),
                "expectancy": sko.get("expectancy"),
                "dqal_grade": sko.get("dqal_grade"),
                "tags": sko.get("tags", []),
            },
        }

        return dna

    def sko_to_yaml(self, sko: Dict[str, Any]) -> str:
        """Convert SKO to YAML string."""
        dna_dict = self.sko_to_dna_dict(sko)
        return yaml.dump(dna_dict, default_flow_style=False, sort_keys=False)

    def save_dna_yaml(
        self,
        sko: Dict[str, Any],
        output_dir: str = "config/strategies",
    ) -> str:
        """
        Save SKO as a StrategyDNA YAML file.

        Returns:
            Path to the saved YAML file.
        """
        import os

        short_code = sko.get("short_code", "sko_unknown")
        yaml_content = self.sko_to_yaml(sko)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{short_code}.yaml")

        with open(path, "w") as f:
            f.write(yaml_content)

        logger.info("Saved StrategyDNA YAML: %s", path)
        return path
