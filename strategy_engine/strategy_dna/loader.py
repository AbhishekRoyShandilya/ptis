# strategy_engine/strategy_dna/loader.py
"""
Loads Strategy DNA YAML files into structured objects.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from strategy_engine.strategy_dna.schema import (
    StrategyDNA, SetupCondition, EmotionalTrap,
    RegimeGuidance, MarketConditionProfile,
    StrategyRelationships, PersonalInsights
)
import logging

logger = logging.getLogger(__name__)


class StrategyDNALoader:
    """Load all Strategy DNA files and make them queryable."""

    def __init__(self, strategy_dir: str = "config/strategies"):
        self.strategy_dir = Path(strategy_dir)
        self._strategies: Dict[str, StrategyDNA] = {}

    def load_all(self) -> Dict[str, StrategyDNA]:
        """Load all strategy YAML files."""
        if not self.strategy_dir.exists():
            logger.warning(f"Strategy directory not found: {self.strategy_dir}")
            return {}

        for path in self.strategy_dir.glob("*.yaml"):
            if path.name.startswith("_"):
                continue  # skip schema/template files

            # Pre-check: must be a flat dict with a top-level 'name' key.
            # Files like strategies_quick.yaml use a 'strategies: [...]' list
            # format for other purposes — skip them silently.
            try:
                import yaml as _yaml
                with open(path, encoding="utf-8") as _f:
                    _raw = _yaml.safe_load(_f)
                if not isinstance(_raw, dict) or "name" not in _raw:
                    logger.debug(f"Skipping {path.name} — not a DNA strategy file")
                    continue
            except Exception:
                pass  # let _parse_file produce the proper error below

            try:
                dna = self._parse_file(path)
                self._strategies[dna.name] = dna
                logger.info(
                    f"Loaded Strategy DNA: {dna.display_name} "
                    f"({len(dna.setup_conditions)} conditions, "
                    f"{len(dna.emotional_traps)} traps)"
                )
            except Exception as e:
                logger.error(f"Failed to load {path.name}: {e}", exc_info=True)

        logger.info(f"Loaded {len(self._strategies)} strategy DNA profiles")
        return self._strategies

    def get(self, name: str) -> Optional[StrategyDNA]:
        if not self._strategies:
            self.load_all()
        return self._strategies.get(name)

    def get_all(self) -> Dict[str, StrategyDNA]:
        if not self._strategies:
            self.load_all()
        return self._strategies

    def get_names(self) -> List[Dict]:
        """Get strategy names for UI."""
        if not self._strategies:
            self.load_all()
        return [
            {
                "name": s.name,
                "display_name": s.display_name,
                "short_code": s.short_code,
                "category": s.category,
                "description": s.description[:100],
                "checklist_count": len(s.checklist),
                "tags": s.tags,
            }
            for s in self._strategies.values()
        ]

    def _parse_file(self, path: Path) -> StrategyDNA:
        """Parse a single YAML file into StrategyDNA."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        entry = raw.get("entry_logic", {})
        execution = raw.get("execution", {})
        risk = raw.get("risk", {})
        exit_rules = raw.get("exit_rules", {})
        mkt = raw.get("market_conditions", {})
        psych = raw.get("psychology", {})
        rels = raw.get("relationships", {})
        insights = raw.get("personal_insights", {})

        # Parse setup conditions
        conditions = []
        for c in entry.get("setup_conditions", []):
            conditions.append(SetupCondition(
                name=c["name"],
                description=c.get("description", ""),
                importance=c.get("importance", "preferred"),
                what_good_looks_like=c.get("what_good_looks_like", ""),
                what_bad_looks_like=c.get("what_bad_looks_like", ""),
            ))

        # Parse emotional traps
        traps = []
        for t in psych.get("common_emotional_traps", []):
            traps.append(EmotionalTrap(
                trap=t["trap"],
                description=t.get("description", ""),
                solution=t.get("solution", ""),
                frequency=t.get("frequency", "sometimes"),
            ))

        # Parse market conditions
        best_regimes = [
            RegimeGuidance(
                regime=r["name"],
                notes=r.get("notes", ""),
                expected_win_rate=r.get("expected_win_rate"),
            )
            for r in mkt.get("works_best_in", {}).get("regimes", [])
        ]
        worst_regimes = [
            RegimeGuidance(
                regime=r["name"],
                notes=r.get("notes", ""),
                expected_win_rate=r.get("expected_win_rate"),
                recommendation=r.get("recommendation"),
            )
            for r in mkt.get("works_poorly_in", {}).get("regimes", [])
        ]

        market_profile = MarketConditionProfile(
            works_best_regimes=best_regimes,
            works_poorly_regimes=worst_regimes,
            works_best_sessions=mkt.get("works_best_in", {}).get("sessions", []),
            works_poorly_sessions=mkt.get("works_poorly_in", {}).get("sessions", []),
            works_best_additional=mkt.get("works_best_in", {}).get("additional_conditions", []),
            works_poorly_additional=mkt.get("works_poorly_in", {}).get("additional_conditions", []),
            regime_sizing=mkt.get("regime_sizing", {}),
        )

        strategy_rels = StrategyRelationships(
            fallback_strategies=rels.get("fallback_strategies", []),
            conflicts_with=rels.get("conflicts_with", []),
            pairs_well_with=rels.get("pairs_well_with", []),
            regime_alternatives=rels.get("regime_alternatives", {}),
        )

        personal = PersonalInsights(
            strengths=insights.get("strengths", []),
            weaknesses=insights.get("weaknesses", []),
            edge_conditions=insights.get("edge_conditions", []),
            learned_statistics=insights.get("learned_statistics", {}),
        )

        sl = risk.get("stop_loss", {})
        ps = risk.get("position_sizing", {})
        tgt = risk.get("target", {})

        return StrategyDNA(
            name=raw["name"],
            display_name=raw.get("display_name", raw["name"]),
            short_code=raw.get("short_code", raw["name"][:3].upper()),
            version=raw.get("version", 1),
            category=raw.get("category", "unknown"),
            timeframe_primary=raw.get("timeframe_primary", "5m"),
            timeframe_confirmation=raw.get("timeframe_confirmation"),
            typical_hold=raw.get("typical_hold", "unknown"),
            description=raw.get("description", ""),
            entry_narrative=entry.get("narrative", ""),
            setup_conditions=conditions,
            a_plus_conditions=entry.get("a_plus_conditions", []),
            skip_conditions=entry.get("skip_conditions", []),
            entry_trigger=execution.get("entry_trigger", ""),
            entry_timing=execution.get("entry_timing", ""),
            common_entry_mistakes=execution.get("common_mistakes_i_make", []),
            ideal_entry_description=execution.get("ideal_entry_description", ""),
            stop_method=sl.get("primary_method", ""),
            stop_description=sl.get("description", ""),
            stop_common_mistakes=sl.get("common_mistakes", []),
            sizing_method=ps.get("method", "risk_based"),
            risk_per_trade=ps.get("risk_per_trade", "1%"),
            max_risk=ps.get("max_risk_per_trade", "1.5%"),
            size_adjustments=ps.get("size_adjustments", {}),
            target_primary=tgt.get("primary", "2R"),
            target_secondary=tgt.get("secondary", ""),
            target_method=tgt.get("method", ""),
            profit_taking_rules=exit_rules.get("profit_taking", []),
            time_exit_rules=exit_rules.get("time_based_exit", []),
            common_exit_mistakes=exit_rules.get("common_exit_mistakes", []),
            market_conditions=market_profile,
            emotional_traps=traps,
            ideal_mental_state=psych.get("ideal_mental_state", []),
            do_not_trade_when=psych.get("do_not_trade_when", []),
            relationships=strategy_rels,
            personal_insights=personal,
            checklist=raw.get("checklist", []),
            tags=raw.get("tags", []),
        )