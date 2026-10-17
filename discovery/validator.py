# discovery/validator.py
"""
Stage 7 — Validator + Knowledge Storer

Final stage of the pattern discovery pipeline:
  1. Validates the generated strategy YAML against StrategyDNA schema
  2. Presents the discovery report for human review (accept/reject)
  3. On acceptance: writes YAML to config/strategies/, reloads StrategyDNALoader
  4. Indexes the new strategy in the RAG knowledge base
  5. Publishes STRATEGY_DISCOVERED event so agents become aware
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.events import Event, EventType, get_event_bus
from discovery.models import DiscoveryReport, DiscoveryStatus, GeneratedStrategy

logger = logging.getLogger(__name__)

# Path to strategy YAML directory
_STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "config" / "strategies"


class StrategyValidator:
    """
    Validates a generated strategy and manages human review workflow.

    Usage:
        validator = StrategyValidator()
        issues = validator.validate(generated_strategy)
        if not issues:
            validator.accept(report)
    """

    # Required top-level fields in strategy YAML
    REQUIRED_FIELDS = [
        "name", "display_name", "short_code", "version", "category",
        "timeframe_primary", "description",
    ]

    # Required nested sections
    REQUIRED_SECTIONS = [
        "entry_logic", "risk_management", "market_conditions",
    ]

    def validate(self, strategy: GeneratedStrategy) -> List[str]:
        """
        Validate a generated strategy.
        Returns list of issues (empty = valid).
        """
        issues: List[str] = []

        if not strategy.yaml_dict:
            try:
                strategy.yaml_dict = yaml.safe_load(strategy.yaml_content)
            except yaml.YAMLError as e:
                return [f"Invalid YAML: {e}"]

        data = strategy.yaml_dict

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data or not data[field]:
                issues.append(f"Missing required field: {field}")

        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in data:
                issues.append(f"Missing required section: {section}")

        # Validate name format
        name = data.get("name", "")
        if name and not name.replace("_", "").isalnum():
            issues.append(f"Strategy name must be alphanumeric + underscores: '{name}'")

        # Check for duplicate names
        existing_yamls = list(_STRATEGIES_DIR.glob("*.yaml"))
        for yf in existing_yamls:
            if yf.stem == name:
                issues.append(f"Strategy name '{name}' already exists at {yf}")

        # Validate category
        valid_categories = {"momentum", "mean_reversion", "breakout", "trend", "scalp", "swing", "catalyst"}
        cat = data.get("category", "")
        if cat and cat not in valid_categories:
            issues.append(f"Invalid category '{cat}'. Must be one of: {valid_categories}")

        # Validate entry_logic has conditions
        entry = data.get("entry_logic", {})
        if isinstance(entry, dict):
            conditions = entry.get("setup_conditions", [])
            if not conditions:
                issues.append("entry_logic.setup_conditions is empty — need at least 2 conditions")

        # Validate risk_management
        risk = data.get("risk_management", {})
        if isinstance(risk, dict):
            if not risk.get("stop_loss") and not risk.get("stop_description"):
                issues.append("risk_management must define a stop loss method")

        # Validate market_conditions has regime sizing
        mc = data.get("market_conditions", {})
        if isinstance(mc, dict):
            if not mc.get("regime_sizing"):
                issues.append("market_conditions.regime_sizing is missing")

        return issues

    def accept(
        self,
        report: DiscoveryReport,
        reviewer_notes: str = "",
    ) -> str:
        """
        Accept the discovered strategy:
          1. Write YAML to config/strategies/
          2. Publish event
          3. Update report status

        Returns the file path of the saved YAML.
        """
        if not report.generated_strategy:
            raise ValueError("No generated strategy to accept")

        strategy = report.generated_strategy
        name = strategy.strategy_name

        # Write YAML file
        filepath = _STRATEGIES_DIR / f"{name}.yaml"
        _STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

        # Add discovery metadata header
        header = (
            f"# {filepath.name}\n"
            f"# ═══════════════════════════════════════════════════════\n"
            f"# STRATEGY DNA: {strategy.display_name}\n"
            f"# Discovered by Pattern Discovery Engine\n"
            f"# Discovery Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"# Discovery ID: {report.discovery_id}\n"
            f"# Confidence: {strategy.confidence:.0%}\n"
            f"# Based on: {report.total_matches} historical matches\n"
            f"# Win Rate: {report.overall_win_rate:.0%} | Avg R: {report.overall_avg_r:.1f}\n"
            f"# ═══════════════════════════════════════════════════════\n\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(strategy.yaml_content)

        logger.info("Strategy YAML saved to %s", filepath)

        # Update report
        report.status = DiscoveryStatus.ACCEPTED
        report.accepted = True
        report.reviewer_notes = reviewer_notes
        report.stored_strategy_id = name

        # Publish discovery event
        bus = get_event_bus()
        bus.publish(Event(
            event_type=EventType.STRATEGY_DISCOVERED,
            source="discovery.validator",
            data={
                "discovery_id": str(report.discovery_id),
                "strategy_name": name,
                "display_name": strategy.display_name,
                "file_path": str(filepath),
                "total_matches": report.total_matches,
                "win_rate": report.overall_win_rate,
                "avg_r": report.overall_avg_r,
                "confidence": strategy.confidence,
                "reviewer_notes": reviewer_notes,
            },
        ))

        return str(filepath)

    def reject(self, report: DiscoveryReport, reason: str = "") -> None:
        """Reject the discovered strategy."""
        report.status = DiscoveryStatus.REJECTED
        report.accepted = False
        report.reviewer_notes = reason
        logger.info("Strategy rejected: %s (reason: %s)",
                     report.generated_strategy.strategy_name if report.generated_strategy else "N/A",
                     reason)

    async def index_in_rag(self, report: DiscoveryReport, db: Any = None) -> None:
        """
        Index the accepted strategy in the RAG knowledge base so
        coaching agents become aware of it.
        """
        if not report.generated_strategy or not report.accepted:
            return

        strategy = report.generated_strategy

        try:
            from rag.indexer import KnowledgeIndexer

            indexer = KnowledgeIndexer()

            # Create knowledge entry from the strategy
            content = (
                f"Strategy: {strategy.display_name}\n"
                f"Category: {strategy.category}\n"
                f"Discovered from pattern matching: {report.total_matches} instances\n"
                f"Win Rate: {report.overall_win_rate:.0%}\n"
                f"Average R: {report.overall_avg_r:.1f}\n"
                f"Best Regime: {report.correlation.best_regime if report.correlation else 'N/A'}\n"
                f"Best Time: {report.correlation.best_time_window if report.correlation else 'N/A'}\n\n"
                f"Entry Logic:\n{strategy.yaml_dict.get('entry_logic', {}).get('narrative', 'N/A')}\n\n"
                f"Skip Conditions:\n"
            )
            skip = strategy.yaml_dict.get("entry_logic", {}).get("skip_conditions", [])
            for sc in skip:
                content += f"  - {sc}\n"

            await indexer.index_text(
                text=content,
                category="strategy",
                title=f"Discovered: {strategy.display_name}",
                tags=strategy.yaml_dict.get("tags", ["discovered"]),
                source="pattern_discovery",
                db=db,
            )
            logger.info("Strategy indexed in RAG: %s", strategy.strategy_name)
        except Exception as e:
            logger.warning("RAG indexing failed (non-critical): %s", e)
