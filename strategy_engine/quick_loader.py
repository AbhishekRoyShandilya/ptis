"""
strategy_engine/quick_loader.py
Lightweight YAML-based strategy loader for checklist engine.
Wraps the quick-format YAML files in config/strategies/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QuickStrategy:
    """Lightweight strategy definition used by the checklist engine."""

    name: str
    display_name: str = ""
    description: str = ""

    # Checklist items (trader answers met/skipped/failed for each)
    checklist: List[str] = field(default_factory=list)

    # Time window (HH:MM strings)
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None

    # Simple validation flags
    requires_volume_spike: bool = False
    requires_trend: bool = False
    typical_hold_minutes: int = 60

    # Default risk/reward parameters
    default_r_target: float = 2.0
    default_stop_type: str = "atr"
    typical_session: str = "any"
    tags: List[str] = field(default_factory=list)

    # Shadow trade management
    trailing_stop_enabled: bool = False
    trailing_activation_r: float = 1.5
    partial_exit_at_r: float = 2.0

    @classmethod
    def from_dict(cls, data: dict) -> "QuickStrategy":
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            description=data.get("description", ""),
            checklist=data.get("checklist", []),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            requires_volume_spike=data.get("requires_volume_spike", False),
            requires_trend=data.get("requires_trend", False),
            typical_hold_minutes=data.get("typical_hold_minutes", 60),
            default_r_target=data.get("default_r_target", 2.0),
            default_stop_type=data.get("default_stop_type", "atr"),
            typical_session=data.get("typical_session", "any"),
            tags=data.get("tags", []),
            trailing_stop_enabled=data.get("trailing_stop_enabled", False),
            trailing_activation_r=data.get("trailing_activation_r", 1.5),
            partial_exit_at_r=data.get("partial_exit_at_r", 2.0),
        )


class QuickStrategyLoader:
    """
    Loads QuickStrategy objects from a YAML file.

    Expected YAML format:
        strategies:
          - name: opening_range_breakout
            display_name: "Opening Range Breakout"
            checklist:
              - "Condition 1"
              - "Condition 2"
            ...
    """

    def __init__(self, yaml_path: str = "config/strategies/strategies_quick.yaml"):
        self._path = Path(yaml_path)
        self._strategies: Dict[str, QuickStrategy] = {}
        self._loaded = False

    def load(self) -> "QuickStrategyLoader":
        """Load strategies from YAML. Returns self for chaining."""
        if not self._path.exists():
            logger.warning("quick_loader: %s not found — starting empty", self._path)
            self._loaded = True
            return self

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            for entry in data.get("strategies", []):
                qs = QuickStrategy.from_dict(entry)
                self._strategies[qs.name] = qs

            logger.info("quick_loader: loaded %d strategies from %s", len(self._strategies), self._path)
        except Exception as exc:
            logger.error("quick_loader: failed to load %s — %s", self._path, exc)

        self._loaded = True
        return self

    def get(self, name: str) -> Optional[QuickStrategy]:
        if not self._loaded:
            self.load()
        return self._strategies.get(name)

    def get_all(self) -> Dict[str, QuickStrategy]:
        if not self._loaded:
            self.load()
        return dict(self._strategies)

    def names(self) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self._strategies.keys())

    def __len__(self) -> int:
        return len(self._strategies)
