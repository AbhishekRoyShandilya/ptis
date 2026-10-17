# strategy_engine/templates/loader.py
"""
Parses YAML strategy templates into strongly-typed Python objects.
Validates structure and provides query interface for conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from core.enums import MarketRegime, TradeDirection

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class ConditionDef:
    """A single entry/avoid/invalidation condition from YAML."""
    type: str
    params: dict[str, Any]
    required: bool = True
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class TimeWindow:
    start: str              # "09:30"
    end: str                # "16:00"
    timezone: str = "Asia/Kolkata"


@dataclass(frozen=True)
class StopLossRule:
    type: str               # fixed, atr, structure, percentage
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetLevel:
    r_multiple: float
    size_pct: float


@dataclass(frozen=True)
class TrailingRule:
    enabled: bool = False
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    activate_at_r: float = 1.0


@dataclass(frozen=True)
class TimeExit:
    enabled: bool = False
    max_duration_minutes: Optional[int] = None
    close_before: Optional[str] = None


@dataclass(frozen=True)
class ExitRules:
    stop_loss: StopLossRule
    targets: list[TargetLevel]
    trailing: TrailingRule
    time_exit: TimeExit


@dataclass(frozen=True)
class RiskRules:
    max_risk_pct: float = 1.0
    max_risk_amount: Optional[float] = None
    max_position_size: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_daily_loss_pct: Optional[float] = None
    min_r_target: float = 1.5
    max_correlated_positions: Optional[int] = None


@dataclass(frozen=True)
class ExecutionPrefs:
    entry_type: str = "market"
    preferred_timeframe: str = "5m"
    notes: str = ""


@dataclass
class StrategyTemplate:
    """Fully parsed strategy template ready for matching and override detection."""
    name: str
    id: str
    version: int
    description: str
    author: str
    active: bool

    # Regime
    preferred_regimes: list[MarketRegime]
    avoid_regimes: list[MarketRegime]
    neutral_regimes: list[MarketRegime]

    # Entry
    direction: str                         # "long", "short", "any"
    entry_confirmations: list[ConditionDef]
    avoid_conditions: list[ConditionDef]
    time_window: Optional[TimeWindow]

    # Exit
    exit_rules: ExitRules

    # Risk
    risk_rules: RiskRules

    # Invalidation
    invalidation_conditions: list[ConditionDef]

    # Execution
    execution: ExecutionPrefs

    # ── Derived helpers ──

    @property
    def required_confirmations(self) -> list[ConditionDef]:
        return [c for c in self.entry_confirmations if c.required]

    @property
    def supporting_confirmations(self) -> list[ConditionDef]:
        return [c for c in self.entry_confirmations if not c.required]

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.entry_confirmations)

    @property
    def required_weight(self) -> float:
        return sum(c.weight for c in self.required_confirmations)

    def is_regime_compatible(self, regime: MarketRegime) -> bool:
        if regime in self.avoid_regimes:
            return False
        return True

    def is_regime_preferred(self, regime: MarketRegime) -> bool:
        return regime in self.preferred_regimes

    def direction_compatible(self, trade_direction: TradeDirection) -> bool:
        if self.direction == "any":
            return True
        if self.direction == "long" and trade_direction == TradeDirection.LONG:
            return True
        if self.direction == "short" and trade_direction == TradeDirection.SHORT:
            return True
        return False


# ══════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════

def _parse_regimes(names: list[str] | None) -> list[MarketRegime]:
    if not names:
        return []
    mapping = {r.value: r for r in MarketRegime}
    result = []
    for n in names:
        regime = mapping.get(n)
        if regime:
            result.append(regime)
        else:
            logger.warning("Unknown regime '%s' in template — skipping", n)
    return result


def _parse_condition(raw: dict[str, Any]) -> ConditionDef:
    return ConditionDef(
        type=raw["type"],
        params=raw.get("params", {}),
        required=raw.get("required", True),
        weight=raw.get("weight", 1.0),
        description=raw.get("description", ""),
    )


def _parse_template(raw: dict[str, Any]) -> StrategyTemplate:
    s = raw["strategy"]
    entry = s.get("entry", {})
    exit_ = s.get("exit", {})
    risk = s.get("risk", {})
    regime = s.get("market_regime", {})
    execution = s.get("execution", {})

    # Time window
    tw_raw = entry.get("time_window")
    time_window = TimeWindow(**tw_raw) if tw_raw else None

    # Stop loss
    sl_raw = exit_.get("stop_loss", {"type": "fixed", "params": {}})
    stop_loss = StopLossRule(type=sl_raw["type"], params=sl_raw.get("params", {}))

    # Targets
    targets = [TargetLevel(**t) for t in exit_.get("targets", [])]

    # Trailing
    tr_raw = exit_.get("trailing", {})
    trailing = TrailingRule(
        enabled=tr_raw.get("enabled", False),
        type=tr_raw.get("type", ""),
        params=tr_raw.get("params", {}),
        activate_at_r=tr_raw.get("activate_at_r", 1.0),
    )

    # Time exit
    te_raw = exit_.get("time_exit", {})
    time_exit = TimeExit(
        enabled=te_raw.get("enabled", False),
        max_duration_minutes=te_raw.get("max_duration_minutes"),
        close_before=te_raw.get("close_before"),
    )

    return StrategyTemplate(
        name=s["name"],
        id=s["id"],
        version=s.get("version", 1),
        description=s.get("description", ""),
        author=s.get("author", "trader"),
        active=s.get("active", True),
        preferred_regimes=_parse_regimes(regime.get("preferred")),
        avoid_regimes=_parse_regimes(regime.get("avoid")),
        neutral_regimes=_parse_regimes(regime.get("neutral")),
        direction=entry.get("direction", "any"),
        entry_confirmations=[_parse_condition(c) for c in entry.get("confirmations", [])],
        avoid_conditions=[_parse_condition(c) for c in entry.get("avoid_conditions", [])],
        time_window=time_window,
        exit_rules=ExitRules(
            stop_loss=stop_loss,
            targets=targets,
            trailing=trailing,
            time_exit=time_exit,
        ),
        risk_rules=RiskRules(
            max_risk_pct=risk.get("max_risk_pct", 1.0),
            max_risk_amount=risk.get("max_risk_amount"),
            max_position_size=risk.get("max_position_size"),
            max_daily_trades=risk.get("max_daily_trades"),
            max_daily_loss_pct=risk.get("max_daily_loss_pct"),
            min_r_target=risk.get("min_r_target", 1.5),
            max_correlated_positions=risk.get("max_correlated_positions"),
        ),
        invalidation_conditions=[_parse_condition(c) for c in s.get("invalidation", [])],
        execution=ExecutionPrefs(
            entry_type=execution.get("entry_type", "market"),
            preferred_timeframe=execution.get("preferred_timeframe", "5m"),
            notes=execution.get("notes", ""),
        ),
    )


# ══════════════════════════════════════════════
#  TEMPLATE LOADER
# ══════════════════════════════════════════════

# strategy_engine/templates/loader.py (continued from TemplateLoader)

class TemplateLoader:
    """
    Loads all YAML strategy templates from a directory.
    Validates, parses, and provides access by ID.
    """

    def __init__(self, templates_dir: str | Path | None = None):
        self._templates_dir = Path(templates_dir) if templates_dir else Path("config/strategies")
        self._templates: dict[str, StrategyTemplate] = {}
        self._load_errors: list[dict[str, Any]] = []

    @property
    def templates(self) -> dict[str, StrategyTemplate]:
        return dict(self._templates)

    @property
    def active_templates(self) -> dict[str, StrategyTemplate]:
        return {k: v for k, v in self._templates.items() if v.active}

    @property
    def load_errors(self) -> list[dict[str, Any]]:
        return list(self._load_errors)

    def load_all(self) -> int:
        """Load all YAML files from templates directory. Returns count loaded."""
        self._templates.clear()
        self._load_errors.clear()

        if not self._templates_dir.exists():
            logger.error("Templates directory does not exist: %s", self._templates_dir)
            return 0

        # Load from root dir + library/ subfolder
        yaml_files = sorted(self._templates_dir.glob("*.yaml"))
        yaml_files += sorted((self._templates_dir / "library").glob("*.yaml")) if (self._templates_dir / "library").exists() else []
        loaded = 0

        for filepath in yaml_files:
            if filepath.name.startswith("_"):
                logger.debug("Skipping template file: %s", filepath.name)
                continue

            try:
                # Peek at the YAML to see if it's template-format (has 'strategy:' key).
                # DNA-format files (starting with 'name:') live in the same directory
                # and are handled by StrategyDNALoader — skip them silently here.
                with open(filepath, "r", encoding="utf-8") as _f:
                    _peek = yaml.safe_load(_f)
                if not _peek or "strategy" not in _peek:
                    logger.debug("Skipping non-template file: %s", filepath.name)
                    continue

                # Skip built-in "library" templates unless explicitly enabled.
                # Keeps the system purely user-driven — the registry (and every
                # engine reading it: intelligence, co-planner, behavioral,
                # setup scanner) stays empty until the user teaches a strategy.
                if (_peek.get("strategy") or {}).get("is_library"):
                    try:
                        from config.settings import settings as _settings
                        _allow_library = _settings.LOAD_LIBRARY_STRATEGY_TEMPLATES
                    except Exception:
                        _allow_library = False
                    if not _allow_library:
                        logger.debug("Skipping library template: %s", filepath.name)
                        continue

                template = self.load_file(filepath)
                if template.id in self._templates:
                    logger.warning(
                        "Duplicate strategy ID '%s' — overwriting from %s",
                        template.id, filepath.name,
                    )
                self._templates[template.id] = template
                loaded += 1
                logger.info("Loaded strategy template: %s (%s)", template.name, template.id)
            except Exception as e:
                self._load_errors.append({
                    "file": str(filepath),
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                logger.error("Failed to load template %s: %s", filepath.name, e)

        logger.info(
            "Template loading complete: %d loaded, %d errors",
            loaded, len(self._load_errors),
        )
        return loaded

    def load_file(self, filepath: str | Path) -> StrategyTemplate:
        """Load and parse a single YAML template file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "strategy" not in raw:
            raise ValueError(f"Invalid template format in {filepath.name}: missing 'strategy' key")

        template = _parse_template(raw)
        self._validate_template(template, filepath.name)
        return template

    def load_from_dict(self, raw: dict[str, Any]) -> StrategyTemplate:
        """Parse a template from a raw dictionary (e.g., from DB or API)."""
        template = _parse_template(raw)
        self._validate_template(template, "<dict>")
        self._templates[template.id] = template
        return template

    def get(self, strategy_id: str) -> StrategyTemplate | None:
        return self._templates.get(strategy_id)

    def get_by_regime(self, regime: MarketRegime) -> list[StrategyTemplate]:
        """Return templates compatible with given regime, preferred first."""
        preferred = []
        compatible = []
        for t in self._templates.values():
            if not t.active:
                continue
            if t.is_regime_preferred(regime):
                preferred.append(t)
            elif t.is_regime_compatible(regime):
                compatible.append(t)
        return preferred + compatible

    def get_by_direction(self, direction: TradeDirection) -> list[StrategyTemplate]:
        return [
            t for t in self._templates.values()
            if t.active and t.direction_compatible(direction)
        ]

    def get_candidates(
        self,
        direction: TradeDirection,
        regime: MarketRegime | None = None,
    ) -> list[StrategyTemplate]:
        """Return active templates matching direction and regime."""
        candidates = []
        for t in self._templates.values():
            if not t.active:
                continue
            if not t.direction_compatible(direction):
                continue
            if regime and not t.is_regime_compatible(regime):
                continue
            candidates.append(t)
        return candidates

    def _validate_template(self, template: StrategyTemplate, source: str) -> None:
        errors = []
        if not template.id:
            errors.append("missing 'id'")
        if not template.name:
            errors.append("missing 'name'")
        if not template.entry_confirmations:
            errors.append("no entry confirmations defined")
        if not any(c.required for c in template.entry_confirmations):
            errors.append("no required entry confirmations — at least one needed")
        if not template.exit_rules.targets and not template.exit_rules.trailing.enabled:
            errors.append("no exit targets or trailing rule defined")
        if template.risk_rules.max_risk_pct <= 0:
            errors.append("max_risk_pct must be > 0")

        if errors:
            raise ValueError(f"Template validation failed ({source}): {'; '.join(errors)}")

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary list of all loaded templates."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "version": t.version,
                "active": t.active,
                "direction": t.direction,
                "required_confirmations": len(t.required_confirmations),
                "total_confirmations": len(t.entry_confirmations),
                "avoid_conditions": len(t.avoid_conditions),
                "preferred_regimes": [r.value for r in t.preferred_regimes],
            }
            for t in self._templates.values()
        ]