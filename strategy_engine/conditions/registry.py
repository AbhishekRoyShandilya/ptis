# strategy_engine/conditions/registry.py
"""
Global condition evaluator registry.
Conditions self-register using the @register_condition decorator.
"""

from __future__ import annotations

import logging
from typing import Any

from strategy_engine.conditions.base_condition import (
    BaseCondition,
    ConditionResult,
    ConditionStatus,
    MarketSnapshot,
    TradeContext,
)

logger = logging.getLogger(__name__)

# Global registry
_condition_evaluators: dict[str, BaseCondition] = {}


def register_condition(cls: type[BaseCondition]) -> type[BaseCondition]:
    """Decorator to register a condition evaluator class."""
    instance = cls()
    ctype = instance.condition_type
    if ctype in _condition_evaluators:
        logger.warning("Overwriting condition evaluator for type '%s'", ctype)
    _condition_evaluators[ctype] = instance
    logger.debug("Registered condition evaluator: %s", ctype)
    return cls


def condition_registry() -> dict[str, BaseCondition]:
    return dict(_condition_evaluators)


def evaluate_condition(
    condition_type: str,
    params: dict[str, Any],
    trade: TradeContext,
    market: MarketSnapshot,
    required: bool = True,
    weight: float = 1.0,
) -> ConditionResult:
    """
    Evaluate a single condition by type name.
    Returns INCONCLUSIVE if evaluator not found.
    """
    evaluator = _condition_evaluators.get(condition_type)
    if not evaluator:
        logger.warning("No evaluator registered for condition type '%s'", condition_type)
        return ConditionResult(
            condition_type=condition_type,
            status=ConditionStatus.INCONCLUSIVE,
            details=f"No evaluator registered for '{condition_type}'",
            confidence=0.0,
            required=required,
            weight=weight,
        )

    try:
        result = evaluator.evaluate(params, trade, market)
        result.required = required
        result.weight = weight
        result.condition_type = condition_type
        return result
    except Exception as e:
        logger.error("Error evaluating condition '%s': %s", condition_type, e, exc_info=True)
        return ConditionResult(
            condition_type=condition_type,
            status=ConditionStatus.ERROR,
            details=str(e),
            error=str(e),
            required=required,
            weight=weight,
        )


class ConditionRegistry:
    """
    Class-based facade over the module-level condition registry.
    Allows callers to use ConditionRegistry() as an injectable object.
    """

    def get(self, condition_type: str) -> Any:
        return _condition_evaluators.get(condition_type)

    def all(self) -> dict[str, BaseCondition]:
        return condition_registry()

    def evaluate(
        self,
        condition_type: str,
        params: dict[str, Any],
        trade: TradeContext,
        market: MarketSnapshot,
        required: bool = True,
        weight: float = 1.0,
    ) -> ConditionResult:
        return evaluate_condition(condition_type, params, trade, market, required, weight)