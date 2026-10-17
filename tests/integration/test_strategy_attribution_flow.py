# tests/integration/test_strategy_attribution_flow.py
"""Tests for the strategy attribution pipeline."""

import pytest
from core.enums import AttributionTier, ConditionStatus, OverrideType


class TestAttributionTiers:
    def test_tier_values(self):
        assert AttributionTier.TIER1_DECLARED.value == "tier1_declared"
        assert AttributionTier.TIER2_AUTO_MATCHED.value == "tier2_auto_matched"
        assert AttributionTier.TIER3_OVERRIDE.value == "tier3_override"


class TestConditionStatus:
    def test_all_statuses(self):
        assert ConditionStatus.MET.value == "met"
        assert ConditionStatus.NOT_MET.value == "not_met"
        assert ConditionStatus.PARTIALLY_MET.value == "partially_met"
        assert ConditionStatus.INCONCLUSIVE.value == "inconclusive"
        assert ConditionStatus.ERROR.value == "error"


class TestStrategyRegistry:
    def test_registry_loads_strategies(self, registry):
        ids = registry.list_ids()
        assert "test_breakout" in ids

    def test_registry_get_strategy(self, registry):
        strategy = registry.get("test_breakout")
        assert strategy is not None

    def test_registry_get_nonexistent(self, registry):
        strategy = registry.get("nonexistent_strategy")
        assert strategy is None

    def test_registry_count(self, registry):
        count = len(registry.list_ids())
        assert count >= 1


class TestAttributionWithRegistry:
    """The registry parses YAML into a typed, frozen StrategyTemplate (not a dict)."""

    def test_strategy_has_entry_conditions(self, registry):
        strategy = registry.get("test_breakout")
        assert strategy is not None
        assert strategy.direction == "long"
        types = [c.type for c in strategy.entry_confirmations]
        assert types == ["price_breakout", "volume_surge", "ema_alignment"]
        assert strategy.entry_confirmations[0].required is True

    def test_strategy_has_risk_params(self, registry):
        risk = registry.get("test_breakout").risk_rules
        assert risk.max_risk_pct > 0

    def test_strategy_has_exit_rules(self, registry):
        exit_rules = registry.get("test_breakout").exit_rules
        assert exit_rules.stop_loss.type == "structure"
        assert [t.r_multiple for t in exit_rules.targets] == [1.0, 2.0]
        assert sum(t.size_pct for t in exit_rules.targets) == 100
