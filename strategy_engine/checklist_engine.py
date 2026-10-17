# strategy_engine/checklist_engine.py
"""
Checklist-based strategy engine.
This is the PRACTICAL replacement for full algorithmic matching.
Works with Level 0 and Level 1 strategies.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
from strategy_engine.quick_loader import QuickStrategy, QuickStrategyLoader
from core.enums import TradeDirection, SessionType
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChecklistResult:
    """Result of a trader completing the pre-trade checklist."""
    strategy_name: str
    total_conditions: int
    conditions_met: int
    conditions_skipped: int
    conditions_failed: int
    checklist_score: float               # 0-100
    responses: Dict[str, str]            # condition → "met" | "skipped" | "failed"
    is_valid_setup: bool                 # enough conditions met?
    warnings: List[str]


@dataclass
class SimpleAttribution:
    """
    Trade attribution result that works at ALL levels.
    No complex ML or pattern matching needed.
    """
    # ── Tier 1: What user declared ──
    declared_strategy: str
    declared_strategy_display: str

    # ── Tier 1.5: Checklist result ──
    checklist_result: Optional[ChecklistResult]
    checklist_score: float                # 0-100

    # ── Tier 2: Simple auto-validation ──
    auto_validation_passed: bool
    auto_validation_notes: List[str]
    auto_confidence: float                # 0-1

    # ── Tier 3: Simple override detection ──
    overrides_detected: List[str]
    execution_type: str                   # "disciplined", "minor_deviation", "override"

    # ── Combined ──
    overall_discipline_score: float       # 0-100
    shadow_rules: Dict                    # rules for shadow trade management


class ChecklistEngine:
    """
    Practical engine that works with checklists instead of full algo code.

    This is what you use for Phase 1 testing.
    Gives you 80% of the intelligence with 20% of the effort.
    """

    def __init__(self, strategy_loader: QuickStrategyLoader):
        self.loader = strategy_loader
        self.strategies = strategy_loader.get_all()

    def get_checklist(self, strategy_name: str) -> Optional[List[str]]:
        """Get checklist for a strategy (for frontend display)."""
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return None
        return strategy.checklist

    def process_trade_entry(
        self,
        strategy_name: str,
        checklist_responses: Dict[str, str],   # {condition: "met"|"skipped"|"failed"}
        entry_time: datetime,
        entry_price: float,
        stop_loss: Optional[float],
        direction: TradeDirection,
        pre_emotion: Optional[str] = None,
        was_planned: Optional[bool] = None,
    ) -> SimpleAttribution:
        """
        Process a trade entry and return full attribution.
        This is called when user submits the trade entry form.
        """
        strategy = self.strategies.get(strategy_name)

        # ── Tier 1: User declared ──
        declared = strategy_name
        display = strategy.display_name if strategy else strategy_name

        # ── Tier 1.5: Checklist scoring ──
        checklist_result = None
        checklist_score = 100.0  # default if no checklist

        if strategy and strategy.checklist and checklist_responses:
            checklist_result = self._score_checklist(strategy, checklist_responses)
            checklist_score = checklist_result.checklist_score

        # ── Tier 2: Simple auto-validation ──
        validation_passed, validation_notes = self._simple_auto_validate(
            strategy, entry_time, entry_price, stop_loss, direction
        )

        # ── Tier 3: Simple override detection ──
        overrides = self._detect_simple_overrides(
            strategy, checklist_result, entry_time, stop_loss,
            entry_price, direction, pre_emotion, was_planned
        )

        # ── Execution classification ──
        if not overrides and checklist_score >= 80:
            execution_type = "disciplined"
        elif len(overrides) <= 1 and checklist_score >= 60:
            execution_type = "minor_deviation"
        else:
            execution_type = "override"

        # ── Overall discipline score ──
        discipline = self._calculate_discipline_score(
            checklist_score, validation_passed, overrides, was_planned
        )

        # ── Shadow trade rules ──
        shadow_rules = self._get_shadow_rules(strategy, stop_loss, entry_price, direction)

        # Auto-validation confidence
        auto_confidence = 1.0 if validation_passed else 0.5
        if not strategy:
            auto_confidence = 0.0

        return SimpleAttribution(
            declared_strategy=declared,
            declared_strategy_display=display,
            checklist_result=checklist_result,
            checklist_score=checklist_score,
            auto_validation_passed=validation_passed,
            auto_validation_notes=validation_notes,
            auto_confidence=auto_confidence,
            overrides_detected=overrides,
            execution_type=execution_type,
            overall_discipline_score=discipline,
            shadow_rules=shadow_rules,
        )

    def _score_checklist(
        self,
        strategy: QuickStrategy,
        responses: Dict[str, str]
    ) -> ChecklistResult:
        """Score the completed checklist."""
        total = len(strategy.checklist)
        met = 0
        skipped = 0
        failed = 0
        warnings = []

        # Map responses by index or condition text
        for i, condition in enumerate(strategy.checklist):
            # responses can be keyed by index or condition text
            key = str(i) if str(i) in responses else condition
            response = responses.get(key, "skipped")

            if response == "met":
                met += 1
            elif response == "failed":
                failed += 1
                warnings.append(f"⚠ Failed: {condition}")
            else:
                skipped += 1
                warnings.append(f"⏭ Skipped: {condition}")

        # Score: met conditions weighted more, skipped neutral, failed negative
        if total > 0:
            score = ((met * 1.0) + (skipped * 0.5)) / total * 100
        else:
            score = 100

        is_valid = met >= (total * 0.6)  # at least 60% of conditions met

        return ChecklistResult(
            strategy_name=strategy.name,
            total_conditions=total,
            conditions_met=met,
            conditions_skipped=skipped,
            conditions_failed=failed,
            checklist_score=round(score, 1),
            responses=responses,
            is_valid_setup=is_valid,
            warnings=warnings,
        )

    def _simple_auto_validate(
        self,
        strategy: Optional[QuickStrategy],
        entry_time: datetime,
        entry_price: float,
        stop_loss: Optional[float],
        direction: TradeDirection,
    ) -> Tuple[bool, List[str]]:
        """
        Simple validation — no complex indicators needed.
        Just checks what we CAN check with basic trade data.
        """
        notes = []
        issues = 0

        if not strategy:
            return False, ["No strategy selected"]

        # ── Time window check ──
        if strategy.time_window_start and strategy.time_window_end:
            entry_t = entry_time.time() if isinstance(entry_time, datetime) else entry_time
            start = time.fromisoformat(strategy.time_window_start)
            end = time.fromisoformat(strategy.time_window_end)

            if start <= entry_t <= end:
                notes.append(f"✅ Time OK: {entry_t.strftime('%H:%M')} within {strategy.time_window_start}-{strategy.time_window_end}")
            else:
                notes.append(f"⚠️ Time OUTSIDE window: {entry_t.strftime('%H:%M')} vs {strategy.time_window_start}-{strategy.time_window_end}")
                issues += 1

        # ── Stop loss check ──
        if stop_loss is None:
            notes.append("🚨 No stop loss defined!")
            issues += 1
        else:
            # Check stop is on correct side
            if direction == TradeDirection.LONG and stop_loss >= entry_price:
                notes.append("🚨 Stop loss ABOVE entry for long trade!")
                issues += 1
            elif direction == TradeDirection.SHORT and stop_loss <= entry_price:
                notes.append("🚨 Stop loss BELOW entry for short trade!")
                issues += 1
            else:
                notes.append("✅ Stop loss correctly placed")

        # ── R:R check ──
        if stop_loss and stop_loss != entry_price:
            risk = abs(entry_price - stop_loss)
            min_target = risk * strategy.default_r_target
            notes.append(f"ℹ️ Risk per share: ${risk:.2f}, min target ({strategy.default_r_target}R): ${min_target:.2f}")

        passed = issues == 0
        return passed, notes

    def _detect_simple_overrides(
        self,
        strategy: Optional[QuickStrategy],
        checklist_result: Optional[ChecklistResult],
        entry_time: datetime,
        stop_loss: Optional[float],
        entry_price: float,
        direction: TradeDirection,
        pre_emotion: Optional[str],
        was_planned: Optional[bool],
    ) -> List[str]:
        """Detect overrides using simple heuristics."""
        overrides = []

        # ── Unplanned trade ──
        if was_planned is False:
            overrides.append("unplanned_trade")

        # ── No strategy selected or discretionary ──
        if not strategy or strategy.name in ("discretionary", "unplanned"):
            overrides.append("no_strategy_framework")

        # ── Failed checklist conditions ──
        if checklist_result and checklist_result.conditions_failed > 0:
            overrides.append(
                f"checklist_failures:{checklist_result.conditions_failed}"
            )

        # ── Low checklist score ──
        if checklist_result and checklist_result.checklist_score < 50:
            overrides.append("low_checklist_score")

        # ── Time window violation ──
        if strategy and strategy.time_window_start:
            entry_t = entry_time.time()
            start = time.fromisoformat(strategy.time_window_start)
            end = time.fromisoformat(strategy.time_window_end) if strategy.time_window_end else time(16, 0)
            if not (start <= entry_t <= end):
                overrides.append("outside_time_window")

        # ── Emotional state concerns ──
        risky_emotions = {"revenge", "fomo", "frustrated", "euphoric", "anxious"}
        if pre_emotion and pre_emotion.lower() in risky_emotions:
            overrides.append(f"risky_emotion:{pre_emotion}")

        # ── No stop loss ──
        if stop_loss is None:
            overrides.append("no_stop_loss")

        return overrides

    def _calculate_discipline_score(
        self,
        checklist_score: float,
        validation_passed: bool,
        overrides: List[str],
        was_planned: Optional[bool],
    ) -> float:
        """Calculate overall discipline score 0-100."""
        score = 0

        # Checklist contributes 40%
        score += checklist_score * 0.4

        # Auto-validation contributes 20%
        score += 20 if validation_passed else 5

        # Override penalty: -8 points per override (max 40)
        override_penalty = min(len(overrides) * 8, 40)
        score += (40 - override_penalty)

        # Planned trade bonus
        if was_planned is True:
            score = min(100, score + 5)
        elif was_planned is False:
            score = max(0, score - 10)

        return round(max(0, min(100, score)), 1)

    def _get_shadow_rules(
        self,
        strategy: Optional[QuickStrategy],
        stop_loss: Optional[float],
        entry_price: float,
        direction: TradeDirection,
    ) -> Dict:
        """
        Get rules for shadow trade management.
        These tell the shadow engine HOW to manage the parallel trade.
        """
        if not strategy:
            return {
                "stop_type": "fixed",
                "r_target": 2.0,
                "trailing_enabled": False,
            }

        rules = {
            "stop_type": strategy.default_stop_type,
            "r_target": strategy.default_r_target,
            "trailing_enabled": strategy.trailing_stop_enabled,
            "trailing_activation_r": strategy.trailing_activation_r,
        }

        # Calculate target price
        if stop_loss and stop_loss != entry_price:
            risk = abs(entry_price - stop_loss)
            if direction == TradeDirection.LONG:
                rules["target_price"] = entry_price + (risk * strategy.default_r_target)
            else:
                rules["target_price"] = entry_price - (risk * strategy.default_r_target)

            # Partial exit
            if strategy.partial_exit_at_r:
                if direction == TradeDirection.LONG:
                    rules["partial_exit_price"] = entry_price + (risk * strategy.partial_exit_at_r)
                else:
                    rules["partial_exit_price"] = entry_price - (risk * strategy.partial_exit_at_r)
                rules["partial_exit_percent"] = 50  # exit 50% at first target

        return rules