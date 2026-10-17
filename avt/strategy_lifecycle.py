# avt/strategy_lifecycle.py
"""
Strategy Lifecycle Manager
===========================

State machine that governs strategy lifecycle from ingestion
to autonomous trading. Wraps the Strategy Understanding Engine
(SUE) from strategy_quality/ as its core intelligence.

States:
  INGESTED → HEALTH_CHECK → [REJECTED | PROBATION | READY]
                                   ↓              ↓
                             CALIBRATING ← (optional)
                                   ↓
                             AUTONOMOUS → MONITORING → [DEGRADED → PAUSED]

The lifecycle manager is the SINGLE SOURCE OF TRUTH for:
  - Whether AVT can trade a strategy
  - What allocation multiplier to use
  - Whether calibration is needed
  - Whether edge is degrading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from avt.models import (
    LIFECYCLE_ALLOCATION,
    LifecycleInfo,
    StrategyGrade,
    StrategyLifecycleState,
)

logger = logging.getLogger(__name__)

# Probation evaluation threshold
PROBATION_TRADE_COUNT = 30
PROBATION_MIN_EXPECTANCY = 0.0
PROBATION_MIN_PF = 1.0


@dataclass
class HealthIssue:
    """A single issue found during health gate validation."""
    category: str       # "structural" | "coherence" | "statistical"
    severity: str       # "info" | "warning" | "critical"
    field: str          # e.g., "exit_rules.stop_loss"
    message: str
    suggestion: str = ""


@dataclass
class HealthGateResult:
    """Result of strategy health gate validation."""
    strategy_id: str = ""
    grade: str = StrategyGrade.REJECTED.value
    issues: List[HealthIssue] = field(default_factory=list)
    structural_score: float = 0.0
    coherence_score: float = 0.0
    statistical_score: float = 0.0
    overall_health_score: float = 0.0
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)


class StrategyHealthGate:
    """
    Pre-trading validation. Three check layers:
      1. Structural completeness (beyond YAML schema)
      2. Logical coherence (contradiction detection)
      3. Statistical validation (if trade history provided)

    Uses strategy_quality.StrategyUnderstandingEngine for deep
    checks when available.
    """

    def validate(
        self,
        strategy: Dict[str, Any],
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> HealthGateResult:
        """Run all validation layers on a strategy."""
        result = HealthGateResult(
            strategy_id=str(strategy.get("id", "")),
        )

        # DNA-format strategies (have 'name' key but no 'exit.stop_loss' structure)
        # skip structural/coherence checks — they use a different schema
        is_dna_format = "name" in strategy and "exit" not in strategy
        if is_dna_format:
            result.structural_score = 75.0
            result.coherence_score = 75.0
            result.overall_health_score = 75.0
            result.grade = StrategyGrade.PROBATION.value  # cold-start → probation
            return result

        # Layer 1: Structural completeness
        structural_issues = self._check_structural(strategy)
        result.issues.extend(structural_issues)
        critical_count = sum(
            1 for i in structural_issues if i.severity == "critical"
        )
        total_struct = max(1, 6)  # 6 structural checks
        result.structural_score = round(
            (total_struct - critical_count) / total_struct * 100, 1
        )

        # Layer 2: Logical coherence
        coherence_issues = self._check_coherence(strategy)
        result.issues.extend(coherence_issues)
        coherence_critical = sum(
            1 for i in coherence_issues if i.severity == "critical"
        )
        total_coher = max(1, 5)  # 5 coherence checks
        result.coherence_score = round(
            (total_coher - coherence_critical) / total_coher * 100, 1
        )

        # Layer 3: Statistical validation (if history provided)
        if trade_history and len(trade_history) >= 20:
            stat_issues = self._check_statistical(trade_history)
            result.issues.extend(stat_issues)
            stat_critical = sum(
                1 for i in stat_issues if i.severity == "critical"
            )
            result.statistical_score = round(
                100.0 if stat_critical == 0 else 0.0, 1
            )
        else:
            result.statistical_score = -1.0  # not evaluated

        # Overall score
        if result.statistical_score >= 0:
            result.overall_health_score = round(
                result.structural_score * 0.35
                + result.coherence_score * 0.30
                + result.statistical_score * 0.35,
                1,
            )
        else:
            result.overall_health_score = round(
                result.structural_score * 0.55
                + result.coherence_score * 0.45,
                1,
            )

        # Grade
        if result.has_critical:
            result.grade = StrategyGrade.REJECTED.value
        elif result.statistical_score < 0:
            # No history — probation
            result.grade = StrategyGrade.PROBATION.value
        elif result.statistical_score == 0:
            # Stats show no edge
            result.grade = StrategyGrade.REJECTED.value
        else:
            result.grade = StrategyGrade.READY.value

        return result

    # ── Layer 1: Structural ────────────────────────────────────────

    def _check_structural(
        self, strategy: Dict[str, Any]
    ) -> List[HealthIssue]:
        issues = []

        # 1. Stop loss defined with params
        exit_rules = strategy.get("exit", strategy.get("exit_rules", {}))
        stop = exit_rules.get("stop_loss", {})
        if not stop or not stop.get("type"):
            issues.append(HealthIssue(
                category="structural",
                severity="critical",
                field="exit.stop_loss",
                message="No stop loss defined",
                suggestion="Add stop_loss with type (atr/fixed/structure/percentage) and params",
            ))
        elif stop.get("type") == "structure" and not stop.get("params"):
            issues.append(HealthIssue(
                category="structural",
                severity="warning",
                field="exit.stop_loss.params",
                message="Structure-based stop has no quantifiable params",
                suggestion="Add params like {buffer_atr: 0.2} so AVT can compute stop level",
            ))

        # 2. At least one profit target or trailing stop
        targets = exit_rules.get("targets", [])
        trailing = exit_rules.get("trailing", {})
        if not targets and not trailing.get("enabled"):
            issues.append(HealthIssue(
                category="structural",
                severity="critical",
                field="exit.targets",
                message="No profit targets and no trailing stop",
                suggestion="Add at least one R-multiple target or enable trailing",
            ))

        # 3. Risk params in sane range
        risk = strategy.get("risk", strategy.get("risk_rules", {}))
        max_risk = risk.get("max_risk_pct", 0)
        if max_risk <= 0:
            issues.append(HealthIssue(
                category="structural",
                severity="critical",
                field="risk.max_risk_pct",
                message="max_risk_pct is 0 or negative",
                suggestion="Set max_risk_pct between 0.25 and 3.0",
            ))
        elif max_risk > 3.0:
            issues.append(HealthIssue(
                category="structural",
                severity="warning",
                field="risk.max_risk_pct",
                message=f"max_risk_pct={max_risk}% is very high",
                suggestion="Consider capping at 2-3% per trade",
            ))

        # 4. min_r_target >= 1.0
        min_r = risk.get("min_r_target", 0)
        if min_r < 1.0 and min_r != 0:
            issues.append(HealthIssue(
                category="structural",
                severity="warning",
                field="risk.min_r_target",
                message=f"min_r_target={min_r} is below 1.0",
                suggestion="Target at least 1:1 risk-reward for positive expectancy",
            ))

        # 5. Entry confirmations
        entry = strategy.get("entry", {})
        confirmations = entry.get("confirmations", [])
        if len(confirmations) < 2:
            issues.append(HealthIssue(
                category="structural",
                severity="warning",
                field="entry.confirmations",
                message=f"Only {len(confirmations)} entry confirmations",
                suggestion="Add at least 2 confirmations (trend + trigger)",
            ))

        required_confs = [c for c in confirmations if c.get("required")]
        if confirmations and not required_confs:
            issues.append(HealthIssue(
                category="structural",
                severity="critical",
                field="entry.confirmations",
                message="No required confirmations — AVT will take weak setups",
                suggestion="Mark at least 1 confirmation as required: true",
            ))

        # 6. Time window (for intraday)
        time_window = entry.get("time_window", {})
        if not time_window.get("start") and not time_window.get("end"):
            issues.append(HealthIssue(
                category="structural",
                severity="info",
                field="entry.time_window",
                message="No time window defined",
                suggestion="Add start/end hours to prevent off-hours trading",
            ))

        return issues

    # ── Layer 2: Coherence ─────────────────────────────────────────

    def _check_coherence(
        self, strategy: Dict[str, Any]
    ) -> List[HealthIssue]:
        issues = []

        regime = strategy.get("market_regime", {})
        preferred = set(regime.get("preferred", []))
        avoid = set(regime.get("avoid", []))

        # 1. Regime contradiction
        overlap = preferred & avoid
        if overlap:
            issues.append(HealthIssue(
                category="coherence",
                severity="critical",
                field="market_regime",
                message=f"Regime in both preferred AND avoid: {overlap}",
                suggestion="Remove from one list",
            ))

        # 2. Direction + regime mismatch
        direction = strategy.get("entry", {}).get("direction", "any")
        if direction == "long" and "trending_down" in preferred:
            issues.append(HealthIssue(
                category="coherence",
                severity="warning",
                field="entry.direction + market_regime",
                message="Long-only strategy prefers trending_down",
                suggestion="Either allow short direction or remove trending_down from preferred",
            ))
        if direction == "short" and "trending_up" in preferred:
            issues.append(HealthIssue(
                category="coherence",
                severity="warning",
                field="entry.direction + market_regime",
                message="Short-only strategy prefers trending_up",
                suggestion="Either allow long direction or remove trending_up from preferred",
            ))

        # 3. Too many confirmations (over-fitted)
        confirmations = strategy.get("entry", {}).get("confirmations", [])
        if len(confirmations) > 8:
            issues.append(HealthIssue(
                category="coherence",
                severity="warning",
                field="entry.confirmations",
                message=f"{len(confirmations)} confirmations — likely over-fitted",
                suggestion="Reduce to 4-6 core confirmations. More isn't better.",
            ))

        # 4. Duplicate condition types
        types = [c.get("type", "") for c in confirmations]
        from collections import Counter
        dupes = {t: n for t, n in Counter(types).items() if n > 1 and t}
        if dupes:
            issues.append(HealthIssue(
                category="coherence",
                severity="warning",
                field="entry.confirmations",
                message=f"Duplicate confirmation types: {dupes}",
                suggestion="Merge into a single confirmation or differentiate params",
            ))

        # 5. Target achievability
        exit_rules = strategy.get("exit", strategy.get("exit_rules", {}))
        targets = exit_rules.get("targets", [])
        if targets:
            max_r = max(
                (t.get("r_multiple", 0) for t in targets), default=0
            )
            if max_r > 5.0:
                issues.append(HealthIssue(
                    category="coherence",
                    severity="warning",
                    field="exit.targets",
                    message=f"Max target is {max_r}R — rarely achievable",
                    suggestion="Consider capping primary targets at 3-4R",
                ))

        return issues

    # ── Layer 3: Statistical ───────────────────────────────────────

    def _check_statistical(
        self, trades: List[Dict[str, Any]]
    ) -> List[HealthIssue]:
        """Validate edge from trade history."""
        issues = []

        r_values = []
        for t in trades:
            r = t.get("r_multiple") or t.get("pnl_r")
            if r is not None:
                try:
                    r_values.append(float(r))
                except (ValueError, TypeError):
                    pass

        if len(r_values) < 20:
            issues.append(HealthIssue(
                category="statistical",
                severity="warning",
                field="trade_history",
                message=f"Only {len(r_values)} trades with R data (need 20+)",
                suggestion="Provide more trade history for reliable validation",
            ))
            return issues

        import numpy as np

        arr = np.array(r_values)
        expectancy = float(np.mean(arr))
        win_rate = float(np.mean(arr > 0))
        winners = arr[arr > 0]
        losers = arr[arr < 0]
        profit_factor = (
            float(np.sum(winners) / abs(np.sum(losers)))
            if len(losers) > 0 and np.sum(losers) != 0
            else 999.0
        )

        if expectancy <= 0:
            issues.append(HealthIssue(
                category="statistical",
                severity="critical",
                field="expectancy",
                message=f"Negative expectancy: {expectancy:.3f}R per trade",
                suggestion="Strategy has no statistical edge. Review entry/exit rules.",
            ))

        if profit_factor < 1.0:
            issues.append(HealthIssue(
                category="statistical",
                severity="critical",
                field="profit_factor",
                message=f"Profit factor {profit_factor:.2f} < 1.0",
                suggestion="Gross losses exceed gross wins. Strategy is net negative.",
            ))

        if win_rate < 0.30:
            issues.append(HealthIssue(
                category="statistical",
                severity="warning",
                field="win_rate",
                message=f"Win rate {win_rate:.0%} is below 30%",
                suggestion="Very low win rate — needs high R:R to compensate. Verify targets.",
            ))

        return issues


class StrategyLifecycleManager:
    """
    Manages strategy state through the full lifecycle.
    Single source of truth for lifecycle state.

    Coordinates: HealthGate, CalibrationTracker, EdgeMonitor.
    """

    def __init__(
        self,
        health_gate: Optional[StrategyHealthGate] = None,
    ):
        self._gate = health_gate or StrategyHealthGate()
        self._states: Dict[str, StrategyLifecycleState] = {}
        self._grades: Dict[str, StrategyGrade] = {}
        self._health_results: Dict[str, HealthGateResult] = {}
        self._lifecycle_info: Dict[str, LifecycleInfo] = {}

        # Pluggable sub-systems (set after construction)
        self._calibration_trackers: Dict[str, Any] = {}  # strategy_id → CalibrationTracker
        self._edge_monitors: Dict[str, Any] = {}         # strategy_id → ContinuousEdgeMonitor

        # Probation trade tracking
        self._probation_trades: Dict[str, List[float]] = {}  # strategy_id → r_multiples

        # DB factory for calibration signal persistence (set by integrations layer)
        self._calibration_db_factory: Optional[Any] = None

    # ── State Queries ──────────────────────────────────────────────

    def get_state(self, strategy_id: str) -> StrategyLifecycleState:
        return self._states.get(
            strategy_id, StrategyLifecycleState.INGESTED
        )

    def get_lifecycle_info(self, strategy_id: str) -> LifecycleInfo:
        return self._lifecycle_info.get(
            strategy_id,
            LifecycleInfo(strategy_id=strategy_id),
        )

    def get_all_lifecycle_info(self) -> Dict[str, LifecycleInfo]:
        return dict(self._lifecycle_info)

    def can_trade(self, strategy_id: str) -> Tuple[bool, str]:
        """Check if AVT is allowed to trade this strategy."""
        state = self.get_state(strategy_id)
        if state == StrategyLifecycleState.AUTONOMOUS:
            return True, "autonomous"
        if state == StrategyLifecycleState.PROBATION:
            return True, "probation (25% allocation)"
        if state == StrategyLifecycleState.DEGRADED:
            return True, "degraded (50% allocation)"
        if state == StrategyLifecycleState.CALIBRATING:
            return False, "calibrating — signal-only mode"
        return False, f"state={state.value} — trading not allowed"

    def get_allocation_multiplier(self, strategy_id: str) -> float:
        state = self.get_state(strategy_id)
        return LIFECYCLE_ALLOCATION.get(state, 0.0)

    # ── State Transitions ──────────────────────────────────────────

    def ingest_strategy(
        self,
        strategy: Dict[str, Any],
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> HealthGateResult:
        """
        Entry point: validate a strategy and set initial state.
        Called during setup_avt_full() for each strategy.
        """
        sid = str(strategy.get("id", ""))
        self._states[sid] = StrategyLifecycleState.HEALTH_CHECK

        # Run health gate
        result = self._gate.validate(strategy, trade_history)
        self._health_results[sid] = result

        # Set state based on grade
        if result.grade == StrategyGrade.REJECTED.value:
            self._set_state(sid, StrategyLifecycleState.REJECTED)
        elif result.grade == StrategyGrade.PROBATION.value:
            self._set_state(sid, StrategyLifecycleState.PROBATION)
            self._probation_trades[sid] = []
        else:
            # READY — go to autonomous (or calibrating if tracker exists)
            if sid in self._calibration_trackers:
                self._set_state(sid, StrategyLifecycleState.CALIBRATING)
            else:
                self._set_state(sid, StrategyLifecycleState.AUTONOMOUS)

        self._grades[sid] = StrategyGrade(result.grade)
        self._update_info(sid)

        logger.info(
            "Strategy %s: grade=%s, state=%s, health=%.1f",
            sid, result.grade, self._states[sid].value,
            result.overall_health_score,
        )
        return result

    def start_calibration(self, strategy_id: str) -> bool:
        """Enter calibration mode (signal-only, no execution)."""
        state = self.get_state(strategy_id)
        if state not in (
            StrategyLifecycleState.PROBATION,
            StrategyLifecycleState.AUTONOMOUS,
        ):
            logger.warning(
                "Cannot start calibration from state %s", state.value
            )
            return False

        self._set_state(strategy_id, StrategyLifecycleState.CALIBRATING)
        self._update_info(strategy_id)
        return True

    def skip_calibration(self, strategy_id: str) -> bool:
        """Skip calibration, go straight to autonomous."""
        state = self.get_state(strategy_id)
        if state != StrategyLifecycleState.CALIBRATING:
            return False

        self._set_state(strategy_id, StrategyLifecycleState.AUTONOMOUS)
        self._update_info(strategy_id)
        logger.info("Strategy %s: calibration skipped", strategy_id)
        return True

    def graduate_calibration(self, strategy_id: str) -> bool:
        """Graduate from calibration to autonomous."""
        state = self.get_state(strategy_id)
        if state != StrategyLifecycleState.CALIBRATING:
            return False

        self._set_state(strategy_id, StrategyLifecycleState.AUTONOMOUS)
        self._update_info(strategy_id)
        logger.info("Strategy %s: graduated to autonomous", strategy_id)
        return True

    def record_probation_trade(
        self, strategy_id: str, r_multiple: float
    ) -> Optional[str]:
        """
        Record a trade during probation. Returns new state if promoted.
        """
        if self.get_state(strategy_id) != StrategyLifecycleState.PROBATION:
            return None

        self._probation_trades.setdefault(strategy_id, []).append(r_multiple)
        trades = self._probation_trades[strategy_id]

        if len(trades) >= PROBATION_TRADE_COUNT:
            return self._evaluate_probation(strategy_id)

        return None

    def handle_degradation(
        self,
        strategy_id: str,
        level: str,
    ) -> None:
        """Handle edge degradation detected by edge monitor."""
        state = self.get_state(strategy_id)

        if level == "warning":
            logger.warning(
                "Strategy %s: edge WARNING — monitoring", strategy_id
            )
        elif level == "degraded" and state == StrategyLifecycleState.AUTONOMOUS:
            self._set_state(strategy_id, StrategyLifecycleState.DEGRADED)
            logger.warning(
                "Strategy %s: DEGRADED — allocation reduced to 50%%",
                strategy_id,
            )
        elif level == "critical":
            self._set_state(strategy_id, StrategyLifecycleState.PAUSED)
            logger.error(
                "Strategy %s: PAUSED — critical edge loss", strategy_id
            )

        self._update_info(strategy_id)

    def resume_strategy(self, strategy_id: str) -> bool:
        """Resume a paused or degraded strategy after user review."""
        state = self.get_state(strategy_id)
        if state not in (
            StrategyLifecycleState.DEGRADED,
            StrategyLifecycleState.PAUSED,
        ):
            return False

        self._set_state(strategy_id, StrategyLifecycleState.AUTONOMOUS)
        self._update_info(strategy_id)
        logger.info("Strategy %s: resumed to autonomous", strategy_id)
        return True

    # ── Sub-system Registration ────────────────────────────────────

    def set_calibration_tracker(
        self, strategy_id: str, tracker: Any
    ) -> None:
        """Register a calibration tracker and wire DB persistence callbacks."""
        if self._calibration_db_factory is not None:
            self._wire_calibration_persistence(strategy_id, tracker)
        self._calibration_trackers[strategy_id] = tracker

    def _wire_calibration_persistence(self, strategy_id: str, tracker: Any) -> None:
        """Inject DB-persisting callbacks into a calibration tracker."""
        db_factory = self._calibration_db_factory

        def _get_db():
            db = db_factory()
            if hasattr(db, "__next__"):
                return next(db)
            return db

        def persist_signal(signal):
            try:
                db = _get_db()
                try:
                    from avt.repository import AVTCalibrationRepository
                    AVTCalibrationRepository(db).save_signal(strategy_id, signal)
                    db.commit()
                finally:
                    db.close()
            except Exception as _e:
                logger.warning("Calibration signal DB persist failed: %s", _e)

        def persist_trade(trade):
            try:
                db = _get_db()
                try:
                    from avt.repository import AVTCalibrationRepository
                    AVTCalibrationRepository(db).save_user_trade(strategy_id, trade)
                    db.commit()
                finally:
                    db.close()
            except Exception as _e:
                logger.warning("Calibration user trade DB persist failed: %s", _e)

        tracker._persist_signal_callback = persist_signal
        tracker._persist_trade_callback = persist_trade

    def get_calibration_tracker(self, strategy_id: str) -> Any:
        return self._calibration_trackers.get(strategy_id)

    def set_edge_monitor(self, strategy_id: str, monitor: Any) -> None:
        self._edge_monitors[strategy_id] = monitor

    def get_edge_monitor(self, strategy_id: str) -> Any:
        return self._edge_monitors.get(strategy_id)

    def feed_edge_monitor(self, strategy_id: str, r_multiple: float) -> None:
        """Feed a closed trade R-multiple to the edge monitor and check degradation."""
        monitor = self._edge_monitors.get(strategy_id)
        if not monitor:
            return

        monitor.record_trade(r_multiple)
        check = monitor.check_degradation()

        if check.level in ("degraded", "critical"):
            self.handle_degradation(strategy_id, check.level)
        elif check.level == "warning":
            self.handle_degradation(strategy_id, "warning")

        self._update_info(strategy_id)

    @property
    def _lifecycles(self) -> Dict[str, StrategyLifecycleState]:
        """Alias for _states — used by integrations.py logging."""
        return self._states

    # ── Internal ───────────────────────────────────────────────────

    def _set_state(
        self, strategy_id: str, state: StrategyLifecycleState
    ) -> None:
        old = self._states.get(strategy_id)
        self._states[strategy_id] = state
        if old != state:
            logger.info(
                "Lifecycle %s: %s → %s",
                strategy_id,
                old.value if old else "none",
                state.value,
            )

    def _evaluate_probation(self, strategy_id: str) -> str:
        """Evaluate probation period after enough trades."""
        import numpy as np

        trades = self._probation_trades.get(strategy_id, [])
        arr = np.array(trades)
        expectancy = float(np.mean(arr))
        winners = arr[arr > 0]
        losers = arr[arr < 0]
        pf = (
            float(np.sum(winners) / abs(np.sum(losers)))
            if len(losers) > 0 and np.sum(losers) != 0
            else 999.0
        )

        if expectancy > PROBATION_MIN_EXPECTANCY and pf > PROBATION_MIN_PF:
            self._set_state(
                strategy_id, StrategyLifecycleState.AUTONOMOUS
            )
            self._grades[strategy_id] = StrategyGrade.READY
            logger.info(
                "Strategy %s PROMOTED: expectancy=%.3fR, PF=%.2f over %d trades",
                strategy_id, expectancy, pf, len(trades),
            )
            self._update_info(strategy_id)
            return "promoted"
        else:
            self._set_state(
                strategy_id, StrategyLifecycleState.REJECTED
            )
            self._grades[strategy_id] = StrategyGrade.REJECTED
            logger.warning(
                "Strategy %s REJECTED after probation: "
                "expectancy=%.3fR, PF=%.2f over %d trades",
                strategy_id, expectancy, pf, len(trades),
            )
            self._update_info(strategy_id)
            return "rejected"

    def _update_info(self, strategy_id: str) -> None:
        """Refresh the LifecycleInfo cache for a strategy."""
        state = self.get_state(strategy_id)
        grade = self._grades.get(strategy_id)
        health = self._health_results.get(strategy_id)

        info = LifecycleInfo(
            strategy_id=strategy_id,
            state=state.value,
            grade=grade.value if grade else "",
            health_score=health.overall_health_score if health else 0.0,
            allocation_multiplier=LIFECYCLE_ALLOCATION.get(state, 0.0),
            health_issues=[
                {
                    "category": i.category,
                    "severity": i.severity,
                    "field": i.field,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in (health.issues if health else [])
            ],
            state_changed_at=datetime.now(timezone.utc),
            validated_at=health.checked_at if health else None,
        )

        # Calibration info
        cal = self._calibration_trackers.get(strategy_id)
        if cal and hasattr(cal, "get_report"):
            report = cal.get_report()
            info.calibration_alignment = report.get("alignment_score")
            info.calibration_signals = report.get("signals_generated", 0)
            info.calibration_comparisons = report.get("total_comparisons", 0)

        # Edge monitor info
        mon = self._edge_monitors.get(strategy_id)
        if mon and hasattr(mon, "get_status"):
            status = mon.get_status()
            info.degradation_level = status.get("level", "healthy")
            info.rolling_expectancy = status.get("rolling_expectancy")
            info.translation_drift = status.get("drift_score")
            info.baseline_expectancy = status.get("baseline_expectancy")

        self._lifecycle_info[strategy_id] = info
