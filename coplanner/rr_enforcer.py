# coplanner/rr_enforcer.py
"""
Risk-Reward Gate for Co-Planner.

ENFORCES minimum risk-reward standards — not a suggestion, a RULE.
A beautiful pattern with terrible RR is still a bad trade.

Supports three modes:
  STRICT  — blocks trades below threshold
  WARN    — allows but warns and reduces size
  DYNAMIC — adjusts threshold based on conviction

Dynamic conviction adjustment:
  conviction >= 0.85 → min RR * 0.80  (slightly lenient)
  conviction >= 0.70 → min RR * 0.90
  conviction >= 0.55 → min RR * 1.00  (standard)
  conviction <  0.55 → min RR * 1.30  (stricter)
  Floor: never below 1.0 (breakeven)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from coplanner.models import RRAssessment, RRGrade

logger = logging.getLogger(__name__)


class RiskRewardEnforcer:
    """
    Enforces minimum risk-reward standards on all trade setups.

    Usage:
        enforcer = RiskRewardEnforcer(user_profile)
        assessment = enforcer.assess(entry=100, stop=97, target=108)
        assessment = enforcer.assess_with_conviction(
            entry=100, stop=97, target=105, conviction_score=0.85
        )
    """

    DEFAULT_MIN_RR = 1.5
    DEFAULT_PREFERRED_RR = 2.0
    DEFAULT_EXCELLENT_RR = 2.5
    DEFAULT_MODE = "WARN"

    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        profile = user_profile or {}
        self.min_rr = profile.get("min_rr_threshold", self.DEFAULT_MIN_RR)
        self.preferred_rr = profile.get("preferred_rr", self.DEFAULT_PREFERRED_RR)
        self.excellent_rr = profile.get("excellent_rr", self.DEFAULT_EXCELLENT_RR)
        self.mode = profile.get("enforcement_mode", self.DEFAULT_MODE)
        self.dynamic = profile.get("dynamic_adjustment", True)
        self._enforcement_log: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────
    # CORE ASSESSMENT
    # ─────────────────────────────────────────────

    def assess(
        self,
        entry: float,
        stop_loss: float,
        target: float,
        trade_direction: str = "auto",
    ) -> RRAssessment:
        """Assess risk-reward of a trade setup."""
        validation = self._validate_prices(entry, stop_loss, target, trade_direction)
        if validation:
            return validation
        rr = self.compute_rr(entry, stop_loss, target)
        return self._grade_rr(rr, entry, stop_loss, target)

    def assess_from_rr(self, expected_rr: float) -> RRAssessment:
        """When RR is already computed."""
        return self._grade_rr(expected_rr)

    def assess_with_conviction(
        self,
        entry: float,
        stop_loss: float,
        target: float,
        conviction_score: float,
        trade_direction: str = "auto",
    ) -> RRAssessment:
        """DYNAMIC MODE: conviction affects minimum RR threshold."""
        if not self.dynamic:
            return self.assess(entry, stop_loss, target, trade_direction)

        validation = self._validate_prices(entry, stop_loss, target, trade_direction)
        if validation:
            return validation

        rr = self.compute_rr(entry, stop_loss, target)
        adj_min = self._dynamic_min_rr(conviction_score)
        adj_pref = self._dynamic_preferred_rr(conviction_score)
        return self._grade_rr_dynamic(
            rr, adj_min, adj_pref, conviction_score, entry, stop_loss, target
        )

    # ─────────────────────────────────────────────
    # RR COMPUTATION
    # ─────────────────────────────────────────────

    @staticmethod
    def compute_rr(entry: float, stop_loss: float, target: float) -> float:
        """Compute risk-reward ratio. Handles long and short."""
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        if risk == 0:
            return 0.0
        return round(reward / risk, 2)

    def compute_rr_with_slippage(
        self,
        entry: float,
        stop_loss: float,
        target: float,
        slippage_pct: float = 0.1,
    ) -> float:
        """Realistic RR accounting for slippage on both sides."""
        slip = entry * (slippage_pct / 100)
        if target > entry:  # long
            real_entry = entry + slip
            real_stop = stop_loss - slip
            real_target = target - slip
        else:  # short
            real_entry = entry - slip
            real_stop = stop_loss + slip
            real_target = target + slip
        risk = abs(real_entry - real_stop)
        reward = abs(real_target - real_entry)
        return round(reward / risk, 2) if risk > 0 else 0.0

    # ─────────────────────────────────────────────
    # GRADING
    # ─────────────────────────────────────────────

    def _grade_rr(
        self, rr: float,
        entry: float = 0, stop_loss: float = 0, target: float = 0,
    ) -> RRAssessment:
        if rr >= self.excellent_rr:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.EXCELLENT, acceptable=True,
                message=f"RR {rr:.1f} — Excellent. Full size appropriate.",
                sizing_modifier=1.0,
            )
        if rr >= self.preferred_rr:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.GOOD, acceptable=True,
                message=f"RR {rr:.1f} — Good. Meets preferred {self.preferred_rr}.",
                sizing_modifier=1.0,
            )
        if rr >= self.min_rr:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.ACCEPTABLE, acceptable=True,
                message=(
                    f"RR {rr:.1f} — Acceptable but below preferred "
                    f"({self.preferred_rr}). Consider reduced size."
                ),
                sizing_modifier=0.7,
            )
        if rr >= 1.0:
            suggestion = self._suggest_target(entry, stop_loss, rr)
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.POOR,
                acceptable=(self.mode != "STRICT"),
                message=(
                    f"RR {rr:.1f} — Below minimum {self.min_rr}. "
                    f"{'BLOCKED.' if self.mode == 'STRICT' else 'Strongly recommend skipping.'}"
                ),
                sizing_modifier=0.0 if self.mode == "STRICT" else 0.3,
                suggested_target=suggestion["target"] if suggestion else None,
                suggestion_message=suggestion["message"] if suggestion else None,
            )
        # Below 1.0
        return RRAssessment(
            expected_rr=rr, grade=RRGrade.TERRIBLE, acceptable=False,
            message=f"RR {rr:.1f} — Risk exceeds reward. DO NOT TAKE.",
            sizing_modifier=0.0,
        )

    def _grade_rr_dynamic(
        self, rr: float,
        adj_min: float, adj_pref: float,
        conviction: float,
        entry: float = 0, stop_loss: float = 0, target: float = 0,
    ) -> RRAssessment:
        label = (
            "HIGH" if conviction >= 0.78
            else "MODERATE" if conviction >= 0.55
            else "LOW"
        )
        if rr >= self.excellent_rr:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.EXCELLENT, acceptable=True,
                message=f"RR {rr:.1f} — Excellent. Conviction: {conviction:.2f} ({label}).",
                sizing_modifier=1.0,
            )
        if rr >= adj_pref:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.GOOD, acceptable=True,
                message=f"RR {rr:.1f} — Good for {label} conviction (dynamic: {adj_pref:.1f}).",
                sizing_modifier=0.95 if conviction >= 0.78 else 0.85,
            )
        if rr >= adj_min:
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.ACCEPTABLE, acceptable=True,
                message=(
                    f"RR {rr:.1f} — Passes dynamic min ({adj_min:.1f}) "
                    f"for {label} conviction. Reduced sizing."
                ),
                sizing_modifier=0.6 if conviction >= 0.78 else 0.5,
            )
        if rr >= 1.0:
            suggestion = self._suggest_target(entry, stop_loss, rr)
            return RRAssessment(
                expected_rr=rr, grade=RRGrade.POOR, acceptable=False,
                message=(
                    f"RR {rr:.1f} — Below dynamic min {adj_min:.1f} "
                    f"(conviction: {conviction:.2f}). Skip."
                ),
                sizing_modifier=0.0,
                suggested_target=suggestion["target"] if suggestion else None,
                suggestion_message=suggestion["message"] if suggestion else None,
            )
        return RRAssessment(
            expected_rr=rr, grade=RRGrade.TERRIBLE, acceptable=False,
            message=f"RR {rr:.1f} — Risk exceeds reward. No conviction justifies this.",
            sizing_modifier=0.0,
        )

    # ─────────────────────────────────────────────
    # DYNAMIC THRESHOLDS
    # ─────────────────────────────────────────────

    def _dynamic_min_rr(self, conviction: float) -> float:
        if conviction >= 0.85:
            adj = self.min_rr * 0.80
        elif conviction >= 0.70:
            adj = self.min_rr * 0.90
        elif conviction >= 0.55:
            adj = self.min_rr * 1.00
        else:
            adj = self.min_rr * 1.30
        return max(1.0, round(adj, 2))

    def _dynamic_preferred_rr(self, conviction: float) -> float:
        if conviction >= 0.85:
            adj = self.preferred_rr * 0.85
        elif conviction >= 0.70:
            adj = self.preferred_rr * 0.95
        elif conviction >= 0.55:
            adj = self.preferred_rr * 1.00
        else:
            adj = self.preferred_rr * 1.20
        return max(self._dynamic_min_rr(conviction) + 0.1, round(adj, 2))

    # ─────────────────────────────────────────────
    # TARGET SUGGESTION
    # ─────────────────────────────────────────────

    def _suggest_target(
        self, entry: float, stop_loss: float, current_rr: float,
    ) -> Optional[Dict[str, Any]]:
        if not entry or not stop_loss:
            return None
        risk = abs(entry - stop_loss)
        needed = risk * self.min_rr
        if entry > stop_loss:
            suggested = round(entry + needed, 2)
            direction = "above entry"
        else:
            suggested = round(entry - needed, 2)
            direction = "below entry"
        return {
            "target": suggested,
            "message": (
                f"Target needs to be {suggested} ({direction}) "
                f"for minimum RR {self.min_rr}. Current RR: {current_rr:.1f}."
            ),
        }

    def suggest_multiple_targets(
        self, entry: float, stop_loss: float,
    ) -> Dict[str, Any]:
        """Multi-target exit levels."""
        risk = abs(entry - stop_loss)
        is_long = entry > stop_loss
        targets = {}
        for label, rr in [
            ("minimum", self.min_rr),
            ("preferred", self.preferred_rr),
            ("excellent", self.excellent_rr),
            ("stretch", self.excellent_rr * 1.5),
        ]:
            reward = risk * rr
            t = round((entry + reward) if is_long else (entry - reward), 2)
            targets[label] = {
                "rr": rr, "target_price": t,
                "reward_amount": round(reward, 2), "risk_amount": round(risk, 2),
            }
        return {
            "entry": entry, "stop_loss": stop_loss,
            "risk": round(risk, 2),
            "direction": "LONG" if is_long else "SHORT",
            "targets": targets,
            "suggestion": (
                f"Partial exits: 50% at {targets['preferred']['target_price']} "
                f"(RR {self.preferred_rr}), 30% at "
                f"{targets['excellent']['target_price']} (RR {self.excellent_rr}), "
                f"20% trail from there."
            ),
        }

    # ─────────────────────────────────────────────
    # PORTFOLIO-LEVEL RR
    # ─────────────────────────────────────────────

    def assess_portfolio_rr(self, open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall portfolio risk-reward."""
        if not open_positions:
            return {"portfolio_avg_rr": 0, "assessment": "NO_POSITIONS", "message": "No open positions."}

        rrs = []
        low_rr_count = 0
        for pos in open_positions:
            rr = self.compute_rr(
                pos.get("entry", 0), pos.get("stop_loss", 0), pos.get("target", 0)
            )
            rrs.append(rr)
            if rr < self.min_rr:
                low_rr_count += 1

        avg_rr = sum(rrs) / len(rrs) if rrs else 0
        if avg_rr >= self.preferred_rr and low_rr_count == 0:
            assessment, msg = "STRONG", f"Portfolio RR healthy — avg {avg_rr:.1f}."
        elif avg_rr >= self.min_rr:
            assessment = "ACCEPTABLE"
            msg = f"Portfolio RR acceptable — avg {avg_rr:.1f}. {low_rr_count} below min."
        else:
            assessment = "WEAK"
            msg = (
                f"Portfolio RR weak — avg {avg_rr:.1f}. "
                f"{low_rr_count}/{len(rrs)} below min."
            )
        return {
            "portfolio_avg_rr": round(avg_rr, 2),
            "total_positions": len(rrs),
            "low_rr_positions": low_rr_count,
            "assessment": assessment,
            "message": msg,
        }

    # ─────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────

    def _validate_prices(
        self, entry: float, stop_loss: float, target: float, direction: str,
    ) -> Optional[RRAssessment]:
        if entry <= 0 or stop_loss <= 0 or target <= 0:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Invalid prices — must be positive.", sizing_modifier=0.0,
            )
        if entry == stop_loss:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Entry equals stop — zero risk, undefined RR.", sizing_modifier=0.0,
            )
        is_long = (target > entry) if direction == "auto" else (direction.lower() == "long")
        if is_long and stop_loss >= entry:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Long trade but stop >= entry.", sizing_modifier=0.0,
            )
        if is_long and target <= entry:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Long trade but target <= entry.", sizing_modifier=0.0,
            )
        if not is_long and stop_loss <= entry:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Short trade but stop <= entry.", sizing_modifier=0.0,
            )
        if not is_long and target >= entry:
            return RRAssessment(
                expected_rr=0, grade=RRGrade.TERRIBLE, acceptable=False,
                message="Short trade but target >= entry.", sizing_modifier=0.0,
            )
        return None

    # ─────────────────────────────────────────────
    # LOGGING & CONFIG
    # ─────────────────────────────────────────────

    def log_enforcement(self, symbol: str, assessment: RRAssessment) -> None:
        self._enforcement_log.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "rr": assessment.expected_rr,
            "grade": assessment.grade.value,
            "acceptable": assessment.acceptable,
            "sizing_modifier": assessment.sizing_modifier,
        })

    def get_current_config(self) -> Dict[str, Any]:
        return {
            "min_rr": self.min_rr,
            "preferred_rr": self.preferred_rr,
            "excellent_rr": self.excellent_rr,
            "mode": self.mode,
            "dynamic_adjustment": self.dynamic,
        }
