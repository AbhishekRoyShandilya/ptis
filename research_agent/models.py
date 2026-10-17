"""
Shared dataclasses for research_agent findings.

These are the structured outputs of decay & emergence detectors.
They get persisted to the research_findings table (Phase 4) and surfaced
in the /research page UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DecayConfidence(str, Enum):
    HIGH = "high"          # all 3 detectors fired
    MODERATE = "moderate"  # 2 of 3 fired (majority vote)
    LOW = "low"            # only 1 fired — NOT promoted to finding


@dataclass
class DecayFinding:
    """
    A strategy whose recent performance has degraded vs its baseline.

    Created when ≥2 of 3 detectors (Mann-Whitney + CUSUM + Page-Hinkley) fire.
    NEVER triggers auto-pause — surfaces as alert for trader review.
    """
    strategy_id: str
    detected_at: datetime

    # Statistics
    p_value: float                     # Mann-Whitney U test p-value
    baseline_avg_r: float              # mean R over baseline window (excluding recent)
    recent_avg_r: float                # mean R over recent window
    drift_magnitude: float             # baseline_avg_r - recent_avg_r (positive = decay)

    # Detector vote
    detectors_firing: int              # 0-3 (only ≥2 produces a finding)
    mw_fired: bool
    cusum_fired: bool
    ph_fired: bool
    confidence: DecayConfidence

    # Sample sizes
    baseline_n: int
    recent_n: int

    # Recommended action
    suggested_action: str = ""
    rationale: str = ""

    def short_summary(self) -> str:
        return (
            f"{self.strategy_id}: avg R {self.baseline_avg_r:.2f} → {self.recent_avg_r:.2f} "
            f"(drift {self.drift_magnitude:+.2f}, p={self.p_value:.4f}, "
            f"{self.detectors_firing}/3 detectors, {self.confidence.value})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "detected_at": self.detected_at.isoformat(),
            "p_value": self.p_value,
            "baseline_avg_r": self.baseline_avg_r,
            "recent_avg_r": self.recent_avg_r,
            "drift_magnitude": self.drift_magnitude,
            "detectors_firing": self.detectors_firing,
            "detectors": {
                "mann_whitney": self.mw_fired,
                "cusum": self.cusum_fired,
                "page_hinkley": self.ph_fired,
            },
            "confidence": self.confidence.value,
            "baseline_n": self.baseline_n,
            "recent_n": self.recent_n,
            "suggested_action": self.suggested_action,
            "rationale": self.rationale,
        }


@dataclass
class EmergenceFinding:
    """
    A strategy whose preferred regime is becoming more frequent
    AND whose win rate is rising in recent trades.

    Trader gets a "consider enabling / increasing sizing on this strategy" alert.
    """
    strategy_id: str
    detected_at: datetime

    preferred_regime: str              # the regime that's growing in frequency

    # Regime distribution shift
    regime_pct_baseline: float         # % of bars in this regime, long-term baseline
    regime_pct_recent: float           # % in recent 30-day window
    regime_pct_change: float           # recent - baseline (positive = emerging)

    # Win rate shift
    win_rate_baseline: float
    win_rate_recent: float
    win_rate_change: float             # recent - baseline (positive = improving)

    # Sample sizes
    baseline_n: int
    recent_n: int

    suggested_action: str = ""
    rationale: str = ""

    def short_summary(self) -> str:
        return (
            f"{self.strategy_id}: WR {self.win_rate_baseline:.0%} → {self.win_rate_recent:.0%}, "
            f"regime '{self.preferred_regime}' freq {self.regime_pct_baseline:.0%} → "
            f"{self.regime_pct_recent:.0%} (Δ{self.regime_pct_change:+.0%})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "detected_at": self.detected_at.isoformat(),
            "preferred_regime": self.preferred_regime,
            "regime_pct_baseline": self.regime_pct_baseline,
            "regime_pct_recent": self.regime_pct_recent,
            "regime_pct_change": self.regime_pct_change,
            "win_rate_baseline": self.win_rate_baseline,
            "win_rate_recent": self.win_rate_recent,
            "win_rate_change": self.win_rate_change,
            "baseline_n": self.baseline_n,
            "recent_n": self.recent_n,
            "suggested_action": self.suggested_action,
            "rationale": self.rationale,
        }
