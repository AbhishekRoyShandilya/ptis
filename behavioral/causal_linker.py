"""
Causal Linker
===============

Connects: trade outcome + replay observations + matched strategy
   →  one human-readable causal sentence per trade.

Example output:
    "Loss = LATE_ENTRY (move 75% complete) + REGIME_MISMATCH (1h ranging,
     ORB needs trending) + RR_KILLED (peaked +2.1R, exited -0.4R)"

Why this exists:
  Existing detectors (RRKillerDetector, ExecutionDepthAnalyzer, BehaviorAnalytics)
  detect symptoms in isolation. The CausalLinker connects them into a single
  story per trade — what the trader needs to understand "why this lost".

Reuses:
  - `improvement_lab/rr_killer_detector.py` (we already have rule-based
    late-entry / early-exit logic) — we'll inline-detect using replay's
    observations to avoid extra DB queries.
  - `core/enums.py:BehavioralEventType` for cause tags
  - Replay's ReplayObservations + matcher's TradeMatchResult

Output:
  - List of CauseTag objects (one per detected cause)
  - One synthesized causal sentence
  - Per-cause severity scores

Real-money safety: linker is pure inference. No DB writes here — caller decides
which findings to persist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from behavioral.replay_models import ReplayedTrade
from behavioral.trade_matcher import TradeMatchResult

logger = logging.getLogger(__name__)


# ── Cause types (subset that maps to BehavioralEventType where possible) ──


CAUSE_LATE_ENTRY = "LATE_ENTRY"
CAUSE_EARLY_EXIT = "EARLY_EXIT"
CAUSE_RR_KILLED = "RR_KILLED"
CAUSE_REGIME_MISMATCH = "REGIME_MISMATCH"
CAUSE_REGIME_SHIFT = "REGIME_SHIFT"
CAUSE_AGAINST_TREND = "AGAINST_TREND"
CAUSE_OVEREXTENDED = "OVEREXTENDED"
CAUSE_STRATEGY_MISMATCH = "STRATEGY_MISMATCH"     # declared != matched
CAUSE_NO_STRATEGY_MATCH = "NO_STRATEGY_MATCH"     # didn't fit any configured strategy

# Positive causes (when trade WAS clean)
CAUSE_CLEAN_EXECUTION = "CLEAN_EXECUTION"
CAUSE_DISCIPLINE_HOLD = "DISCIPLINE_HOLD"
CAUSE_PATIENCE_REWARD = "PATIENCE_REWARD"


# Map cause tag → existing BehavioralEventType for persistence
_CAUSE_TO_EVENT_TYPE = {
    CAUSE_LATE_ENTRY: "fomo_entry",            # late entry often = FOMO
    CAUSE_EARLY_EXIT: "emotional_exit",
    CAUSE_RR_KILLED: "profit_giving_back",
    CAUSE_REGIME_MISMATCH: "impulse_entry",
    CAUSE_REGIME_SHIFT: None,                  # no exact mapping; stored in description
    CAUSE_AGAINST_TREND: "impulse_entry",
    CAUSE_OVEREXTENDED: "fomo_entry",
    CAUSE_STRATEGY_MISMATCH: "impulse_entry",
    CAUSE_NO_STRATEGY_MATCH: "impulse_entry",
    CAUSE_CLEAN_EXECUTION: "clean_execution",
    CAUSE_DISCIPLINE_HOLD: "discipline_hold",
    CAUSE_PATIENCE_REWARD: "patience_reward",
}


# ── Output ──────────────────────────────────────────────────────────


@dataclass
class CauseTag:
    """A single detected cause for a trade outcome."""
    cause: str                          # one of CAUSE_* constants
    severity: float                     # 0-1 (1 = strongly contributed)
    detail: str                         # one-line human-readable explanation
    evidence: Dict[str, Any] = field(default_factory=dict)
    is_positive: bool = False           # for clean execution / discipline events

    def to_event_type(self) -> Optional[str]:
        """Map to BehavioralEventType string (for DB persistence)."""
        return _CAUSE_TO_EVENT_TYPE.get(self.cause)


@dataclass
class CausalExplanation:
    """Complete causal analysis for a single trade."""
    trade_id: str
    outcome: str                        # "win" | "loss" | "breakeven"
    causes: List[CauseTag] = field(default_factory=list)
    sentence: str = ""                  # the synthesized one-line causal sentence
    declared_strategy_id: Optional[str] = None
    matched_strategy_id: Optional[str] = None

    def has_negative_causes(self) -> bool:
        return any(not c.is_positive for c in self.causes)

    def primary_cause(self) -> Optional[CauseTag]:
        """The single most-contributing cause (highest severity)."""
        if not self.causes:
            return None
        return max(self.causes, key=lambda c: c.severity)


# ── Linker ──────────────────────────────────────────────────────────


class CausalLinker:
    """
    Stateless analyzer. Given a replay + match result + trade outcome,
    produces a CausalExplanation.

    Thresholds are configurable but defaults work for most use cases.
    """

    # Detection thresholds (tunable)
    LATE_ENTRY_PCT = 0.65               # >65% of move complete = late
    LATE_ENTRY_RSI_OVERBOUGHT = 75.0
    LATE_ENTRY_RSI_OVERSOLD = 25.0
    EARLY_EXIT_SEVERITY = 0.30          # 0.3+ on the 0-1 scale (≈0.6R given back)
    EARLY_EXIT_MIN_PEAK_R = 0.8         # only flag if trade actually had upside
    RR_KILLED_PEAK_R = 1.0              # peaked above 1R but exited at loss
    RR_KILLED_FINAL_R = 0.0
    OVEREXTENDED_PEAK_R = 2.0
    REGIME_MISMATCH_LOW_CONFIDENCE = 0.55  # if matcher confidence below this, regime likely off

    def link(
        self,
        replay: ReplayedTrade,
        match_result: Optional[TradeMatchResult] = None,
        declared_strategy_id: Optional[str] = None,
        trade_pnl: Optional[float] = None,
        trade_r_multiple: Optional[float] = None,
    ) -> CausalExplanation:
        """
        Main entry point. Returns a CausalExplanation with a list of detected
        causes ranked by severity.
        """
        outcome = self._classify_outcome(trade_pnl, trade_r_multiple, replay)

        explanation = CausalExplanation(
            trade_id=replay.trade_id,
            outcome=outcome,
            declared_strategy_id=declared_strategy_id,
            matched_strategy_id=match_result.best_match_id if match_result else None,
        )

        causes: List[CauseTag] = []

        # Run all detectors — each returns Optional[CauseTag]
        causes.append(self._detect_late_entry(replay))
        causes.append(self._detect_early_exit(replay))
        causes.append(self._detect_rr_killed(replay))
        causes.append(self._detect_regime_mismatch(replay, match_result))
        causes.append(self._detect_regime_shift(replay))
        causes.append(self._detect_against_trend(replay))
        causes.append(self._detect_overextended(replay))
        causes.append(self._detect_strategy_mismatch(match_result, declared_strategy_id))
        causes.append(self._detect_no_strategy_match(match_result))

        # Positive causes (only when trade was a win + clean)
        if outcome == "win":
            causes.append(self._detect_clean_execution(replay, match_result))
            causes.append(self._detect_patience_reward(replay))

        # Filter Nones, rank by severity
        explanation.causes = [c for c in causes if c is not None]
        explanation.causes.sort(key=lambda c: c.severity, reverse=True)

        # Synthesize the one-line causal sentence
        explanation.sentence = self._synthesize_sentence(explanation)

        return explanation

    # ── Detectors (each returns Optional[CauseTag]) ──

    def _detect_late_entry(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        obs = replay.observations
        pct = obs.move_pct_complete_at_entry
        rsi = obs.rsi_at_entry

        # Strong late-entry signal: pct > 65% AND (RSI extreme OR no other support)
        if pct is None:
            return None
        if pct >= self.LATE_ENTRY_PCT:
            severity = min(1.0, (pct - self.LATE_ENTRY_PCT) / (1.0 - self.LATE_ENTRY_PCT) + 0.4)
            detail = f"entered when move was {int(pct * 100)}% complete"
            evidence = {"move_pct_complete": pct}
            if rsi is not None:
                is_long = (replay.direction or "long").lower() == "long"
                if is_long and rsi > self.LATE_ENTRY_RSI_OVERBOUGHT:
                    severity = min(1.0, severity + 0.15)
                    detail += f", RSI {rsi:.0f} overbought"
                    evidence["rsi"] = rsi
                elif (not is_long) and rsi < self.LATE_ENTRY_RSI_OVERSOLD:
                    severity = min(1.0, severity + 0.15)
                    detail += f", RSI {rsi:.0f} oversold (for short)"
                    evidence["rsi"] = rsi
            return CauseTag(
                cause=CAUSE_LATE_ENTRY,
                severity=round(severity, 4),
                detail=detail,
                evidence=evidence,
            )
        return None

    def _detect_early_exit(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        obs = replay.observations
        if obs.peak_r_during_trade < self.EARLY_EXIT_MIN_PEAK_R:
            return None  # trade never had real upside — not "early"
        if obs.early_exit_severity < self.EARLY_EXIT_SEVERITY:
            return None
        give_back = obs.peak_r_during_trade - obs.final_r
        return CauseTag(
            cause=CAUSE_EARLY_EXIT,
            severity=round(obs.early_exit_severity, 4),
            detail=(
                f"peaked at +{obs.peak_r_during_trade:.1f}R, exited at "
                f"{obs.final_r:+.1f}R (gave back {give_back:.1f}R)"
            ),
            evidence={
                "peak_r": obs.peak_r_during_trade,
                "final_r": obs.final_r,
                "give_back_r": round(give_back, 4),
            },
        )

    def _detect_rr_killed(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        """Trade reached meaningful favorable R but ended at loss or near-zero."""
        obs = replay.observations
        if obs.peak_r_during_trade < self.RR_KILLED_PEAK_R:
            return None
        if obs.final_r >= self.RR_KILLED_FINAL_R:
            return None
        # Severity scales with how much was given back relative to peak
        give_back_ratio = (obs.peak_r_during_trade - obs.final_r) / max(obs.peak_r_during_trade, 0.01)
        severity = min(1.0, give_back_ratio)
        return CauseTag(
            cause=CAUSE_RR_KILLED,
            severity=round(severity, 4),
            detail=(
                f"peaked +{obs.peak_r_during_trade:.1f}R but ended {obs.final_r:+.1f}R — "
                f"R:R completely flipped"
            ),
            evidence={"peak_r": obs.peak_r_during_trade, "final_r": obs.final_r},
        )

    def _detect_regime_mismatch(
        self,
        replay: ReplayedTrade,
        match_result: Optional[TradeMatchResult],
    ) -> Optional[CauseTag]:
        """If matcher's confidence is low or matched strategy doesn't prefer current regime."""
        if match_result is None or not match_result.best_match_id:
            return None
        # Low matcher confidence is a regime/setup issue
        if match_result.best_match_score >= self.REGIME_MISMATCH_LOW_CONFIDENCE:
            return None
        regime = replay.observations.regime_at_entry or "unknown"
        return CauseTag(
            cause=CAUSE_REGIME_MISMATCH,
            severity=round(1.0 - match_result.best_match_score, 4),
            detail=(
                f"current regime '{regime}' doesn't fit any configured strategy well "
                f"(best match: {match_result.best_match_name} at {int(match_result.best_match_score*100)}%)"
            ),
            evidence={
                "regime": regime,
                "best_match": match_result.best_match_id,
                "best_match_score": match_result.best_match_score,
            },
        )

    def _detect_regime_shift(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        """Regime changed between entry and exit — momentum lost."""
        obs = replay.observations
        if not obs.regime_changed:
            return None
        return CauseTag(
            cause=CAUSE_REGIME_SHIFT,
            severity=0.6,
            detail=(
                f"regime shifted during trade: {obs.regime_at_entry} → {obs.regime_at_exit}"
            ),
            evidence={
                "regime_entry": obs.regime_at_entry,
                "regime_exit": obs.regime_at_exit,
            },
        )

    def _detect_against_trend(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        """Entry was against the higher-timeframe trend."""
        if replay.observations.trend_alignment_at_entry != "against_trend":
            return None
        return CauseTag(
            cause=CAUSE_AGAINST_TREND,
            severity=0.55,
            detail="entered against the 1h-trend (counter-trend trade)",
            evidence={"alignment": "against_trend"},
        )

    def _detect_overextended(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        """Entry was at very high RSI (or very low for shorts)."""
        rsi = replay.observations.rsi_at_entry
        if rsi is None:
            return None
        is_long = (replay.direction or "long").lower() == "long"
        if is_long and rsi > self.LATE_ENTRY_RSI_OVERBOUGHT:
            return CauseTag(
                cause=CAUSE_OVEREXTENDED,
                severity=min(1.0, (rsi - self.LATE_ENTRY_RSI_OVERBOUGHT) / 25.0 + 0.5),
                detail=f"RSI {rsi:.0f} overbought at entry (long)",
                evidence={"rsi": rsi},
            )
        if (not is_long) and rsi < self.LATE_ENTRY_RSI_OVERSOLD:
            return CauseTag(
                cause=CAUSE_OVEREXTENDED,
                severity=min(1.0, (self.LATE_ENTRY_RSI_OVERSOLD - rsi) / 25.0 + 0.5),
                detail=f"RSI {rsi:.0f} oversold at entry (short)",
                evidence={"rsi": rsi},
            )
        return None

    def _detect_strategy_mismatch(
        self,
        match_result: Optional[TradeMatchResult],
        declared_strategy_id: Optional[str],
    ) -> Optional[CauseTag]:
        """User declared one strategy but the conditions matched a different one."""
        if not declared_strategy_id or match_result is None:
            return None
        if not match_result.best_match_id or match_result.declared_match is True:
            return None
        return CauseTag(
            cause=CAUSE_STRATEGY_MISMATCH,
            severity=0.65,
            detail=(
                f"you said '{declared_strategy_id}' but conditions actually matched "
                f"'{match_result.best_match_id}'"
            ),
            evidence={
                "declared": declared_strategy_id,
                "matched": match_result.best_match_id,
                "match_score": match_result.best_match_score,
            },
        )

    def _detect_no_strategy_match(
        self,
        match_result: Optional[TradeMatchResult],
    ) -> Optional[CauseTag]:
        """Trade didn't fit any of the configured strategies."""
        if match_result is None or match_result.best_match_id:
            return None
        return CauseTag(
            cause=CAUSE_NO_STRATEGY_MATCH,
            severity=0.75,
            detail="no configured strategy matched the entry conditions — likely a discretionary/impulse trade",
            evidence={"reason": "no_match_above_threshold"},
        )

    def _detect_clean_execution(
        self,
        replay: ReplayedTrade,
        match_result: Optional[TradeMatchResult],
    ) -> Optional[CauseTag]:
        """Positive: matched a strategy, regime aligned, trend supportive, no early exit."""
        if match_result is None or not match_result.best_match_id:
            return None
        if match_result.best_match_score < 0.65:
            return None
        if replay.observations.early_exit_severity > 0.20:
            return None
        if replay.observations.trend_alignment_at_entry == "against_trend":
            return None
        return CauseTag(
            cause=CAUSE_CLEAN_EXECUTION,
            severity=0.8,
            detail=(
                f"textbook execution of {match_result.best_match_name} — "
                f"strong setup match + no early exit + with trend"
            ),
            evidence={"match_score": match_result.best_match_score},
            is_positive=True,
        )

    def _detect_patience_reward(self, replay: ReplayedTrade) -> Optional[CauseTag]:
        """Positive: held winning trade through volatility instead of bailing early."""
        obs = replay.observations
        if obs.peak_r_during_trade < 1.5:
            return None
        if obs.final_r < 1.0:
            return None
        # Held through some adverse move
        if not replay.excursion_path:
            return None
        # Check for at least one adverse-then-recovery sequence
        had_drawdown = any(p.r_at_this_bar < -0.3 for p in replay.excursion_path)
        if not had_drawdown:
            return None
        return CauseTag(
            cause=CAUSE_PATIENCE_REWARD,
            severity=0.7,
            detail=(
                f"held through pullback to capture +{obs.final_r:.1f}R (peak +{obs.peak_r_during_trade:.1f}R)"
            ),
            evidence={"final_r": obs.final_r, "peak_r": obs.peak_r_during_trade},
            is_positive=True,
        )

    # ── Sentence synthesis ──

    @staticmethod
    def _synthesize_sentence(explanation: CausalExplanation) -> str:
        """
        Build one human-readable line summarizing the causes.
        Format depends on outcome:
          loss: "Loss = CAUSE1 (detail) + CAUSE2 (detail) ..."
          win:  "Win = CAUSE1 (detail) ..."
        """
        if not explanation.causes:
            return f"{explanation.outcome.title()} — no notable behavioral or market causes detected."

        # Take top 3 causes only — never lecture (per partner's "1-3 max" rule)
        top = explanation.causes[:3]

        parts = []
        for c in top:
            parts.append(f"{c.cause} ({c.detail})")

        prefix = explanation.outcome.title()
        return f"{prefix} = " + " + ".join(parts)

    # ── Outcome classification ──

    @staticmethod
    def _classify_outcome(
        trade_pnl: Optional[float],
        trade_r: Optional[float],
        replay: ReplayedTrade,
    ) -> str:
        if trade_pnl is not None:
            if trade_pnl > 0:
                return "win"
            if trade_pnl < 0:
                return "loss"
            return "breakeven"
        if trade_r is not None:
            if trade_r > 0.1:
                return "win"
            if trade_r < -0.1:
                return "loss"
            return "breakeven"
        # Fallback: use replay's final_r
        fr = replay.observations.final_r
        if fr > 0.1:
            return "win"
        if fr < -0.1:
            return "loss"
        return "breakeven"
