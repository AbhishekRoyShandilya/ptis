# discovery/regime_correlator.py
"""
Stage 5 — Regime Correlator

Maps each backtested match to its market regime at occurrence time,
then produces performance breakdowns by:
  1. Market Regime (trending_up, ranging, breakout, etc.)
  2. Time of Day (09:15-10:00, 10:00-11:30, etc.)

Also generates human-readable explanations of *why* the pattern
works or fails in each regime — using the fingerprint dimensions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from discovery.models import (
    BacktestResult,
    CorrelationReport,
    RegimeBreakdown,
    TimeSlotBreakdown,
)

logger = logging.getLogger(__name__)

# ── IST trading session time slots ────────────────────────────────
_TIME_SLOTS = [
    ("09:15", "10:00", "Opening Volatility"),
    ("10:00", "11:30", "Morning Prime"),
    ("11:30", "13:00", "Midday Lull"),
    ("13:00", "14:30", "Afternoon Recovery"),
    ("14:30", "15:30", "Power Hour"),
]

# ── Regime explanation templates ──────────────────────────────────
# Keyed by (regime, outcome) where outcome is "works" or "fails"
_REGIME_EXPLANATIONS: Dict[tuple, str] = {
    ("trending_up", "works"): (
        "Volume surge confirms institutional participation. "
        "Momentum aligned with broader market direction. "
        "Consolidation acts as energy compression before release. "
        "No counter-trend resistance to fight."
    ),
    ("trending_up", "fails"): (
        "Even in uptrends, late-stage breakouts can trap longs. "
        "Distribution disguised as breakout."
    ),
    ("trending_down", "works"): (
        "Short-side breakouts ride the dominant trend. "
        "Panic selling amplifies momentum. "
        "Support levels break faster in fear-driven markets."
    ),
    ("trending_down", "fails"): (
        "Bear market rallies produce false breakdowns. "
        "Short squeezes reverse gains quickly."
    ),
    ("ranging", "works"): (
        "Occasionally range boundaries give clear R:R. "
        "Works when range was about to resolve."
    ),
    ("ranging", "fails"): (
        "Breakouts get absorbed by supply at range boundary. "
        "No directional momentum to sustain move. "
        "Volume surge is short-lived without trend support."
    ),
    ("breakout", "works"): (
        "Fresh breakout from consolidation has maximum energy. "
        "Institutional accumulation releasing all at once. "
        "Stops above/below range trigger cascade."
    ),
    ("breakout", "fails"): (
        "False breakouts trap early entrants. "
        "Need volume confirmation to distinguish real from fake."
    ),
    ("high_volatility", "works"): (
        "Wide moves give large R-multiples when direction is right. "
        "Momentum carries further in volatile conditions."
    ),
    ("high_volatility", "fails"): (
        "Whipsaws hit stops before trend resumes. "
        "ATR-based stops are wider, reducing win rate. "
        "Noise overwhelms signal."
    ),
    ("low_volatility", "works"): (
        "Tight range compression precedes explosive moves. "
        "Squeeze setups are high-probability."
    ),
    ("low_volatility", "fails"): (
        "Breakout lacks energy to sustain. "
        "Volume dries up, move fades quickly."
    ),
    ("mean_reverting", "works"): (
        "Counter-trend bounce has well-defined risk. "
        "Extreme RSI provides mean-reversion edge."
    ),
    ("mean_reverting", "fails"): (
        "Catching a falling knife — momentum overwhelms reversion. "
        "Works only at defined structure levels."
    ),
    ("unknown", "works"): "Mixed conditions — pattern held despite unclear regime.",
    ("unknown", "fails"): "Uncertain regime reduces conviction and pattern reliability.",
}


class RegimeCorrelator:
    """
    Analyses backtest results by market regime and time of day.

    Usage:
        correlator = RegimeCorrelator()
        report = correlator.analyse(backtest_results)
    """

    def analyse(self, results: List[BacktestResult]) -> CorrelationReport:
        """Produce full regime + time-of-day correlation analysis."""
        regime_bd = self._regime_breakdown(results)
        time_bd = self._time_breakdown(results)

        # Find best/worst
        best_regime = ""
        worst_regime = ""
        best_time = ""

        if regime_bd:
            valid_regimes = [r for r in regime_bd if r.count >= 5]  # min sample
            if valid_regimes:
                best = max(valid_regimes, key=lambda r: r.win_rate)
                worst = min(valid_regimes, key=lambda r: r.win_rate)
                best_regime = best.regime
                worst_regime = worst.regime

        if time_bd:
            valid_times = [t for t in time_bd if t.count >= 5]
            if valid_times:
                best_t = max(valid_times, key=lambda t: t.win_rate)
                best_time = best_t.slot

        # Build recommendations
        regime_rec = self._regime_recommendation(regime_bd)
        time_rec = self._time_recommendation(time_bd)

        return CorrelationReport(
            regime_breakdown=regime_bd,
            time_breakdown=time_bd,
            best_regime=best_regime,
            worst_regime=worst_regime,
            best_time_window=best_time,
            regime_recommendation=regime_rec,
            time_recommendation=time_rec,
        )

    # ── Regime breakdown ──────────────────────────────────────────────

    def _regime_breakdown(self, results: List[BacktestResult]) -> List[RegimeBreakdown]:
        """Group results by regime and compute per-regime stats."""
        by_regime: Dict[str, List[BacktestResult]] = defaultdict(list)
        for r in results:
            regime = r.regime_at_entry or "unknown"
            by_regime[regime].append(r)

        breakdowns = []
        for regime, regime_results in by_regime.items():
            r_vals = [r.r_multiple for r in regime_results]
            wins = [r for r in regime_results if r.won]
            losses = [r for r in regime_results if not r.won]
            gross_profit = sum(r.r_multiple for r in wins) if wins else 0
            gross_loss = abs(sum(r.r_multiple for r in losses)) if losses else 0.001

            win_rate = len(wins) / len(regime_results)
            avg_r = float(np.mean(r_vals))

            # Determine if it "works" or "fails" in this regime
            outcome = "works" if win_rate >= 0.55 else "fails"
            why_works = _REGIME_EXPLANATIONS.get((regime, "works"), "")
            why_fails = _REGIME_EXPLANATIONS.get((regime, "fails"), "")

            bd = RegimeBreakdown(
                regime=regime,
                count=len(regime_results),
                win_rate=round(win_rate, 3),
                avg_r=round(avg_r, 2),
                median_r=round(float(np.median(r_vals)), 2),
                profit_factor=round(gross_profit / gross_loss, 2),
                best_r=round(float(max(r_vals)), 2),
                worst_r=round(float(min(r_vals)), 2),
                why_works=why_works if outcome == "works" else "",
                why_fails=why_fails if outcome == "fails" else "",
            )
            breakdowns.append(bd)

        # Sort by count descending
        breakdowns.sort(key=lambda b: b.count, reverse=True)
        return breakdowns

    # ── Time-of-day breakdown ─────────────────────────────────────────

    def _time_breakdown(self, results: List[BacktestResult]) -> List[TimeSlotBreakdown]:
        """Group results by IST time-of-day slots."""
        slot_results: Dict[str, List[BacktestResult]] = defaultdict(list)

        for r in results:
            slot = self._classify_time_slot(r.time_of_day)
            slot_results[slot].append(r)

        breakdowns = []
        for slot_start, slot_end, slot_name in _TIME_SLOTS:
            slot_key = f"{slot_start}-{slot_end}"
            rs = slot_results.get(slot_key, [])
            if not rs:
                breakdowns.append(TimeSlotBreakdown(
                    slot=slot_key, count=0, win_rate=0, avg_r=0, label=slot_name,
                ))
                continue

            wins = sum(1 for r in rs if r.won)
            r_vals = [r.r_multiple for r in rs]
            win_rate = wins / len(rs)

            # Label based on win rate
            if win_rate >= 0.70:
                label = f"{slot_name} — GOLDEN WINDOW"
            elif win_rate >= 0.55:
                label = slot_name
            elif win_rate >= 0.45:
                label = f"{slot_name} — mixed"
            else:
                label = f"{slot_name} — AVOID"

            breakdowns.append(TimeSlotBreakdown(
                slot=slot_key,
                count=len(rs),
                win_rate=round(win_rate, 3),
                avg_r=round(float(np.mean(r_vals)), 2),
                label=label,
            ))

        return breakdowns

    def _classify_time_slot(self, time_str: str) -> str:
        """Map a time string (HH:MM) to the appropriate slot."""
        if not time_str:
            return "09:15-10:00"  # default to opening

        try:
            hh, mm = map(int, time_str.split(":"))
            minutes = hh * 60 + mm

            for slot_start, slot_end, _ in _TIME_SLOTS:
                sh, sm = map(int, slot_start.split(":"))
                eh, em = map(int, slot_end.split(":"))
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                if start_min <= minutes < end_min:
                    return f"{slot_start}-{slot_end}"

            return "14:30-15:30"  # after hours → power hour
        except (ValueError, AttributeError):
            return "09:15-10:00"

    # ── Recommendations ───────────────────────────────────────────────

    def _regime_recommendation(self, breakdowns: List[RegimeBreakdown]) -> str:
        """Generate human-readable regime sizing recommendation."""
        parts = []
        for bd in breakdowns:
            if bd.count < 5:
                continue
            if bd.win_rate >= 0.65:
                parts.append(f"{bd.regime}: FULL SIZE (WR {bd.win_rate:.0%})")
            elif bd.win_rate >= 0.55:
                parts.append(f"{bd.regime}: STANDARD SIZE (WR {bd.win_rate:.0%})")
            elif bd.win_rate >= 0.45:
                parts.append(f"{bd.regime}: REDUCE SIZE (WR {bd.win_rate:.0%})")
            else:
                parts.append(f"{bd.regime}: SKIP (WR {bd.win_rate:.0%})")
        return "; ".join(parts) if parts else "Insufficient data for regime recommendation"

    def _time_recommendation(self, breakdowns: List[TimeSlotBreakdown]) -> str:
        """Generate human-readable time-of-day recommendation."""
        golden = [t for t in breakdowns if "GOLDEN" in t.label]
        avoid = [t for t in breakdowns if "AVOID" in t.label]

        parts = []
        if golden:
            parts.append(f"Best window: {', '.join(t.slot for t in golden)}")
        if avoid:
            parts.append(f"Avoid: {', '.join(t.slot for t in avoid)}")

        return "; ".join(parts) if parts else "No strong time-of-day bias detected"
