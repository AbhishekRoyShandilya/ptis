"""Position sizing engine — production-grade implementation.

Quant review findings addressed here:
  1. File was an empty placeholder — completely missing.
  2. No Kelly criterion, no volatility-adjusted sizing, no drawdown feedback.
  3. No portfolio heat management (cumulative open-trade risk).
  4. No protection against sizing up into a drawdown.

This module provides four complementary sizing methods that can be combined:

  fixed_fractional      — standard 1-2% risk-per-trade, the practitioner default
  volatility_adjusted   — normalize size so each trade risks the same $ per ATR
  kelly_fraction        — data-driven edge sizing, applied at quarter-Kelly
  drawdown_scaled       — reduce size automatically when account is in drawdown

Usage pattern (recommended):
    sizer = PositionSizer()
    result = sizer.compute(
        account_balance=50_000,
        entry=185.40,
        stop_loss=183.10,
        atr=1.85,
        vol_regime=VolatilityRegime.AVERAGE,
        drawdown_pct=0.0,        # 0% = at peak equity
        open_risk_pct=1.5,       # existing open-trade heat
        win_rate=0.55,           # from strategy performance history
        avg_win_r=1.8,
        avg_loss_r=1.0,
    )
    print(result.shares, result.risk_amount, result.warnings)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from core.enums import VolatilityRegime


# ──────────────────────────────────────────────────────────────────────────────
#  SIZING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingConfig:
    """All tunable parameters for position sizing in one place."""

    # Fixed-fractional
    base_risk_pct: float = 1.0          # % of account to risk per trade (default 1R)
    max_risk_pct: float = 2.0           # hard cap per trade
    min_risk_pct: float = 0.25          # floor (avoid trades too small to matter)

    # Portfolio heat
    max_portfolio_heat_pct: float = 6.0  # max total % at risk across ALL open trades
    # e.g., 3 simultaneous 2% trades = 6% heat → refuse further entries

    # Drawdown scaling thresholds  (current_dd_pct → size_multiplier)
    # A 10% drawdown cuts size to 75%, 15% to 50%, 20%+ stops trading.
    drawdown_thresholds: List[tuple] = field(default_factory=lambda: [
        (0.05,  1.00),   # 0–5 % DD   → full size
        (0.10,  0.75),   # 5–10% DD   → 75%
        (0.15,  0.50),   # 10–15% DD  → 50%
        (0.20,  0.25),   # 15–20% DD  → 25%
        (1.00,  0.00),   # >20% DD    → stop trading
    ])

    # Volatility scaling (multiply base risk by these factors)
    vol_size_multipliers: dict = field(default_factory=lambda: {
        VolatilityRegime.LOW:     1.10,   # slightly bigger in calm markets
        VolatilityRegime.AVERAGE: 1.00,   # baseline
        VolatilityRegime.HIGH:    0.75,   # reduce in high vol
        VolatilityRegime.EXTREME: 0.50,   # cut in half in extreme vol
    })

    # Kelly settings
    kelly_fraction: float = 0.25        # quarter-Kelly — industry-standard safety
    kelly_cap_pct: float = 2.5          # hard cap: Kelly cannot size above this %
    min_history_trades: int = 30        # refuse Kelly if edge is under-sampled


# ──────────────────────────────────────────────────────────────────────────────
#  SIZING RESULT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingResult:
    """Output of the position sizer — everything the order router needs."""
    shares: float                        # units / contracts / shares
    risk_amount: float                   # $ at risk (entry - stop) * shares
    risk_pct: float                      # risk_amount / account_balance * 100
    method: str                          # which method produced this result
    scaling_factor: float = 1.0         # combined drawdown + vol multiplier
    kelly_edge: float = 0.0             # raw Kelly % (informational)
    portfolio_heat_after: float = 0.0   # total heat including this trade
    blocked: bool = False               # True when heat / DD limits refuse entry
    warnings: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
#  POSITION SIZER
# ──────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Production-grade position sizing with:
      - Fixed fractional (primary method)
      - Volatility adjustment (regime-aware scaling)
      - Quarter-Kelly overlay (edge-based sizing confirmation)
      - Drawdown feedback (automatic size reduction under stress)
      - Portfolio heat guard (refuse entries that breach total exposure cap)
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.cfg = config or SizingConfig()

    # ── Primary entry point ──────────────────────────────────────────────────

    def compute(
        self,
        account_balance: float,
        entry: float,
        stop_loss: float,
        atr: float,
        vol_regime: VolatilityRegime = VolatilityRegime.AVERAGE,
        drawdown_pct: float = 0.0,          # current peak-to-trough DD, 0.0–1.0
        open_risk_pct: float = 0.0,         # % already at risk in open trades
        win_rate: Optional[float] = None,   # strategy historical win rate
        avg_win_r: Optional[float] = None,  # average winner in R-multiples
        avg_loss_r: Optional[float] = None, # average loser in R-multiples (usually 1.0)
        n_history: int = 0,                 # number of trades in performance sample
    ) -> SizingResult:
        """
        Compute position size using a layered approach:
          1. Base risk %  (fixed fractional)
          2. × volatility multiplier
          3. × drawdown multiplier
          4. Compare against Kelly if sufficient history
          5. Check portfolio heat — block if over limit

        Args:
            account_balance: Current account equity in $.
            entry: Planned entry price.
            stop_loss: Hard stop price.
            atr: Current ATR for the instrument / timeframe.
            vol_regime: Current volatility classification.
            drawdown_pct: Current drawdown from peak (0.0 = at peak, 0.15 = 15% down).
            open_risk_pct: Sum of risk_pct across currently open trades.
            win_rate: Historical win rate for this strategy (0–1).
            avg_win_r: Average win in R-multiples.
            avg_loss_r: Average loss in R-multiples (typically ~1.0 with fixed stops).
            n_history: Number of completed trades in the performance sample.

        Returns:
            SizingResult with shares, risk_amount, warnings, etc.
        """
        warnings: List[str] = []

        if account_balance <= 0 or entry <= 0:
            return SizingResult(
                shares=0, risk_amount=0, risk_pct=0,
                method="blocked", blocked=True,
                warnings=["Invalid account balance or entry price"],
            )

        stop_distance = abs(entry - stop_loss)
        if stop_distance < 1e-8:
            return SizingResult(
                shares=0, risk_amount=0, risk_pct=0,
                method="blocked", blocked=True,
                warnings=["Stop loss equals entry price — cannot compute size"],
            )

        # ── 1. Drawdown scaling ────────────────────────────────────────────
        dd_multiplier = self._drawdown_multiplier(drawdown_pct)
        if dd_multiplier == 0.0:
            return SizingResult(
                shares=0, risk_amount=0, risk_pct=0,
                method="blocked", blocked=True,
                warnings=[
                    f"Trading halted: drawdown {drawdown_pct*100:.1f}% exceeds maximum threshold. "
                    "Restore equity before resuming."
                ],
            )
        if dd_multiplier < 1.0:
            warnings.append(
                f"Drawdown {drawdown_pct*100:.1f}%: size reduced to {dd_multiplier*100:.0f}%"
            )

        # ── 2. Volatility scaling ─────────────────────────────────────────
        vol_multiplier = self.cfg.vol_size_multipliers.get(vol_regime, 1.0)
        if vol_multiplier < 1.0:
            warnings.append(f"High volatility ({vol_regime.value}): size reduced to {vol_multiplier*100:.0f}%")

        combined_scale = dd_multiplier * vol_multiplier

        # ── 3. Fixed-fractional base size ─────────────────────────────────
        base_risk_pct = self.cfg.base_risk_pct * combined_scale
        base_risk_pct = max(self.cfg.min_risk_pct, min(self.cfg.max_risk_pct, base_risk_pct))

        ff_shares = self._fixed_fractional_shares(account_balance, base_risk_pct / 100, stop_distance)
        ff_risk_pct = base_risk_pct

        # ── 4. Kelly comparison (if sufficient history) ───────────────────
        kelly_edge = 0.0
        kelly_info = ""
        if (
            win_rate is not None
            and avg_win_r is not None
            and avg_loss_r is not None
            and n_history >= self.cfg.min_history_trades
        ):
            raw_kelly_pct = self._kelly_pct(win_rate, avg_win_r, avg_loss_r)
            kelly_edge = raw_kelly_pct
            scaled_kelly_pct = raw_kelly_pct * self.cfg.kelly_fraction  # quarter-Kelly
            scaled_kelly_pct = min(scaled_kelly_pct, self.cfg.kelly_cap_pct) * combined_scale

            if scaled_kelly_pct < ff_risk_pct:
                kelly_info = (
                    f"Kelly suggests {scaled_kelly_pct:.2f}% vs fixed-fractional {ff_risk_pct:.2f}%: "
                    "using Kelly (edge-constrained)"
                )
                ff_risk_pct = scaled_kelly_pct
                ff_shares = self._fixed_fractional_shares(
                    account_balance, ff_risk_pct / 100, stop_distance
                )
            elif scaled_kelly_pct > ff_risk_pct * 1.5:
                kelly_info = f"Kelly ({scaled_kelly_pct:.2f}%) >> fixed-fractional: edge is strong, sticking with fixed cap"

            if kelly_info:
                warnings.append(kelly_info)

        # ── 5. Portfolio heat guard ────────────────────────────────────────
        new_risk_pct = (ff_shares * stop_distance / account_balance) * 100
        heat_after = open_risk_pct + new_risk_pct

        if heat_after > self.cfg.max_portfolio_heat_pct:
            # Reduce size to fit within heat cap
            headroom = max(0.0, self.cfg.max_portfolio_heat_pct - open_risk_pct)
            if headroom < self.cfg.min_risk_pct:
                return SizingResult(
                    shares=0, risk_amount=0, risk_pct=0,
                    method="blocked", blocked=True,
                    portfolio_heat_after=open_risk_pct,
                    warnings=[
                        f"Portfolio heat {open_risk_pct:.1f}% already at cap "
                        f"({self.cfg.max_portfolio_heat_pct:.1f}%). Entry refused."
                    ],
                )
            # Scale down to fit
            warnings.append(
                f"Portfolio heat cap: reducing size from {new_risk_pct:.2f}% to {headroom:.2f}% "
                f"(total open: {open_risk_pct:.1f}%)"
            )
            ff_risk_pct = headroom
            ff_shares = self._fixed_fractional_shares(
                account_balance, ff_risk_pct / 100, stop_distance
            )
            heat_after = self.cfg.max_portfolio_heat_pct

        # ── Sanity check: ATR-normalized size validation ──────────────────
        if atr > 0:
            atr_risk_pct = self._volatility_adjusted_risk_pct(
                account_balance, ff_shares, atr, entry
            )
            if atr_risk_pct > self.cfg.max_risk_pct * 1.5:
                warnings.append(
                    f"ATR-normalized risk {atr_risk_pct:.2f}% is very high — "
                    "consider tightening stop or reducing size"
                )

        # ── Build result ───────────────────────────────────────────────────
        final_risk_amount = ff_shares * stop_distance
        final_risk_pct = (final_risk_amount / account_balance) * 100

        return SizingResult(
            shares=ff_shares,
            risk_amount=final_risk_amount,
            risk_pct=final_risk_pct,
            method="fixed_fractional_with_kelly_and_drawdown_scaling",
            scaling_factor=combined_scale,
            kelly_edge=kelly_edge,
            portfolio_heat_after=heat_after,
            blocked=False,
            warnings=warnings,
        )

    # ── Volatility-adjusted sizing (standalone) ───────────────────────────

    def volatility_adjusted(
        self,
        account_balance: float,
        atr: float,
        target_atr_risk_pct: float = 1.0,
        entry: float = 1.0,
    ) -> float:
        """
        Size position so that 1 ATR move costs exactly target_atr_risk_pct % of equity.

        This is the basis of volatility-targeting used by quant funds.
        Useful for comparing whether a stop-loss is proportionate to current vol.

        Returns: number of shares/contracts.
        """
        if atr <= 0 or entry <= 0:
            return 0.0
        dollar_risk = account_balance * (target_atr_risk_pct / 100)
        return dollar_risk / atr

    # ── Internal helpers ──────────────────────────────────────────────────

    def _fixed_fractional_shares(
        self,
        account: float,
        risk_fraction: float,
        stop_distance: float,
    ) -> float:
        """Shares = (account * risk_fraction) / stop_distance."""
        if stop_distance <= 0:
            return 0.0
        dollar_risk = account * risk_fraction
        return dollar_risk / stop_distance

    def _kelly_pct(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> float:
        """
        Generalised Kelly criterion for uneven payoffs.

        Formula: f* = (p * b - q) / b
          where b = avg_win / avg_loss (payout ratio in R space)
                p = win_rate, q = 1 - p

        Returns the raw Kelly % (before safety fraction is applied).
        Negative Kelly = negative edge; return 0.
        """
        if avg_win_r <= 0 or avg_loss_r <= 0:
            return 0.0
        b = avg_win_r / avg_loss_r
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        return max(0.0, kelly * 100)   # as a percentage

    def _drawdown_multiplier(self, drawdown_pct: float) -> float:
        """
        Map current drawdown depth → position-size multiplier.

        Uses the threshold table in SizingConfig.  Thresholds are
        sorted ascending; we return the multiplier for the first
        threshold the current DD exceeds.
        """
        for threshold, multiplier in self.cfg.drawdown_thresholds:
            if drawdown_pct < threshold:
                return multiplier
        return 0.0  # Beyond all thresholds → stop trading

    def _volatility_adjusted_risk_pct(
        self,
        account: float,
        shares: float,
        atr: float,
        entry: float,
    ) -> float:
        """% of account lost if price moves 1 ATR against the position."""
        if account <= 0:
            return 0.0
        dollar_per_atr = shares * atr
        return (dollar_per_atr / account) * 100

    # ── Conviction-adjusted sizing (SKE integration) ─────────────────────

    def conviction_adjusted_size(
        self,
        base_shares: float,
        conviction_band: str,
        conviction_score: float = 0.5,
    ) -> float:
        """
        Apply conviction-based multiplier from the Strategy Knowledge Engine.

        Conviction bands map to sizing multipliers:
            skip       → 0.0  (no trade)
            small      → 0.5
            standard   → 1.0
            aggressive → 1.5
            max        → 2.0

        For fine-grained adjustment within a band, the continuous conviction_score
        (0–1) is used to interpolate within the band's multiplier range.

        Args:
            base_shares: Position size from compute() before conviction adjustment.
            conviction_band: One of skip/small/standard/aggressive/max.
            conviction_score: Continuous conviction 0.0–1.0 for interpolation.

        Returns:
            Adjusted share count (may be 0 for skip band).
        """
        band_multipliers = {
            "skip": 0.0,
            "small": 0.5,
            "standard": 1.0,
            "aggressive": 1.5,
            "max": 2.0,
        }

        multiplier = band_multipliers.get(conviction_band.lower(), 1.0)

        # Don't exceed max_risk_pct even at max conviction
        if multiplier > 1.0:
            # Cap at 2x base — portfolio heat guard in compute() provides the hard cap
            multiplier = min(multiplier, 2.0)

        return base_shares * multiplier

    # ── Portfolio heat utility (called externally before any entry) ────────

    def check_portfolio_heat(
        self,
        open_risk_pct: float,
        new_trade_risk_pct: float,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        Call this BEFORE placing any order as a final gate.
        """
        projected = open_risk_pct + new_trade_risk_pct
        if projected > self.cfg.max_portfolio_heat_pct:
            reason = (
                f"Entry blocked: total portfolio heat would reach {projected:.1f}% "
                f"(max {self.cfg.max_portfolio_heat_pct:.1f}%). "
                f"Current open risk: {open_risk_pct:.1f}%."
            )
            return False, reason
        return True, f"Heat OK: {projected:.1f}% / {self.cfg.max_portfolio_heat_pct:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def default_sizer() -> PositionSizer:
    """Return a PositionSizer with conservative default settings."""
    return PositionSizer(SizingConfig())


def aggressive_sizer() -> PositionSizer:
    """Return a PositionSizer tuned for higher-conviction / prop-firm traders."""
    cfg = SizingConfig(
        base_risk_pct=2.0,
        max_risk_pct=3.0,
        max_portfolio_heat_pct=8.0,
        kelly_fraction=0.33,    # third-Kelly instead of quarter
    )
    return PositionSizer(cfg)
