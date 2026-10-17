"""Risk analytics — production implementation.

Quant review findings addressed:
  1. File was an empty placeholder.
  2. Missing Sharpe ratio, Sortino ratio, Calmar ratio.
  3. Missing Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
  4. Missing Risk of Ruin calculation.
  5. Missing profit factor and expectancy per strategy.
  6. No volatility-adjusted performance attribution.

All calculations follow CFA / GIPS / industry-standard definitions.
Period conventions:
  - Daily returns → periods_per_year = 252
  - Trade returns → periods_per_year = number of trades per year (use n_trades)
  - Use the same convention consistently within a session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskMetrics:
    """Full risk-adjusted performance summary for a strategy or account."""
    # Returns
    total_return_pct: float
    annualised_return_pct: float
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Drawdown
    max_drawdown_pct: float
    avg_drawdown_pct: float
    # Tail risk
    var_95: float           # 5% VaR (daily $ loss exceeded 5% of time)
    var_99: float           # 1% VaR
    cvar_95: float          # Expected Shortfall at 95%
    # Trading edge
    profit_factor: float
    expectancy_r: float     # Expected R per trade
    risk_of_ruin: float     # Probability of 50% account loss
    # Win/loss
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    n_trades: int


# ──────────────────────────────────────────────────────────────────────────────
#  RISK ANALYTICS ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    """
    Volatility-adjusted performance and risk analytics.

    Designed to be used both for historical backtests and for live
    rolling performance windows (last 30/60/90 trades).
    """

    # ── Sharpe Ratio ──────────────────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(
        returns: Sequence[float],
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """
        Sharpe ratio: (mean_return - risk_free) / std_return, annualised.

        Args:
            returns: Sequence of periodic returns (fractional, e.g. 0.01 = 1%).
            risk_free_rate: Annual risk-free rate (e.g. 0.05 = 5%).
            periods_per_year: 252 for daily, 52 weekly, n_trades for trade-based.

        Returns:
            Annualised Sharpe ratio.  > 1.0 good, > 2.0 excellent.
        """
        r = np.array(returns, dtype=float)
        if len(r) < 2:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        excess = r - rf_per_period
        std = np.std(excess, ddof=1)
        if std < 1e-10:
            return 0.0
        return float((np.mean(excess) / std) * math.sqrt(periods_per_year))

    # ── Sortino Ratio ─────────────────────────────────────────────────────

    @staticmethod
    def sortino_ratio(
        returns: Sequence[float],
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
        mar: Optional[float] = None,
    ) -> float:
        """
        Sortino ratio: (mean_return - MAR) / downside_deviation, annualised.

        Sortino is superior to Sharpe for traders because it only penalises
        downside volatility — upside vol (big wins) does NOT hurt the score.

        Args:
            mar: Minimum Acceptable Return per period (defaults to risk_free/periods).

        Returns:
            Annualised Sortino ratio.
        """
        r = np.array(returns, dtype=float)
        if len(r) < 2:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        threshold = mar if mar is not None else rf_per_period
        downside = r[r < threshold] - threshold
        if len(downside) == 0:
            return float("inf")
        downside_std = math.sqrt(np.mean(downside ** 2))
        if downside_std < 1e-10:
            return 0.0
        excess_return = np.mean(r) - threshold
        return float((excess_return / downside_std) * math.sqrt(periods_per_year))

    # ── Calmar Ratio ──────────────────────────────────────────────────────

    @staticmethod
    def calmar_ratio(annual_return: float, max_drawdown_pct: float) -> float:
        """
        Calmar = annualised_return / max_drawdown.

        A simple and widely-used risk-adjusted metric favoured by prop firms.
        Typical targets: > 2.0 good, > 3.0 excellent.
        """
        if max_drawdown_pct < 1e-8:
            return float("inf")
        return float(annual_return / max_drawdown_pct)

    # ── Value at Risk ─────────────────────────────────────────────────────

    @staticmethod
    def var_historical(
        returns: Sequence[float],
        confidence: float = 0.95,
        account_value: float = 1.0,
    ) -> float:
        """
        Historical simulation VaR.

        Returns the $ (or fractional) loss that will not be exceeded with
        `confidence` probability.  Negative return = a loss.

        Historical simulation is preferred over parametric VaR for trading
        because real P&L distributions have fat tails that normality ignores.
        """
        r = np.array(returns, dtype=float)
        if len(r) < 20:
            return 0.0
        cutoff = np.percentile(r, (1 - confidence) * 100)
        return float(abs(cutoff) * account_value)

    @staticmethod
    def cvar(
        returns: Sequence[float],
        confidence: float = 0.95,
        account_value: float = 1.0,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall) — the average loss BEYOND VaR.

        CVaR is the standard tail risk measure for professional risk management.
        It answers: 'Given that a bad day happens, how bad is it on average?'
        """
        r = np.array(returns, dtype=float)
        if len(r) < 20:
            return 0.0
        cutoff = np.percentile(r, (1 - confidence) * 100)
        tail = r[r <= cutoff]
        if len(tail) == 0:
            return 0.0
        return float(abs(np.mean(tail)) * account_value)

    # ── Risk of Ruin ─────────────────────────────────────────────────────

    @staticmethod
    def risk_of_ruin(
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        risk_pct_per_trade: float,
        ruin_level: float = 0.50,
        n_simulations: int = 10_000,
        n_trades: int = 1_000,
    ) -> float:
        """
        Probability of account dropping to ruin_level (e.g. 50% of starting equity).

        Uses Monte Carlo simulation (default 10k paths) because the closed-form
        formula breaks down for variable R-multiples.  For a quick closed-form
        estimate, use `risk_of_ruin_formula` instead.

        Args:
            win_rate: Historical win rate (0–1).
            avg_win_r: Average winner as R-multiple.
            avg_loss_r: Average loser as R-multiple (typically ~1.0 with fixed stop).
            risk_pct_per_trade: Fraction of account risked per trade (e.g. 0.01 = 1%).
            ruin_level: Account fraction below which trader is 'ruined' (0.5 = 50% loss).
            n_simulations: Monte Carlo paths.
            n_trades: Trade horizon.

        Returns:
            Probability of ruin (0–1).
        """
        rng = np.random.default_rng(seed=42)
        ruin_count = 0
        ruin_threshold = 1.0 - ruin_level  # fraction of starting equity at ruin

        for _ in range(n_simulations):
            equity = 1.0
            ruined = False
            for _ in range(n_trades):
                if equity <= ruin_threshold:
                    ruined = True
                    break
                # Simulate trade outcome
                if rng.random() < win_rate:
                    equity *= (1 + risk_pct_per_trade * avg_win_r)
                else:
                    equity *= (1 - risk_pct_per_trade * avg_loss_r)
            if ruined or equity <= ruin_threshold:
                ruin_count += 1

        return float(ruin_count / n_simulations)

    @staticmethod
    def risk_of_ruin_formula(
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> float:
        """
        Closed-form Risk of Ruin approximation (Ralph Vince formula).

        RoR = ((1 - A) / (1 + A)) ^ N
        where A = edge ratio = (p*b - q) / (p*b + q)
              b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate

        Less precise than Monte Carlo but instant — useful for quick UI display.
        Returns probability 0–1 of eventual ruin (infinite time horizon).
        """
        if avg_win_r <= 0 or avg_loss_r <= 0:
            return 1.0
        b = avg_win_r / avg_loss_r
        p = win_rate
        q = 1.0 - p
        edge = (p * b - q) / (p * b + q)
        if edge <= 0:
            return 1.0  # Negative edge → guaranteed ruin eventually
        # RoR approaches 0 as edge → 1; we return the formula for 1 unit of risk capital
        return float(((1 - edge) / (1 + edge)))

    # ── Trading Edge Metrics ──────────────────────────────────────────────

    @staticmethod
    def profit_factor(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> float:
        """
        Profit Factor = gross_profit / gross_loss.

        > 1.5 is considered healthy.  < 1.0 is losing system.
        """
        if trades_df.empty or pnl_col not in trades_df.columns:
            return 0.0
        pnl = trades_df[pnl_col]
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        if gross_loss < 1e-8:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def expectancy(
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
    ) -> float:
        """
        Expected R per trade: E = p * avg_win_r - (1-p) * avg_loss_r.

        Positive expectancy is the minimum requirement for a viable strategy.
        A score of 0.2R means you expect to make 0.2 × your risk per trade.
        """
        return float(win_rate * avg_win_r - (1 - win_rate) * avg_loss_r)

    @staticmethod
    def r_multiples(trades_df: pd.DataFrame) -> pd.Series:
        """
        Compute R-multiple for each trade.

        R = pnl / initial_risk_amount.  Requires 'pnl' and 'risk_amount' columns.
        """
        if trades_df.empty:
            return pd.Series(dtype=float)
        required = {"pnl", "risk_amount"}
        if not required.issubset(trades_df.columns):
            return pd.Series(dtype=float)
        risk = trades_df["risk_amount"].replace(0, np.nan)
        return (trades_df["pnl"] / risk).dropna()

    # ── Rolling analytics ─────────────────────────────────────────────────

    @staticmethod
    def rolling_sharpe(
        returns: Sequence[float],
        window: int = 20,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """Rolling Sharpe over `window` trades — detect edge degradation."""
        r = pd.Series(returns, dtype=float)
        rf = risk_free_rate / periods_per_year
        excess = r - rf
        roll_mean = excess.rolling(window).mean()
        roll_std = excess.rolling(window).std(ddof=1)
        return (roll_mean / roll_std.replace(0, np.nan)) * math.sqrt(periods_per_year)

    @staticmethod
    def rolling_win_rate(r_series: pd.Series, window: int = 20) -> pd.Series:
        """Rolling win rate over `window` trades — detect regime shifts."""
        wins = (r_series > 0).astype(float)
        return wins.rolling(window).mean()

    # ── Full report ────────────────────────────────────────────────────────

    def full_report(
        self,
        trades_df: pd.DataFrame,
        equity: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> RiskMetrics:
        """
        Produce a complete RiskMetrics object from a trades DataFrame.

        Required columns: pnl, risk_amount
        Optional: entry_price, exit_price, stop_loss, direction
        """
        if trades_df.empty:
            return RiskMetrics(
                total_return_pct=0, annualised_return_pct=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown_pct=0, avg_drawdown_pct=0,
                var_95=0, var_99=0, cvar_95=0,
                profit_factor=0, expectancy_r=0, risk_of_ruin=1.0,
                win_rate=0, avg_win_r=0, avg_loss_r=0, n_trades=0,
            )

        pnl = trades_df.get("pnl", pd.Series(dtype=float))
        r_series = self.r_multiples(trades_df)

        wins = r_series[r_series > 0]
        losses = r_series[r_series < 0]
        win_rate = len(wins) / len(r_series) if len(r_series) > 0 else 0.0
        avg_win_r = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss_r = float(abs(losses.mean())) if len(losses) > 0 else 0.0

        # Fractional returns per trade (for Sharpe/Sortino)
        if "risk_amount" in trades_df.columns and "account_balance_at_entry" in trades_df.columns:
            frac_returns = (pnl / trades_df["account_balance_at_entry"]).fillna(0).tolist()
        elif equity is not None and len(equity) > 1:
            frac_returns = equity.pct_change().dropna().tolist()
        else:
            # Fallback: treat R-multiples as returns (scaled by 1% risk)
            frac_returns = (r_series * 0.01).tolist()

        sharpe = self.sharpe_ratio(frac_returns, risk_free_rate, periods_per_year)
        sortino = self.sortino_ratio(frac_returns, risk_free_rate, periods_per_year)

        # Drawdown metrics
        if equity is not None and len(equity) > 1:
            from analytics.drawdown_analyzer import DrawdownAnalyzer
            dd_analyzer = DrawdownAnalyzer()
            max_dd, _ = dd_analyzer.max_drawdown(equity)
            dd_series = dd_analyzer.compute_series(equity)
            avg_dd = float(abs(dd_series["drawdown_pct"].mean()))
            total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] > 0 else 0.0
            n_years = len(equity) / periods_per_year
            ann_ret = (1 + total_ret) ** (1 / max(n_years, 1e-8)) - 1 if total_ret > -1 else -1.0
        else:
            max_dd = 0.0
            avg_dd = 0.0
            ann_ret = 0.0
            total_ret = float(pnl.sum())

        calmar = self.calmar_ratio(ann_ret, max_dd) if max_dd > 0 else 0.0
        pf = self.profit_factor(trades_df)
        exp_r = self.expectancy(win_rate, avg_win_r, avg_loss_r)
        ror_formula = self.risk_of_ruin_formula(win_rate, avg_win_r, avg_loss_r) if avg_win_r > 0 else 1.0

        var_95 = self.var_historical(frac_returns, 0.95)
        var_99 = self.var_historical(frac_returns, 0.99)
        cvar_95 = self.cvar(frac_returns, 0.95)

        return RiskMetrics(
            total_return_pct=round(total_ret * 100, 2),
            annualised_return_pct=round(ann_ret * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3) if math.isfinite(calmar) else 0.0,
            max_drawdown_pct=round(max_dd * 100, 2),
            avg_drawdown_pct=round(avg_dd * 100, 2),
            var_95=round(var_95, 4),
            var_99=round(var_99, 4),
            cvar_95=round(cvar_95, 4),
            profit_factor=round(pf, 3),
            expectancy_r=round(exp_r, 4),
            risk_of_ruin=round(ror_formula, 4),
            win_rate=round(win_rate, 4),
            avg_win_r=round(avg_win_r, 3),
            avg_loss_r=round(avg_loss_r, 3),
            n_trades=len(trades_df),
        )

    def strategy_report(
        self,
        trades_df: pd.DataFrame,
        strategy_col: str = "strategy_id",
        equity: Optional[pd.Series] = None,
    ) -> Dict[str, RiskMetrics]:
        """Produce per-strategy RiskMetrics breakdown."""
        if trades_df.empty or strategy_col not in trades_df.columns:
            return {}
        results = {}
        for sid, group in trades_df.groupby(strategy_col):
            results[str(sid)] = self.full_report(group.reset_index(drop=True))
        return results
