"""
Instrument Specifications Registry
====================================

Central registry of tradeable instruments with their lot sizes, point values,
typical margin requirements, and exchange details.

This is intentionally code-first (no DB) so it deploys without migrations and
can be overridden at runtime via user-defined config.  Broker APIs should feed
real-time margin data into `InstrumentSpec.margin_per_lot_inr` where available.

Why this matters
----------------
Indian retail F&O traders are uniquely exposed to the "capital constraint
problem": minimum lot sizes force a risk floor that can far exceed the trader's
target risk percentage.

Example — ₹20,000 capital on Nifty:
  1 lot = 75 units, 1pt = ₹75
  At a 50-point stop: risk per lot = 50 × 75 = ₹3,750 = 18.75% of capital
  Target risk 1%: you would need ₹3,75,000 capital

The system MUST understand this distinction:
  ● ₹20k trader taking 18.75% is NOT bad discipline — it is a capital constraint
  ● Coaching output should address the constraint, not blame the trader

Indian lot sizes as of Q1 2026 (post SEBI Oct-2024 revision)
------------------------------------------------------------
  Nifty 50 F&O         :  75 units / lot   (revised from 50, Nov 2024)
  Bank Nifty F&O       :  30 units / lot   (revised from 25, Mar 2024)
  Fin Nifty F&O        :  65 units / lot
  Midcap Select F&O    :  75 units / lot
  Nifty Next 50 F&O    :  25 units / lot
  Nifty IT F&O         :  35 units / lot
  Sensex Futures (BSE) :  20 units / lot
  BankEx Futures (BSE) :  30 units / lot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
#  ENUMERATIONS
# ──────────────────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    EQUITY          = "equity"
    INDEX_FUTURES   = "index_futures"
    EQUITY_FUTURES  = "equity_futures"
    INDEX_OPTIONS   = "index_options"
    EQUITY_OPTIONS  = "equity_options"
    CURRENCY        = "currency"
    COMMODITY       = "commodity"
    ETF             = "etf"


class Exchange(str, Enum):
    NSE  = "NSE"
    BSE  = "BSE"
    NFO  = "NFO"   # NSE F&O segment
    BFO  = "BFO"   # BSE F&O segment
    MCX  = "MCX"   # Commodity
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CME  = "CME"


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"


# ──────────────────────────────────────────────────────────────────────────────
#  INSTRUMENT SPEC
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InstrumentSpec:
    """
    All capital-related properties for a single tradeable instrument.

    Point value definition
    ----------------------
    point_value = lot_size × per_unit_value

    For Indian index futures, per_unit_value = ₹1 per point per unit, so:
      Nifty:     point_value = 75 × 1 = ₹75  per point
      BankNifty: point_value = 30 × 1 = ₹30  per point
      Sensex:    point_value = 20 × 1 = ₹20  per point
    """
    symbol:          str
    name:            str
    asset_class:     AssetClass
    exchange:        Exchange
    currency:        Currency
    lot_size:        int          # minimum tradeable units / contract size

    # Value per 1-point move for the whole lot (lot_size × per_unit_value)
    point_value:     float        # INR or USD per point per lot

    # Approximate margin per lot (INR/USD).  Actual values vary with SPAN.
    # Use as a floor check: if capital < margin, cannot trade.
    margin_per_lot:  float

    # Sensible stop-loss distances in index/price points for typical setups.
    # Used when the trader hasn't specified a stop (for MVC calculations).
    typical_stop_tight:  float   # conservative / scalp stop
    typical_stop_normal: float   # standard swing stop
    typical_stop_wide:   float   # positional / trend stop

    # Tick size (minimum price increment)
    tick_size:       float = 0.05

    # For options: is risk capped at premium paid? (buyer perspective)
    risk_capped_at_premium: bool = False

    # Short description for UI
    description:     str = ""

    # ── Derived helpers ───────────────────────────────────────────

    def risk_per_lot(self, stop_points: float) -> float:
        """Capital at risk for 1 lot at given stop distance (in points)."""
        return stop_points * self.point_value

    def min_capital_for_risk_pct(self, stop_points: float, risk_pct: float) -> float:
        """
        Minimum capital required to keep risk at or below risk_pct%.

        Example:
            Nifty, 50-pt stop, 1% risk → 50 × 75 / 0.01 = ₹3,75,000
        """
        if risk_pct <= 0:
            return float("inf")
        return self.risk_per_lot(stop_points) / (risk_pct / 100)

    def actual_risk_pct(self, capital: float, stop_points: float) -> float:
        """Actual risk % for 1 lot at given capital and stop."""
        if capital <= 0:
            return float("inf")
        return (self.risk_per_lot(stop_points) / capital) * 100

    def min_capital_for_margin(self, lots: int = 1) -> float:
        """Capital needed just to meet margin (ignoring risk)."""
        return self.margin_per_lot * lots

    def max_lots_for_capital(self, capital: float) -> int:
        """Maximum lots affordable by margin alone."""
        if self.margin_per_lot <= 0:
            return 0
        return int(capital / self.margin_per_lot)

    def lots_for_risk(self, capital: float, stop_points: float, risk_pct: float) -> int:
        """
        Number of lots consistent with risk_pct% of capital.

        Returns 0 if even 1 lot would breach the risk limit.
        """
        if stop_points <= 0 or self.point_value <= 0:
            return 0
        risk_budget = capital * (risk_pct / 100)
        lots = int(risk_budget / self.risk_per_lot(stop_points))
        return max(0, lots)


# ──────────────────────────────────────────────────────────────────────────────
#  INSTRUMENT REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, InstrumentSpec] = {}


def _reg(spec: InstrumentSpec) -> InstrumentSpec:
    """Register an instrument under its symbol (and common aliases)."""
    _REGISTRY[spec.symbol.upper()] = spec
    return spec


# ── Indian Index F&O ─────────────────────────────────────────────────────────

NIFTY50 = _reg(InstrumentSpec(
    symbol          = "NIFTY",
    name            = "Nifty 50 Index Futures/Options",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 75,
    point_value     = 75.0,        # ₹75 per 1-point move
    margin_per_lot  = 100_000,     # ~₹1,00,000 SPAN margin (approximate)
    typical_stop_tight  = 30.0,   # scalper: 30 points
    typical_stop_normal = 60.0,   # swing: 60 points
    typical_stop_wide   = 100.0,  # positional: 100 points
    tick_size       = 0.05,
    description     = "NSE Nifty 50 — India's benchmark index F&O. "
                      "Lot size 75 (revised Nov 2024). ₹75 per point.",
))

# Also register common aliases
_REGISTRY["NIFTY50"] = NIFTY50
_REGISTRY["NIFTY 50"] = NIFTY50

BANKNIFTY = _reg(InstrumentSpec(
    symbol          = "BANKNIFTY",
    name            = "Bank Nifty Index Futures/Options",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 30,
    point_value     = 30.0,        # ₹30 per 1-point move
    margin_per_lot  = 55_000,     # ~₹55,000 SPAN margin
    typical_stop_tight  = 75.0,
    typical_stop_normal = 150.0,
    typical_stop_wide   = 300.0,
    tick_size       = 0.05,
    description     = "NSE Bank Nifty — high-volatility banking index. "
                      "Lot size 30 (revised Mar 2024). ₹30 per point.",
))
_REGISTRY["BANK NIFTY"] = BANKNIFTY
_REGISTRY["BANKNIFFY"] = BANKNIFTY

FINNIFTY = _reg(InstrumentSpec(
    symbol          = "FINNIFTY",
    name            = "Fin Nifty (Financial Services Index) F&O",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 65,
    point_value     = 65.0,        # ₹65 per point
    margin_per_lot  = 25_000,     # ~₹25,000 — much more capital-efficient
    typical_stop_tight  = 25.0,
    typical_stop_normal = 50.0,
    typical_stop_wide   = 100.0,
    tick_size       = 0.05,
    description     = "NSE Fin Nifty — financial services index. "
                      "Smaller margin (~₹25k) makes it accessible to ₹50k+ accounts.",
))
_REGISTRY["FIN NIFTY"] = FINNIFTY
_REGISTRY["FINANCIAL SERVICES"] = FINNIFTY

MIDCAPSELECT = _reg(InstrumentSpec(
    symbol          = "MIDCPNIFTY",
    name            = "Nifty Midcap Select Index F&O",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 75,
    point_value     = 75.0,
    margin_per_lot  = 120_000,
    typical_stop_tight  = 40.0,
    typical_stop_normal = 80.0,
    typical_stop_wide   = 150.0,
    tick_size       = 0.05,
    description     = "NSE Midcap Select Index — higher margin, higher volatility.",
))

NIFTYNEXT50 = _reg(InstrumentSpec(
    symbol          = "NIFTYNXT50",
    name            = "Nifty Next 50 Index F&O",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 25,
    point_value     = 25.0,
    margin_per_lot  = 80_000,
    typical_stop_tight  = 50.0,
    typical_stop_normal = 100.0,
    typical_stop_wide   = 200.0,
    tick_size       = 0.05,
    description     = "NSE Nifty Next 50 — smaller lot, lower notional.",
))

NIFTYIT = _reg(InstrumentSpec(
    symbol          = "NIFTYIT",
    name            = "Nifty IT Index F&O",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.NFO,
    currency        = Currency.INR,
    lot_size        = 35,
    point_value     = 35.0,
    margin_per_lot  = 60_000,
    typical_stop_tight  = 60.0,
    typical_stop_normal = 120.0,
    typical_stop_wide   = 250.0,
    tick_size       = 0.05,
    description     = "NSE Nifty IT Index — tech sector, 35 units/lot.",
))

SENSEX = _reg(InstrumentSpec(
    symbol          = "SENSEX",
    name            = "BSE Sensex Futures/Options",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.BFO,
    currency        = Currency.INR,
    lot_size        = 20,
    point_value     = 20.0,        # ₹20 per point — more capital-efficient
    margin_per_lot  = 55_000,
    typical_stop_tight  = 100.0,
    typical_stop_normal = 200.0,
    typical_stop_wide   = 400.0,
    tick_size       = 0.05,
    description     = "BSE Sensex — 20 units/lot, ₹20 per point. "
                      "Lower notional risk per lot vs Nifty.",
))

BANKEX = _reg(InstrumentSpec(
    symbol          = "BANKEX",
    name            = "BSE BankEx Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.BFO,
    currency        = Currency.INR,
    lot_size        = 30,
    point_value     = 30.0,
    margin_per_lot  = 50_000,
    typical_stop_tight  = 75.0,
    typical_stop_normal = 150.0,
    typical_stop_wide   = 300.0,
    tick_size       = 0.05,
    description     = "BSE BankEx — banking sector index on BSE.",
))

# ── Indian Equity (generic — lot size 1 for cash segment) ────────────────────

EQUITY_INDIA = _reg(InstrumentSpec(
    symbol          = "EQUITY_NSE",
    name            = "Indian Equity (Cash Segment)",
    asset_class     = AssetClass.EQUITY,
    exchange        = Exchange.NSE,
    currency        = Currency.INR,
    lot_size        = 1,           # 1 share minimum — fully flexible sizing
    point_value     = 1.0,         # ₹1 per ₹1 price move per share
    margin_per_lot  = 0,           # No margin needed for cash delivery
    typical_stop_tight  = 0.005,  # 0.5% of price
    typical_stop_normal = 0.015,  # 1.5% of price
    typical_stop_wide   = 0.03,   # 3.0% of price
    tick_size       = 0.05,
    description     = "NSE cash equity — lot size 1, no margin needed for delivery. "
                      "Best choice for very small capital (₹5,000–₹50,000).",
))

# ── US Futures ────────────────────────────────────────────────────────────────

ES = _reg(InstrumentSpec(
    symbol          = "ES",
    name            = "E-mini S&P 500 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,           # 1 contract
    point_value     = 50.0,        # $50 per point
    margin_per_lot  = 14_000,     # ~$14k initial margin
    typical_stop_tight  = 5.0,
    typical_stop_normal = 10.0,
    typical_stop_wide   = 20.0,
    tick_size       = 0.25,
    description     = "CME E-mini S&P 500 — $50/point. ~$14k margin per contract.",
))

MES = _reg(InstrumentSpec(
    symbol          = "MES",
    name            = "Micro E-mini S&P 500 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 5.0,         # $5 per point (1/10th of ES)
    margin_per_lot  = 1_400,      # ~$1.4k margin
    typical_stop_tight  = 5.0,
    typical_stop_normal = 10.0,
    typical_stop_wide   = 20.0,
    tick_size       = 0.25,
    description     = "CME Micro E-mini S&P 500 — $5/point. Ideal for <$25k accounts. "
                      "1/10th the size of ES — same market, fraction of capital.",
))

NQ = _reg(InstrumentSpec(
    symbol          = "NQ",
    name            = "E-mini Nasdaq 100 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 20.0,        # $20 per point
    margin_per_lot  = 20_000,
    typical_stop_tight  = 20.0,
    typical_stop_normal = 50.0,
    typical_stop_wide   = 100.0,
    tick_size       = 0.25,
    description     = "CME E-mini Nasdaq 100 — $20/point. High volatility.",
))

MNQ = _reg(InstrumentSpec(
    symbol          = "MNQ",
    name            = "Micro E-mini Nasdaq 100 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 2.0,         # $2 per point (1/10th of NQ)
    margin_per_lot  = 2_000,
    typical_stop_tight  = 20.0,
    typical_stop_normal = 50.0,
    typical_stop_wide   = 100.0,
    tick_size       = 0.25,
    description     = "CME Micro Nasdaq 100 — $2/point. Accessible for small accounts.",
))

RTY = _reg(InstrumentSpec(
    symbol          = "RTY",
    name            = "E-mini Russell 2000 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 50.0,
    margin_per_lot  = 7_000,
    typical_stop_tight  = 5.0,
    typical_stop_normal = 10.0,
    typical_stop_wide   = 20.0,
    tick_size       = 0.10,
    description     = "CME E-mini Russell 2000 — $50/point, lower margin than ES.",
))

M2K = _reg(InstrumentSpec(
    symbol          = "M2K",
    name            = "Micro E-mini Russell 2000 Futures",
    asset_class     = AssetClass.INDEX_FUTURES,
    exchange        = Exchange.CME,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 5.0,
    margin_per_lot  = 700,
    typical_stop_tight  = 5.0,
    typical_stop_normal = 10.0,
    typical_stop_wide   = 20.0,
    tick_size       = 0.10,
    description     = "CME Micro Russell 2000 — $5/point. Very accessible margin.",
))

# ── US Equity (generic) ───────────────────────────────────────────────────────

EQUITY_US = _reg(InstrumentSpec(
    symbol          = "EQUITY_US",
    name            = "US Equity (Cash/Marginable)",
    asset_class     = AssetClass.EQUITY,
    exchange        = Exchange.NYSE,
    currency        = Currency.USD,
    lot_size        = 1,
    point_value     = 1.0,
    margin_per_lot  = 0,
    typical_stop_tight  = 0.005,
    typical_stop_normal = 0.015,
    typical_stop_wide   = 0.03,
    tick_size       = 0.01,
    description     = "US equity — lot size 1, fully flexible sizing. "
                      "Pattern Day Trader rule: ≥$25k for intraday.",
))


# ── Public lookup helpers ─────────────────────────────────────────────────────

def get_spec(symbol: str) -> Optional[InstrumentSpec]:
    """
    Look up an instrument by symbol (case-insensitive).

    Returns None if symbol is unknown (treat as generic equity).
    """
    return _REGISTRY.get(symbol.upper().strip())


def get_spec_or_equity(symbol: str, currency: Currency = Currency.INR) -> InstrumentSpec:
    """
    Return spec if known, otherwise return generic equity spec for the currency.
    Useful when the instrument is stocks where lot_size=1 applies.
    """
    spec = get_spec(symbol)
    if spec is not None:
        return spec
    # Default: treat as equity with flexible lot size
    if currency == Currency.USD:
        return EQUITY_US
    return EQUITY_INDIA


def list_specs() -> list[InstrumentSpec]:
    """Return all registered instrument specs (no duplicates)."""
    return list({id(s): s for s in _REGISTRY.values()}.values())


def indian_index_specs() -> list[InstrumentSpec]:
    """Return all Indian index F&O specs."""
    return [
        s for s in list_specs()
        if s.exchange in (Exchange.NFO, Exchange.BFO)
        and s.asset_class == AssetClass.INDEX_FUTURES
    ]


def us_micro_futures() -> list[InstrumentSpec]:
    """Return US micro futures (smallest lot size, accessible to small accounts)."""
    return [s for s in list_specs() if s.symbol in ("MES", "MNQ", "M2K")]
