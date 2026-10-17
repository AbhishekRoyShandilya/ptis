"""
marketdata.regime_signature — the multi-timeframe regime SIGNATURE.

Compresses the per-timeframe regimes (5m/15m/1h/1d, already detected by
marketdata.regime_store and frozen on every AVT trade) into ONE interpretable
label, so PTIS can LEARN strategy performance conditioned on the *alignment*
across timeframes — not just a single composite regime.

This answers the trader questions like:
  • "1h bearish but 5m bullish" → counter_htf-down_ltf-up (counter-trend bounce)
  • "1h up, 5m down"            → counter_htf-up_ltf-down (pullback in uptrend)
  • "everything down"           → aligned_down
  • "HTF range + LTF compressed"→ compression_stack (the 'switch to 5m' trigger)

ONE canonical definition, shared by analytics.edge_matrix (strategy×context edge)
and analytics.strategy_symbol_perf (per-stock ranking), so the signature means the
same thing everywhere.

NOTE: direction comes from the directional regime labels (trending_up/_down). The
oversold/overbought refinement (e.g. "oversold in 5m+15m within a 1h downtrend")
needs per-TF OB/OS captured in the regime snapshot — a separate, deferred step.
"""
from __future__ import annotations

from typing import Optional

# Regime label → coarse directional/structural bucket.
_DIR = {
    "trending_up": "up", "trending_bullish": "up",
    "trending_down": "down", "trending_bearish": "down",
    "breakout": "breakout",
    "ranging": "range", "range": "range",
    "mean_reverting": "revert",
    "high_volatility": "volatile",
    "low_volatility": "compression",
}


def _bucket(regime) -> Optional[str]:
    return _DIR.get(str(regime or "").strip().lower()) or None


def classify_mtf_signature(r5=None, r15=None, r1h=None, r1d=None) -> Optional[str]:
    """
    Multi-TF regime signature from the four per-TF regime labels. Returns a compact
    interpretable string (without the 'mtf:' prefix), or None if no TF is known.

    HTF context = 1h (1d as fallback); LTF timing = 5m (15m as fallback).
    """
    htf = _bucket(r1h) or _bucket(r1d)     # trend context
    ltf = _bucket(r5) or _bucket(r15)      # timing context
    if htf is None and ltf is None:
        return None

    # Compression stack — narrow ranges stacked across TFs; the partner's
    # "switch to 5m intelligence mode" trigger (volatility expansion pending).
    if htf in ("compression", "range") and ltf == "compression":
        return "compression_stack"

    directional = {"up", "down"}
    if htf in directional and ltf in directional:
        if htf == ltf:
            return f"aligned_{htf}"                  # aligned_up / aligned_down
        return f"counter_htf-{htf}_ltf-{ltf}"        # counter_htf-down_ltf-up, etc.

    # Volatility expansion / breakout on either side
    if "volatile" in (htf, ltf) or "breakout" in (htf, ltf):
        return "expansion"

    # HTF directional, LTF range/revert (the structural home of "fade in trend")
    if htf in directional:
        return f"htf-{htf}_ltf-{ltf or 'mixed'}"

    return "mixed"


def signature_from_row(row) -> Optional[str]:
    """Convenience: build the signature from a dict/Series carrying regime_<tf>."""
    g = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)
    return classify_mtf_signature(
        g("regime_5m"), g("regime_15m"), g("regime_1h"), g("regime_1d"),
    )
