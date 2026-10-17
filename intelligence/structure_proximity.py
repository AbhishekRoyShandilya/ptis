"""
intelligence.structure_proximity — "clear air": how far until the next support /
resistance in the move's direction.

THE magnitude driver behind a directional move. A break into CLEAR AIR runs; a
break into NEARBY structure stalls. Concrete: OIL 2026-06-09 broke its intraday
range and ran −2.9% — and stopped *exactly* at the prior launch base (~470),
because that was the nearest support. The trigger tells you a move STARTED; this
tells you how FAR it can travel before it hits structure.

Reuses the swing/level primitives in strategy_engine.reversal_logic. Levels should
be computed on a STRUCTURAL lookback (multi-day / daily) — the supports that cap a
move are prior bases/consolidations, not today's wiggles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from strategy_engine.reversal_logic import _swing_highs, _swing_lows


def _significant_levels(highs, lows, atr, price, window=5, min_touches=2):
    """Cluster swing pivots into ATR-width ZONES weighted by touch count, keeping
    only TESTED levels (≥min_touches). A level every few points is noise; real
    support/resistance is a zone price reacted at repeatedly."""
    pivots = sorted(_swing_highs(highs, window) + _swing_lows(lows, window))
    if not pivots:
        return []
    band = max(0.5 * atr, price * 0.004)
    zones, cur = [], [pivots[0]]
    for p in pivots[1:]:
        if p - cur[-1] <= band:
            cur.append(p)
        else:
            zones.append(cur); cur = [p]
    zones.append(cur)
    levels = [(sum(z) / len(z), len(z)) for z in zones]      # (zone price, touches)
    tested = [(lv, t) for lv, t in levels if t >= min_touches]
    return [lv for lv, _ in (tested or levels)]              # fall back if none tested twice


@dataclass
class StructureProximity:
    price: float
    atr: float
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    support_dist_atr: Optional[float] = None     # (price − support)/atr; None if none below
    resistance_dist_atr: Optional[float] = None  # (resistance − price)/atr
    clear_air_below: bool = False                # no support within clear_air_atr below → room to fall
    clear_air_above: bool = False                # no resistance within clear_air_atr above → room to rise
    levels: list = field(default_factory=list)

    def room_to_run(self, direction: str) -> Optional[float]:
        """ATRs of clear room in the move direction (None = clear air / unbounded)."""
        if direction in ("bearish", "down", "short"):
            return None if self.clear_air_below else self.support_dist_atr
        return None if self.clear_air_above else self.resistance_dist_atr

    def summary(self, direction: str = "bearish") -> str:
        if direction in ("bearish", "down", "short"):
            if self.clear_air_below:
                return f"CLEAR AIR below (no support within {self.support_dist_atr or '∞'} ATR) — room to run"
            return f"support {self.nearest_support:.1f} ≈ {self.support_dist_atr:.1f} ATR below — limited downside"
        if self.clear_air_above:
            return "CLEAR AIR above — room to run"
        return f"resistance {self.nearest_resistance:.1f} ≈ {self.resistance_dist_atr:.1f} ATR above — capped"


def structure_proximity(
    candles: pd.DataFrame, price: Optional[float] = None, atr: Optional[float] = None,
    swing_window: int = 5, clear_air_atr: float = 3.0,
) -> StructureProximity:
    """
    Support/resistance proximity for `price` (default: last close) given a candle
    history. `clear_air_atr`: a level beyond this many ATRs counts as "clear air".
    """
    c = candles.sort_values("ts").reset_index(drop=True) if "ts" in candles else candles
    highs = list(c["high"].astype(float)); lows = list(c["low"].astype(float))
    closes = list(c["close"].astype(float))
    if price is None:
        price = closes[-1]
    if atr is None:
        rng = (c["high"] - c["low"]).rolling(14, min_periods=5).mean()
        atr = float(rng.iloc[-1]) if len(rng) and pd.notna(rng.iloc[-1]) else max(price * 0.01, 1e-6)
    atr = max(float(atr), 1e-6)

    levels = _significant_levels(highs, lows, atr, price, window=swing_window)
    # only levels meaningfully away from the current price (avoid self-hits)
    supports = [l for l in levels if l < price * 0.999]
    resistances = [l for l in levels if l > price * 1.001]

    pr = StructureProximity(price=float(price), atr=atr, levels=levels)
    if supports:
        pr.nearest_support = max(supports)
        pr.support_dist_atr = round((price - pr.nearest_support) / atr, 2)
        pr.clear_air_below = pr.support_dist_atr > clear_air_atr
    else:
        pr.clear_air_below = True
    if resistances:
        pr.nearest_resistance = min(resistances)
        pr.resistance_dist_atr = round((pr.nearest_resistance - price) / atr, 2)
        pr.clear_air_above = pr.resistance_dist_atr > clear_air_atr
    else:
        pr.clear_air_above = True
    return pr
