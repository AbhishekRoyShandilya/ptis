"""
avt.execution.option_market — the seam the AVT engine uses to price OPTION positions. The engine reads underlying
candles for detection + the structural stop (unchanged); when it holds an OPTION it needs the option premium (and
the underlying) at the cursor, plus a way to resolve the ATM weekly contract. This abstraction provides exactly
that, with two implementations:

  · LiveOptionMarket  — wraps `avt.live.feed.LiveFeed` (real broker LTP) for live paper trading.
  · ReplayOptionMarket — backtest, sourced from `data/lake/options_my` 1-min data; lives in the tester
    (`scripts/research/verify_avt_options.py`) so `avt/` never imports research-data loaders.

Design rule (user's): the OPTION lifecycle stays in `avt.execution.options` / `avt.live.options_forward` (the proven,
money-test-verified primitives); the engine DELEGATES to them and only asks this market for prices. `ride_ticks`
lets a backtest hand the whole 1-min (spot, premium) path to `manage_to_exit` in one atomic step (reproducing the
money test); live returns None and is ticked cycle-by-cycle instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class OptionMarket(ABC):
    """Prices options for the engine. Contract dict = {tradeable, strike, expiry, lot_size, right, ...}."""

    @abstractmethod
    def resolve_atm(self, underlying: str, ref_px: float, right: str, as_of: Any) -> Optional[Dict[str, Any]]:
        """Nearest-weekly ATM contract for `underlying` at `ref_px` (right = 'CE'|'PE'), or None."""
        raise NotImplementedError

    @abstractmethod
    def premium(self, contract: Dict[str, Any], as_of: Any) -> Optional[float]:
        """Option premium (LTP) for `contract` at `as_of` (the entry fill + live management tick)."""
        raise NotImplementedError

    @abstractmethod
    def spot(self, underlying: str, as_of: Any) -> Optional[float]:
        """Underlying spot at `as_of` (drives the structural-stop check during option management)."""
        raise NotImplementedError

    def ride_ticks(self, contract: Dict[str, Any], underlying: str, entry_ts: Any,
                   eod: str = "15:20") -> Optional[List[Tuple[float, float]]]:
        """BACKTEST convenience: the ordered (spot, premium) minute ticks AFTER entry up to `eod`, so the engine can
        ride the whole option to exit atomically via `manage_to_exit`. Live returns None (ticked cycle-by-cycle)."""
        return None


class LiveOptionMarket(OptionMarket):
    """Live paper option pricing via the broker feed. Thin adapter over `avt.live.feed.LiveFeed` (whose `resolve_atm`
    uses the REAL instrument master and `option_ltp` the REAL LTP). `as_of` is ignored (live = now)."""

    def __init__(self, feed: Any = None):
        if feed is None:
            from avt.live.feed import LiveFeed
            feed = LiveFeed()
        self._feed = feed

    def resolve_atm(self, underlying: str, ref_px: float, right: str, as_of: Any) -> Optional[Dict[str, Any]]:
        return self._feed.resolve_atm(underlying, ref_px, right, as_of)

    def premium(self, contract: Dict[str, Any], as_of: Any) -> Optional[float]:
        return self._feed.option_ltp(contract)

    def spot(self, underlying: str, as_of: Any) -> Optional[float]:
        return self._feed.spot(underlying)

    # ride_ticks → None (live is ticked cycle-by-cycle by the engine's manage loop)
