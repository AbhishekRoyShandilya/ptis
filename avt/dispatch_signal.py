"""
avt.dispatch_signal — per-strategy signal-gen dispatch. ONE engine, MANY strategy kinds: a system cascade
template is driven by `CascadeSignalGenerator` (multi-TF), a taught strategy by the tag-signature
`SignalGenerator`. The engine just calls `evaluate(sko, candles, symbol)`; this router picks the right generator
by the SKO (via `avt.templates.is_cascade_sko`), so the engine never learns about strategy kinds. This is what
lets a user run BOTH the cascade templates and their taught strategies in the same AVT.

Priming (cascade only): `prime(symbol, start, end)` for replay; `prime_live(symbol, today, now)` per cycle for
live — both delegate to the shared `CascadeSignalGenerator`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from avt.templates import is_cascade_sko, is_breakdown_sko, preferred_instrument


class _NullGenerator:
    """Stand-in for a system-template generator that is not part of the public build (avt.templates is empty
    there, so no SKO is ever routed to it)."""
    def prime(self, *args, **kwargs):
        return None

    def prime_live(self, *args, **kwargs):
        return None

    def evaluate(self, *args, **kwargs):
        return None

    def evaluate_intent(self, *args, **kwargs):
        return None


class DispatchingSignalGenerator:
    def __init__(self, taught: Any = None, cascade: Any = None, breakdown: Any = None, live_prime: bool = True,
                 prime_ttl: int = 270, cascade_dedup: str = "day_dir"):
        # cascade_dedup defaults to "day_dir" (one signal per symbol/direction/day) — the LIVE-safe, honest
        # behaviour that suppresses same-move rebirths/duplicates/chasing (see cascade_detect.cascade_confirmations
        # + master §2n). The raw research golden builds CascadeSignalGenerator directly (dedup="none"), so this
        # only affects the live per-user AVT — the golden reference is untouched.
        if taught is None:
            from ske.signal_generator import SignalGenerator
            taught = SignalGenerator()
        # Public build: the system-template generators are private plugins and are not shipped.
        if cascade is None:
            cascade = _NullGenerator()
        if breakdown is None:
            breakdown = _NullGenerator()
        self.taught = taught
        self.cascade = cascade
        self.breakdown = breakdown
        self.live_prime = live_prime          # live per-user AVT: self-prime generators from today's warm tail
        self.prime_ttl = prime_ttl            # re-prime a symbol at most every prime_ttl seconds (cache-window safe)
        self._primed_at: Dict[str, Any] = {}      # cascade: symbol -> (date, last_ts)
        self._primed_at_bd: Dict[str, Any] = {}   # breakdown: symbol -> (date, last_ts)

    # ── priming (cascade + breakdown SKOs; taught strategies need none) ──
    def prime(self, symbol: str, start, end) -> None:
        self.cascade.prime(symbol, start, end)
        self.breakdown.prime(symbol, start, end)

    def prime_live(self, symbol: str, today, now, cs=None) -> None:
        self.cascade.prime_live(symbol, today, now, cs=cs)
        self.breakdown.prime_live(symbol, today, now, cs=cs)

    def _throttled_prime_live(self, gen, cache: Dict[str, Any], symbol: str, cur_ts) -> None:
        """Lazily (re)prime today's warm tail for a symbol — throttled per generator, so the scan loop stays light.
        REPLAY (live_prime=False) primes over its range explicitly instead."""
        if not self.live_prime or cur_ts is None:
            return
        import pandas as pd
        cur = pd.Timestamp(cur_ts); today = cur.normalize()
        last = cache.get(symbol)
        if last is not None and last[0] == today and (cur - last[1]).total_seconds() < self.prime_ttl:
            return
        try:
            gen.prime_live(symbol, today, cur)
            cache[symbol] = (today, cur)
        except Exception:
            pass

    # ── the engine's entry point — route by SKO ──
    def evaluate(self, sko: Dict[str, Any], candles: List[Dict[str, Any]], symbol: str, *args, **kwargs):
        cur_ts = (candles[-1].get("ts") or candles[-1].get("timestamp")) if candles else None
        if is_breakdown_sko(sko):
            self._throttled_prime_live(self.breakdown, self._primed_at_bd, symbol, cur_ts)
            return self.breakdown.evaluate(sko, candles, symbol, *args, **kwargs)
        if is_cascade_sko(sko):
            self._throttled_prime_live(self.cascade, self._primed_at, symbol, cur_ts)
            return self.cascade.evaluate(sko, candles, symbol, *args, **kwargs)
        return self.taught.evaluate(sko, candles, symbol, *args, **kwargs)

    def evaluate_intent(self, sko: Dict[str, Any], candles: List[Dict[str, Any]], symbol: str, **kwargs):
        """New-spine path: emit an Intent (cascade + breakdown templates; taught → Intent in M4)."""
        if is_breakdown_sko(sko):
            return self.breakdown.evaluate_intent(
                sko, candles, symbol,
                preferred_instrument=preferred_instrument(sko).value, template=sko.get("template"))
        if is_cascade_sko(sko):
            return self.cascade.evaluate_intent(
                sko, candles, symbol,
                preferred_instrument=preferred_instrument(sko).value, template=sko.get("template"))
        return None

    # transparent passthrough for any other attribute the engine reads off the signal generator
    def __getattr__(self, name):
        return getattr(self.taught, name)
