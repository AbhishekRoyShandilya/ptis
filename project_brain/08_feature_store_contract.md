# 08 — Feature Store Contract

The shared perception layer. Settled philosophy (user + partner + coder, 2026-06-28).

## Invariants
1. **Cache, not source of truth.** The Feature ENGINE (`marketdata/features.py::compute_features`) is
   authoritative. The Store is persistent storage of its output. Direction:
   `OHLCV → candle_store → Feature Engine → {live calc, persist→Store}`. NEVER Store→Engine.
2. **Never compute, only request.** Consumers call `get_features(...)` (read-through): serve stored
   rows; on miss/stale → Engine computes, persist, return. No consumer recomputes indicators itself.
3. **Versioned.** Every row carries `feature_version`. Read-time GATE: a row at a stale version is
   recomputed, never served. Bump the version when any feature's logic changes.
4. **Immutable closed bars; live forming bar.** Cache only CLOSED candles (immutable). The current
   forming bar is computed live and persisted only on close.
5. **Key/grain.** One row per `(symbol, interval, ts, feature_version)`; one row = one candle's
   deterministic, observer-independent perception.

## What goes in (the 3 tiers)
- **Tier 1 — per-bar, causal, observer-independent → CACHE:** ATR, EMA(s), RSI, VWAP, SuperTrend,
  bb_width, rvol, adx, mom, body/wick ratios, gap, session phase, opening type, dist-from-VWAP, **HTF
  regime** (completed bars only). (FEATURE_COLS today is Tier-1.)
- **Tier 2 — purely time-based, no search horizon → CACHE:** session phase, opening type, gap, bars-
  since-open. (Deterministic AND horizon-free.)
- **Tier 3 — CONTEXTUAL / observer-or-horizon-dependent → DO NOT CACHE (Eye-owned):** swing highs/lows,
  structure (HH_HL/LH_LL), structural/liquidity levels, room — these SCAN history with per-bar ATR, so
  they're warm-up/search-horizon sensitive (PROVEN 2026-06-29: the store-vs-eye divergence was here, not
  in any indicator). ALSO move-state quantities: extension (ATR from birth), room-to-the-move, trend-
  relation-to-active-move, peak-ext, born-counter, move sequence. A property of a (bar, horizon/move),
  not of a bar. (Correction: structure/swings were briefly mis-filed as Tier-2 — they are Tier-3.)

## NEVER store (interpretations — brain only)
"absorption", "distribution", "acceptance", "threat", "energy fading", "weakening", "major move",
"momentum increasing", "buyers in control". These are the reasoner's live, falsifiable conclusions.

## ⚠ TWO LENSES — store vs eye (INVESTIGATED + resolved 2026-06-29)
Investigation (the backfill flipped RELIANCE's narration on 1 field of 1 step) traced the divergence to
its exact source and settled the architecture:
- **It is NOT in any pure-math feature.** The eye reads `atr, vwap, ema20, ema50, bb_width_pct, rvol,
  rsi` — all CONVERGE before the session bars (it doesn't even use ema200). At the bars consumers read,
  pure-math features are window-independent and the store has them correct.
- **The divergence is the SWING-LEVEL detector** (`situation_assembler._gather_levels` → `atr_zigzag`),
  which SCANS the early-window bars where ATR (slow Wilder smoothing, ~184-bar convergence) is still
  under-converged. So it is a CONTEXTUAL computation reacting to warm-up history — Category 2.
- **No clean bit-identical fix exists**: even a warm-up buffer (converged ATR over the same 10-day search
  horizon) makes the detector find a swing it currently misses — more correct, but DIFFERENT → would
  need RE-VALIDATION of the library. Per the stop-condition, we did NOT pursue it.

**RESOLUTION (the two-lens model):** the EYE = its fixed ~10-day observation window (research semantics;
slow indicators are intentionally under-converged at the window front; the validated library is built on
this lens). The STORE = GLOBAL fully-warmed pure-math features (for consumers wanting absolute truth).
**The eye does NOT read the store** — different lenses, both legitimate. Structure/levels/room are
Category-2 contextual → EYE-ONLY, NEVER in the store (see Tier-3 below). If "global semantics for the
eye" is ever wanted, it's a deliberate re-validation project, not a transparent swap.

## Success criterion (beyond bit-identical)
After migration: **no downstream module computes SuperTrend/HTF/ATR/VWAP/EMA/RSI directly.** Any direct
indicator math outside the Feature Engine is a BUG. Enforced by a test (grep guard) + `07`.

## Migration (staged, each bit-identical vs data/_perf_ref/)
1. Read-through `get_features` + version gate; populate; point the hot Eye reads at it. ← Stage 1.
2. Move HTF ownership (`story_context._htf_trend`) into the Feature Engine.
3. Tier-2 (structure/swings) point-in-time into the Engine.
4. Enforce the invariant (the grep test) + migrate remaining of the 45 sites.
Then flip Feature Store DORMANT→LIVE in `04`.
