# 09 — PTIS Core v1 — FROZEN (2026-06-29)

The architectural phase is **done**. The core is stable, validated, and fingerprint-protected.
We now stop building infrastructure and let **reality be the teacher** (dogfood → real traders → only
then return to Layer-2 perf). Velocity comes from a floor that doesn't move.

## The rule
**Core v1 is FROZEN: bug-fixes only. No architecture rewrites, no "one more optimization."** New value
is built in the PRODUCT LAYER, on top of the frozen core. If you feel the urge to refactor the core,
write the reason in `06_open_questions.md` and leave it for a deliberate, scheduled un-freeze — not an
in-the-moment yak-shave.

## The boundary (what's frozen vs what's active)
| FROZEN — Core v1 (bug-fixes only) | ACTIVE — Product layer (build freely on top) |
|---|---|
| Feature Engine (`marketdata/features.py`) + store | COACH (`agents/coach.py`) |
| The Eye (`state_narrator` + birth/balance/move/momentum/story_context/bias/session_clock/zone) | Unified Partner (`agents/partner.py`) + proactive UX |
| Memory (`working_memory`, `episodic`, `narrative`) | Learning Journal (to build) |
| Reasoner (`reasoner.py`) + staging (`staged.py`) | Onboarding / watchlist intelligence / strategy teaching UX |
| Cognition seam (`cognition_service.read`) | Anything new = a consumer of `cognition_service.read()` |

Protected by: `data/_perf_ref/` fingerprints + `tests/test_feature_ownership.py` ratchet. A core change
that moves a fingerprint is a RED FLAG — stop and justify, don't ship.

## Explicitly DEFERRED (do NOT build until reality asks for it)
- **Layer 2** (incremental reasoning / O(n)→snappy). Maybe narration speed isn't the bottleneck —
  maybe explanation clarity or onboarding is. Users decide. Revisit only with live evidence.
- Backfilling the universe feature store, migrating the remaining ~18 recompute sites, fully-warmed-
  canonical re-validation. All real, none urgent. They wait.

## The roadmap that replaces "more infra"
1. **Phase 0 — make it RUN LIVE** (the prerequisite nobody verified): feed flowing intraday, proactive
   loop on at a sane cadence, you can sit down and have it watch your watchlist.
2. **Phase 1 — dogfood (2–4 wks):** use it on real decisions. Daily: did it warn me? miss something?
   was the explanation useful? did it change my action? Capture observations structurally.
3. **Phase 2 — 5–10 traders:** don't sell, watch. Confusion / ignored / repeatedly-asked → features.
4. **Phase 3 — return to Layer 2** only when live evidence shows where latency actually hurts.
