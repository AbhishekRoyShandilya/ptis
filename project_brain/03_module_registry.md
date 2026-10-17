# 03 — Module Registry

Core spine only (the duplication-prone parts). One line each. See `01` for owners, `04` for status.

## marketdata/ (data + features)
- `candle_store.py` — single read-through OHLCV interface (`get()`).
- `lake.py` — Parquet partition layout + DuckDB read/write.
- `providers.py` — system data feed (Upstox v3 historical/live).
- `features.py` — Feature Engine/Store: precompute indicators once, store, read. **[DORMANT]**
- `daily_update.py`, `backfill.py` — keep the lake current / backfill.

## intelligence/ — THE EYE
- `state_narrator.py` — orchestrator; narrate evolving session state bar-by-bar.
- `birth.py` — BirthTracker: has a move been BORN.
- `balance_state.py` — `read_balance`: BALANCE vs TRANSITION.
- `move_read.py` — `read_move`: cumulative live move read.
- `momentum_run.py` — up/down run detection (consumed by move_read).
- `story_context.py` — context-state axis + HTF trend (owns HTF for now).
- `directional_bias.py` — early graded directional lean + opening context.
- `session_clock.py` — IST session features / phase.
- `zone_memory.py` — what happened last N times at a like zone.
- `lifecycle.py` — lifecycle stage (situation chain).

## intelligence/ — MEMORY + JUDGMENT
- `working_memory.py` — scoped working memory (BAR/MOVE/SESSION/BIAS/LIBRARY).
- `episodic.py` — episodic library + recall + `signature_from_birth`.
- `narrative.py` — situations-as-stories recall.
- `reasoner.py` — LLM judgment over facts (mechanism/consequence/invalidation).
- `staged.py` — trader staging (Candidate→Money Move).

## intelligence/ + agents/ — SEAM + FACES
- `cognition_service.py` — **THE seam**: `read(symbol)`. Every face calls this.
- `agents/coach.py` — COACH human voice.
- `agents/partner.py` — Unified Partner (co-plan + coach).
- `avt_trader.py` — AVT autonomous trader on the brain.
- `proactive_monitor.py` + `proactive_loop.py` — proactive scan + alert publish.
- `strategy_backtest.py` + `taught_strategies.py` — taught-strategy stats + store.
- `agents/llm_client.py` — multi-provider LLM client.

## SUPERSEDED (GEN-1, see 04) — do not extend
- `realtime_service.py`, `discretionary_engine.py`, `situation_engine/judge/assembler/repr.py`,
  `analog_engine.py`. Archived: `_archive/{coaching_engine,state_transitions,knowledge_activator}.py`.
