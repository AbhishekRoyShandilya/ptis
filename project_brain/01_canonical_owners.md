# 01 — Canonical Owners

Each capability has **ONE owner** (where it's computed/decided). Everyone else **CONSUMES** it.
If you need capability X, call its owner. Never recompute it elsewhere.

> Verified from module docstrings 2026-06-28. `→target` = where ownership is MOVING (not there yet).

## Data layer
| Capability | Owner (canonical) | Consumers | Notes |
|---|---|---|---|
| Raw OHLCV access | `marketdata/candle_store.py` `get()` | everyone | "single read-through interface every consumer uses" — never read the lake directly |
| Parquet/DuckDB storage | `marketdata/lake.py` | candle_store, features | partition layout + read/write |
| System data feed (Upstox) | `marketdata/providers.py` | candle_store | Upstox v3 historical/live; NOT the user's broker (Groww=execution) |
| **Indicators / features** (ATR, EMA, RSI, VWAP, SuperTrend, bb_width, rvol, …) | **`marketdata/features.py`** (Feature Engine/Store) | the whole Eye, scanner, AVT, regime | **DORMANT today** (see 04): designed as precompute-once store, but 45 sites recompute via `compute_features()`. Target = read from store. |
| HTF (1h/daily) trend | `intelligence/story_context.py` `htf_trend_series` (single source; wraps `setup_features._htf_trend`) | `context_frame` + the Eye (threaded once/session) | computed ONCE per narration, threaded into read_move. **→target: persist in Feature Store (Tier-1, Stage 2b)** so scanner/AVT read it too. Never add a 3rd HTF computation. |
| Regime | `strategy_engine/regime_detector.py` | scanner, UI, analytics | derived FACT (deterministic) |
| Session features / phase (IST) | `intelligence/session_clock.py` | Eye | `add_session_features`, `session_phase` |

## The Eye (perception → FACTS, never prediction)
| Capability | Owner | Consumers |
|---|---|---|
| Narrate evolving session state (orchestrator) | `intelligence/state_narrator.py` `narrate_session` | cognition_service |
| Move birth / tracking | `intelligence/birth.py` (`BirthTracker`) | state_narrator, staged |
| Balance vs transition state | `intelligence/balance_state.py` `read_balance` | state_narrator |
| Cumulative move read | `intelligence/move_read.py` `read_move` | state_narrator |
| Momentum run detect (up/down) | `intelligence/momentum_run.py` | move_read |
| Context-state axis | `intelligence/story_context.py` | move_read |
| Directional bias + opening context | `intelligence/directional_bias.py` | state_narrator |
| Zone memory (price-at-zone history) | `intelligence/zone_memory.py` | Eye |
| Lifecycle stage | `intelligence/lifecycle.py` | situation chain |

## Memory
| Capability | Owner | Consumers |
|---|---|---|
| Scoped working memory (BAR/MOVE/SESSION/BIAS/LIBRARY) | `intelligence/working_memory.py` | state_narrator |
| Episodic library + recall + birth signature | `intelligence/episodic.py` | state_narrator, strategy_backtest |
| Narrative (situations-as-stories) | `intelligence/narrative.py` | (recall) |

## Judgment + staging
| Capability | Owner | Consumers |
|---|---|---|
| LLM judgment over facts (mechanism/consequence/invalidation) | `intelligence/reasoner.py` | cognition_service |
| Trader staging (Candidate→Money Move) | `intelligence/staged.py` | cognition_service |

## The seam + the faces (everything funnels through ONE seam)
| Capability | Owner | Consumers |
|---|---|---|
| **THE brain entry point** | **`intelligence/cognition_service.py`** `read()` | EVERY face must call this — never a parallel eval engine |
| **EXECUTIVE — attention scheduler (PLANNED)** | crude today: `proactive_monitor._worth_surfacing`; promote to a first-class subsystem | the Partner / proactive surfacing | ranks what deserves the trader's ATTENTION (grounded score, auditable, metric=noise-reduction). See `11_product_northstar`. |
| COACH voice (human) | `agents/coach.py` | partner, /coach |
| Unified Partner (Co-Planner+COACH) | `agents/partner.py` | /partner, proactive |
| AVT (autonomous trader) | `intelligence/avt_trader.py` | /avt |
| Proactive monitor + alert publish | `intelligence/proactive_monitor.py` + `proactive_loop.py` | /proactive, ProactiveLoop |
| Taught-strategy store + signature stats | `intelligence/taught_strategies.py` + `strategy_backtest.py` | partner, /strategies |
| LLM client (multi-provider) | `agents/llm_client.py` | reasoner, agents |
