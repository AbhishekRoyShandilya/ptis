# 07 — Do NOT Duplicate (trigger → rule)

The anti-duplication enforcement. If you catch yourself doing the LEFT, STOP and do the RIGHT.

| If you're about to… | STOP — instead |
|---|---|
| compute SuperTrend / ATR / EMA / RSI / VWAP / bb_width / rvol | These are owned by the **Feature Engine** (`marketdata/features.py`). Read them. Never hand-roll an indicator or recompute in a loop. (When the store is operationalized: read the store, don't call `compute_features` per bar.) |
| compute HTF (1h/daily) trend in a new place | Owned today by `story_context._htf_trend`, moving to the Feature Engine. Consume it. Do NOT add a 3rd computation. |
| fetch candles from the lake / a broker directly | Call `candle_store.get()` — the single read-through interface. |
| build a "service" that evaluates a symbol (state/setup/idea) | Call `cognition_service.read(symbol)`. The ONE seam. Building a parallel evaluator is exactly the GEN-1 mistake (`realtime_service`, `discretionary_engine` — superseded). |
| write a new coaching/voice layer | `agents/coach.py` already turns a brain read into the human voice. Extend it. |
| make the eye PREDICT (P(next), forecast, probability, target price) | The edge is **recognition, not prediction**. Predictive engines were demoted/archived (`state_transitions`, `analog_engine`). Emit facts; let the reasoner judge. |
| store an interpretation ("momentum increasing", "absorption", "buyers in control") | Store only deterministic FACTS. Interpretations are live, contextual, falsifiable reasoner output — never frozen. |
| optimize perf by recomputing-but-faster | First check if a precomputed source exists (Feature Store). The 2026-06-28 supertrend/threading work was a band-aid over the dormant store — the real fix is to READ the store. |
| add a new strategy into code | Strategies are USER-TAUGHT via the UI (`taught_strategies.py`, starts EMPTY). Never pre-seed your own. |
| persist any user-personal content (watchlist, prefs, strategies) | Default EMPTY + user-entered. Not pre-filled guesses. |

## Enforced (executable)
`tests/test_feature_ownership.py` is a RATCHET: it fails if a NEW module calls `compute_features`
directly (must use `features.get_features`), and forces the BASELINE backlog (19 files, shrinking) to
stay honest. So the top row of this table is now machine-checked, not just a convention.

## The meta-rule
Before building/optimizing ANY capability: read `01_canonical_owners.md`, grep for an existing
implementation, read its docstring + check status in `04_current_status.md`. **Report what exists
before proposing an approach.** Use / fix / extend the owner — do not build a parallel.
