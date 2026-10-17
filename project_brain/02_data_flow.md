# 02 — Data Flow

```
Broker/Upstox feed ─▶ providers.py ─▶ lake.py (Parquet/DuckDB)
                                         │
                                         ▼
                                   candle_store.get()        ← the ONLY way anyone reads candles
                                         │
                                         ▼
                              Feature Engine (features.py)    ← compute indicators ONCE  [DORMANT: see 04]
                                         │  (precompute on candle close / live append)
                                         ▼
                                  Feature Store (lake/features/…)
                                         │  read, never recompute
                                         ▼
┌─────────────────────────── THE EYE (facts, not prediction) ───────────────────────────┐
│ state_narrator (orchestrator) ── reads ──▶ birth · balance_state · move_read           │
│   (momentum_run, story_context/HTF, directional_bias, session_clock, zone_memory)      │
│                         │ accumulates ▶ working_memory + episodic (recall)             │
└────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          ▼
                          reasoner.py (LLM judgment)  +  staged.py (Candidate→Money Move)
                                          │
                                          ▼
                       cognition_service.read()   ◀── THE ONE SEAM every face calls
                                          │
        ┌──────────────┬─────────────────┼───────────────────┬──────────────────┐
        ▼              ▼                 ▼                   ▼                  ▼
     coach          partner           avt_trader      proactive_monitor    strategy_backtest
   (human voice)  (co-plan+coach)   (autonomous)      → proactive_loop       (taught sigs)
        │              │                                    │
        ▼              ▼                                    ▼
   /coach, Coach   /partner,                       COGNITIVE_ALERT event
     tab          Partner tab                      → /ws/session → chat bubble + LiveAlertsPage
```

**Live tick (target):** new candle → Feature Engine computes ONLY the new bar (recursive indicators
need prev value + new bar) → append to store → Eye updates incrementally → faces decide. O(1)/candle.
**Today:** features recomputed on the fly per bar (O(n²) narration) — see 04 + 06.
