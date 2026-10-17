# PTIS — Personal Trading Intelligence System

A full-stack platform for researching, journaling and paper-trading Indian equities (NSE): a market-data lake and
live feeds, a trade journal that attributes every trade to a strategy and flags rule breaks, behavioural analytics,
shadow trading, an autonomous virtual trader, and an LLM coaching agent.

Built solo in 2026 with AI pair-programming ([Claude Code](https://claude.com/claude-code)); see
[How it was built](#how-it-was-built).

**Live demo:** [abhishekroyshandilya.github.io/ptis-demo](https://abhishekroyshandilya.github.io/ptis-demo/) runs the
PTIS strategy engine every weekday evening on real NSE end-of-day prices
([source and nightly run logs](https://github.com/AbhishekRoyShandilya/ptis-demo)).

> **This is a curated public copy of a private repository.** The research code, strategy specifications, live
> execution layer, deployment and all data stay private (see [What is not here](#what-is-not-here)). Everything
> in this repository runs and is tested on its own.

---

## At a glance

| | |
|---|---|
| Backend | Python 3.12 · FastAPI · SQLAlchemy 2 · Alembic · PostgreSQL · pydantic-settings |
| Data | Parquet partitions + DuckDB (OHLCV lake) · pandas / NumPy · scikit-learn |
| Market data | Angel One SmartAPI (WebSocket + REST) · Upstox · Groww · Kite · NSE archives |
| AI | OpenAI-compatible LLM client with provider fallback · RAG (ChromaDB or in-memory) |
| Frontend | React 18 · TypeScript · Vite · Tailwind · Zustand · Recharts |
| Ops | Docker Compose · Makefile · pytest (SQLite, no services needed) |
| Size | ~170k lines of Python, ~15k lines of TypeScript · 30 routers / ~290 endpoints · 49 tables · 37 migrations |

## Architecture

```
                 ┌──────────────── React + TypeScript (frontend/) ────────────────┐
                 │ journal · analytics · shadow · AVT · coach chat · market views │
                 └───────────────────────────────┬────────────────────────────────┘
                                                 │ REST + WebSocket (JWT)
┌────────────────────────────────────────────────┴───────────────────────────────────────────────┐
│ FastAPI (api/)  — auth, rate limiting per route class, request metrics, 30 routers             │
├──────────────────┬──────────────────┬──────────────────┬──────────────────┬────────────────────┤
│ Trade pipeline   │ Strategy engine  │ Behavioural      │ Shadow trading   │ AVT (virtual       │
│ core/            │ strategy_engine/ │ behavioral/      │ shadow/          │ trader) avt/       │
│ entry → attribute│ YAML templates,  │ deviation, bias, │ mirror / strategy│ scan → rank →      │
│ → persist → event│ conditions,      │ tilt detection,  │ / variant shadow │ execute virtually, │
│                  │ 3-tier attribution│ decision quality │ trades, compare  │ capital operator   │
├──────────────────┴──────────────────┴──────────────────┴──────────────────┴────────────────────┤
│ Event bus (core/events.py) · Coaching agent + RAG (agents/, rag/) · Improvement lab            │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Market data (marketdata/): Parquet/DuckDB lake · coverage ledger · sealed sessions · live WS   │
│ feed + tick aggregator · feature engine · regime store       Broker adapters (ingestion/)      │
└───────────────────────────────┬──────────────────────────────────┬─────────────────────────────┘
                          PostgreSQL                        Parquet lake (data/, git-ignored)
```

## Engineering highlights

Each item points at code in this repository.

1. **An append-safe market-data lake.** `marketdata/lake.py` stores candles as
   `type=/symbol=/interval=/YYYY.parquet` partitions read through one DuckDB connection. Writes are idempotent
   (merge on timestamp, keep last), so a re-run backfill never duplicates rows. A coverage ledger records which
   ranges are complete.
2. **Completed sessions are immutable.** I found that a plain *read* could silently rewrite stored history: a cache
   miss re-fetched a past day from whichever feed was configured, and the two brokers disagree on volume. The fix
   was to make the bad state impossible rather than patch the symptom. `marketdata/session_seal.py` freezes every
   finished session: fetched rows for a sealed day are dropped before any write, and a deliberate repair has to
   re-seal, which records a new fingerprint so the change is visible rather than silent.
3. **Architecture rules as tests.** `tests/test_feature_ownership.py` is a ratchet: indicator features have one
   owner (the feature engine), a *new* direct caller fails the build, and the baseline of old callers can only
   shrink.
4. **Registration before testing.** `research_bridge/hyp/registry.py` keys each research hypothesis by the hash of
   its spec. A test refuses to run unless that exact spec was registered first, an edited spec counts as a new
   variant (so the number of tries stays visible), and each hold-out window can be opened only once.
   `research_bridge/hyp/stats.py` holds the shared statistics: cluster-robust t, clustered OLS,
   Benjamini–Hochberg FDR, years-won and top-k concentration.
5. **Strategy attribution with override detection.** `strategy_engine/attribution.py` resolves every journalled
   trade in three tiers: the strategy the trader declared, an automatic match against YAML strategy templates, and
   detection of where the trade broke its own rules (no stop, oversized, outside the time window…). Conditions are a
   plug-in registry (`strategy_engine/conditions/`).
6. **Shadow trading.** `shadow/` opens "what if" twins of every real trade (a mirror, the strategy-correct
   version and variants), drives them with prices, and reports where and why reality diverged.
7. **Behavioural analytics and decision quality.** `behavioral/` detects revenge trading, FOMO, oversizing, tilt and
   cognitive biases against the trader's own baseline. `analytics/decision_quality.py` scores each decision on six
   dimensions, separately from its P&L outcome.
8. **An autonomous virtual trader with honest capital maths.** `avt/` runs strategies on live data in a virtual
   portfolio through one execution interface (`avt/execution/`). The capital operator (`avt/capital/`) sizes
   through lot, margin, leverage and capacity bounds, and reports bounded CAGR / max drawdown / MAR. This replaced
   an earlier naive compounding that produced fantasy curves.
9. **Broker adapters behind one interface.** `ingestion/broker_adapters/` covers Angel One (TOTP login), Upstox,
   Groww, Zerodha Kite, Interactive Brokers, Alpaca and others, plus CSV / statement imports and de-duplication.
   The live WebSocket feed is `marketdata/ws_feed.py`. Every SDK response is unwrapped defensively because broker
   payloads change shape without notice.
10. **Coaching agent.** `agents/` assembles trading context per request (recent trades, decision quality,
    behavioural events, shadow gaps) and retrieves the trader's own knowledge through RAG (`rag/`). It runs on an
    LLM client with retries and provider fallback.

## Lessons that shaped the design

The most valuable work was finding the ways a trading system lies to you. [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md)
covers:

- **Look-ahead in a multi-timeframe backtest:** bars labelled by their start time made a dead strategy look
  profitable. I caught it, killed the strategy, and added a mandatory truncation test.
- **Survivorship bias:** the free price universe held only today's survivors, so I built an all-NSE daily panel
  that includes delisted stocks, from exchange bhavcopy files.
- **Corrupted prices:** unadjusted splits, stitched histories and coverage holes that looked like 200% moves.
- **Two feeds, two truths:** two brokers agree on prices but not on volume or the closing-auction bar.
- **"No defect reaches money undetected":** pre-market invariant checks, dry-run by default, reconciliation.

## What is not here

| Kept private | Why |
|---|---|
| Research code, notebooks, results and the trial registry's contents | they encode the strategies |
| Strategy specifications and validated system templates (`avt/templates.py` ships empty) | alpha |
| Live execution layer: decision jobs, order placement, reconciliation, supervisor, alerts | alpha + account safety |
| Deployment: cloud runbooks, IAM, schedules | account-specific |
| All data: candles, panels, models, ledgers | size + licensing + privacy |

A few modules mention private components in comments or in function-level imports. Those code paths are inactive
in this build (for example, the template registry is empty, so the engine never routes to a private generator).

## Running it

```bash
cp .env.example .env               # set PTIS_JWT_SECRET and an LLM key
docker compose up --build          # PostgreSQL + API on :8000
# or locally
pip install -r requirements.txt -r requirements-dev.txt
PTIS_ENV=test uvicorn api.main:app --port 8001
cd frontend && npm install && npm run dev
```

Tests need no database server or network. They run on in-memory SQLite, with Postgres types mapped in
`tests/conftest.py`:

```bash
pytest
```

## How it was built

I built PTIS alone, with AI pair-programming (Claude Code) throughout. I set the direction, designed the research
method and the architecture, made the research and risk decisions, and reviewed the changes. A large share of the
code was written in those pair-programming sessions. Several of the hardest bugs described in the engineering notes
were AI-written code that my checks caught, which is why the checks exist.

## Disclaimer

A personal engineering project, not investment advice. Nothing here is a recommendation to trade.

## License

Copyright © 2026 Abhishek Kumar. All rights reserved. See [LICENSE](LICENSE).
