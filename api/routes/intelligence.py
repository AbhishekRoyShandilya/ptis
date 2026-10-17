"""
Intelligence API Routes

REST and WebSocket endpoints for the real-time intelligence layer.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.dependencies import get_db, get_current_trader_id
from intelligence.realtime_service import RealtimeIntelligenceService
from intelligence.discretionary_engine import DiscretionaryAssessment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intelligence", tags=["intelligence"], dependencies=[Depends(get_current_user)])


# ─── Dependency ──────────────────────────────────────────────────


async def get_intelligence_service(
    db: AsyncSession = Depends(get_db),
) -> RealtimeIntelligenceService:
    """Get intelligence service instance using MarketDataService (Upstox + Angel One)."""
    from strategy_engine.strategy_registry import StrategyRegistry
    from strategy_engine.conditions.registry import ConditionRegistry
    from ingestion.market_data_service import get_market_data_service
    from datetime import datetime, timedelta
    import pandas as pd

    registry = StrategyRegistry()
    registry.initialize()          # load YAML templates from config/strategies/
    condition_registry = ConditionRegistry()

    # Build a price fetcher that wraps MarketDataService (Angel One historical)
    mds = get_market_data_service()

    # NSE spot indices: Upstox historical first — reliable NSE_INDEX key map.
    # BSE indices: Angel One first — Upstox historical API does NOT support BSE_INDEX candles.
    # Stocks: Angel One first (better depth).
    _NSE_INDICES = {
        "NIFTY", "NIFTY 50", "NIFTY50", "BANKNIFTY", "NIFTY BANK",
        "FINNIFTY", "MIDCPNIFTY", "INDIA VIX", "VIX",
        "NIFTY IT", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG",
        "NIFTY METAL", "NIFTY REALTY",
    }
    _BSE_INDICES = {"SENSEX", "BANKEX", "BSE500", "BSE200", "BSE100"}

    async def _market_data_fetcher(symbol: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
        if mds is None:
            raise ValueError("Market data service not initialized — check broker credentials")

        upstox_tf_map = {
            "1m": "1minute", "5m": "5minute", "15m": "15minute",
            "30m": "30minute", "1h": "60minute", "1d": "day", "D": "day",
        }
        days = {"1m": 1, "5m": 3, "15m": 5, "30m": 10, "1h": 30, "1d": 120, "D": 120}.get(timeframe, 5)

        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days)

        candles = []
        _sym_upper = symbol.upper()
        is_index = _sym_upper in _NSE_INDICES or _sym_upper in _BSE_INDICES

        if _sym_upper in _NSE_INDICES:
            sources = ["upstox", "angel_one"]   # Upstox NSE_INDEX is reliable
        elif _sym_upper in _BSE_INDICES:
            sources = ["angel_one", "upstox"]   # BSE_INDEX not in Upstox historical API
        else:
            sources = ["angel_one", "upstox"]   # Stocks: Angel One has more depth

        for source in sources:
            if candles:
                break
            if source == "angel_one" and mds.has_historical:
                try:
                    # fetch_candles_with_live_bar = historical + today's intraday (current bar)
                    candles = mds.fetch_candles_with_live_bar(symbol, timeframe, from_date, to_date)
                    if candles:
                        logger.info("Intelligence candles: Angel One (+live bar) returned %d bars for %s", len(candles), symbol)
                except Exception as e:
                    logger.warning("Angel One candle fetch failed for %s: %s", symbol, e)

            elif source == "upstox" and mds.has_live and mds._upstox:
                try:
                    from ingestion.broker_adapters.upstox import UpstoxInterval
                    key = mds._upstox.resolve_instrument_key(symbol)
                    interval_str = upstox_tf_map.get(timeframe, "5minute")
                    upstox_interval = UpstoxInterval(interval_str)
                    hist = mds._upstox.get_historical_candles(key, upstox_interval, from_date, to_date)
                    if hist:
                        # Merge live intraday bar on top of Upstox historical too
                        candles = mds.fetch_candles_with_live_bar(symbol, timeframe, from_date, to_date)
                        if not candles:
                            candles = hist  # fallback to pure historical if merge fails
                        logger.info("Intelligence candles: Upstox (+live bar) returned %d bars for %s", len(candles), symbol)
                except Exception as e:
                    logger.warning("Upstox candle fetch failed for %s: %s", symbol, e)

        # Last resort: yfinance fallback for BSE indices only (SENSEX, BANKEX).
        # Angel One rejects BSE index tokens and Upstox historical doesn't support BSE_INDEX.
        # NSE indices (NIFTY, BANKNIFTY) should be served by Upstox — no yfinance needed.
        if not candles and _sym_upper in _BSE_INDICES:
            try:
                from ingestion.market_data_service import fetch_index_candles_yfinance
                candles = fetch_index_candles_yfinance(symbol, timeframe, from_date, to_date)
                if candles:
                    logger.info("Intelligence candles: yfinance fallback returned %d bars for %s", len(candles), symbol)
            except Exception as e:
                logger.warning("yfinance fallback failed for %s: %s", symbol, e)

        if not candles:
            label = "index" if is_index else "symbol"
            raise ValueError(
                f"No candle data available for {label} {symbol} — "
                "check that Angel One / Upstox credentials are set in .env"
            )

        df = pd.DataFrame(candles)
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        # Indices have 0 volume in spot data — fill with 0 so downstream code doesn't crash
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in candle data for {symbol}")
        if "volume" not in df.columns:
            df["volume"] = 0

        return df[["open", "high", "low", "close", "volume"]].tail(bars)

    return RealtimeIntelligenceService(
        db=db,
        strategy_registry=registry,
        condition_registry=condition_registry,
        price_fetcher=_market_data_fetcher,
    )


# ─── Schemas ─────────────────────────────────────────────────────


class EnvironmentRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"


class ScanRequest(BaseModel):
    symbol: str
    timeframe: str = "5m"
    strategy_names: Optional[list[str]] = None


class WatchlistScanRequest(BaseModel):
    symbols: list[str]
    timeframe: str = "5m"


class DiscretionaryQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    symbol: str
    direction: Optional[int] = Field(
        None, ge=-1, le=1,
        description="1=long, -1=short, None=unspecified"
    )
    timeframe: str = "5m"
    todays_trade_count: int = Field(default=0, ge=0)
    current_pnl: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What do you think about this NVDA breakout?",
                "symbol": "NVDA",
                "direction": 1,
                "timeframe": "5m",
                "todays_trade_count": 2,
                "current_pnl": 150.0,
            }
        }


class AlertsRequest(BaseModel):
    symbols: list[str]
    min_priority: str = "medium"


# ─── Endpoints ───────────────────────────────────────────────────


@router.post("/environment")
async def get_market_environment(
    request: EnvironmentRequest,
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
):
    """
    Get real-time market environment analysis for a symbol.

    Returns regime, volatility state, trend assessment, structure
    analysis, and composite environment score.
    """
    try:
        environment = await service.get_environment(
            symbol=request.symbol,
            timeframe=request.timeframe,
            force_refresh=True,
        )
        return environment.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Environment analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Environment analysis failed")


@router.post("/scan")
async def scan_setups(
    request: ScanRequest,
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Scan for active setups on a symbol.

    Evaluates all (or specified) strategy templates against current
    market conditions and returns readiness scores.
    """
    try:
        result = await service.scan_setups(
            symbol=request.symbol,
            trader_id=trader_id,
            timeframe=request.timeframe,
            strategy_names=request.strategy_names,
            force_refresh=True,
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Setup scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Setup scan failed")


@router.post("/scan/watchlist")
async def scan_watchlist(
    request: WatchlistScanRequest,
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Scan all symbols on the watchlist for setups.

    Returns scan results for each symbol with alerts.
    """
    if len(request.symbols) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 symbols per watchlist scan"
        )

    try:
        results = await service.scan_watchlist(
            symbols=request.symbols,
            trader_id=trader_id,
            timeframe=request.timeframe,
        )
        return {
            symbol: result.to_dict()
            for symbol, result in results.items()
        }
    except Exception as e:
        logger.error(f"Watchlist scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Watchlist scan failed")


@router.post("/evaluate")
async def evaluate_trade_idea(
    request: DiscretionaryQueryRequest,
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Evaluate a discretionary trade idea.

    Analyzes the symbol's current conditions against the trader's
    strategy templates, risk profile, and behavioral state to provide
    a structured assessment.

    This does NOT recommend trades — it provides structured analysis
    to help the trader make their own informed decision.
    """
    try:
        assessment = await service.evaluate_discretionary(
            query=request.query,
            symbol=request.symbol,
            trader_id=trader_id,
            direction=request.direction,
            todays_trade_count=request.todays_trade_count,
            current_pnl=request.current_pnl,
            timeframe=request.timeframe,
        )
        return assessment.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Discretionary evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Evaluation failed")


class TaughtStrategyRequest(BaseModel):
    name: str
    direction: str = "both"                 # up / down / both
    require: list[str] = Field(default_factory=list)   # tags that MUST be present
    exclude: list[str] = Field(default_factory=list)   # tags that must be ABSENT


class TeachBootstrapRequest(BaseModel):
    description: str
    examples: list[dict] = Field(default_factory=list)   # [{symbol, when, label, interval?}]
    interval: str = "15m"


@router.post("/teach/bootstrap")
def teach_bootstrap(req: TeachBootstrapRequest):
    """The teaching SESSION's first pass: an NL strategy description (any language) + optional pointed
    examples (symbol+time) → a grounded REVIEW the user verifies before confirming. Returns the LLM's
    proposed signature + honestly-flagged vocab gaps + execution notes, the EYE's read of each pointed
    example, the grounded challenge, validation with MATCHED examples, and proposed refinements. Stores
    NOTHING (confirm via POST /strategies). Sync def → threadpooled (LLM + eye + library)."""
    from intelligence.teaching_service import bootstrap
    try:
        return bootstrap(req.description, examples=req.examples, interval=req.interval).to_dict()
    except Exception as e:
        logger.error(f"teach bootstrap failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="teach bootstrap failed")


class TeachMessageRequest(BaseModel):
    message: str


@router.post("/teach/extract")
def teach_extract(req: TeachMessageRequest):
    """Route a natural chat message → strategy DESCRIPTION, a trade EXAMPLE (symbol+time, validated), or
    an ASK when ambiguous (the chat then asks the user). LLM extracts; deterministic guards validate."""
    from intelligence.teaching_service import extract_message
    try:
        return extract_message(req.message)
    except Exception as e:
        logger.error(f"teach extract failed: {e}", exc_info=True)
        return {"kind": "description"}


class TeachGroundRequest(BaseModel):
    symbol: str
    when: str
    label: str = "take"
    interval: str = "15m"


@router.post("/teach/ground")
def teach_ground(req: TeachGroundRequest):
    """Ground one pointed example on the EYE — what it saw at (symbol, time), on real data."""
    from intelligence.teaching_service import ground_example
    try:
        return ground_example(req.symbol, req.when, req.interval, req.label).to_dict()
    except Exception as e:
        logger.error(f"teach ground failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="ground failed")


class TeachSaveSessionRequest(BaseModel):
    name: str
    direction: str = "up"
    require: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    description: str = ""
    messages: list[dict] = Field(default_factory=list)
    examples: list[dict] = Field(default_factory=list)
    thread: str = "teaching"          # teaching (refine recognition) | analysis (money/hypotheses)


@router.post("/teach/save-session")
def teach_save_session(req: TeachSaveSessionRequest):
    """Persist a strategy's SESSION (signature + conversation + examples + a new version when the signature
    changed). `thread` chooses which conversation is saved — teaching or analysis (they never mix)."""
    from intelligence.taught_strategies import save_session
    try:
        save_session(req.name, direction=req.direction, require=req.require, exclude=req.exclude,
                     description=req.description, messages=req.messages, examples=req.examples, thread=req.thread)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"save session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="save session failed")


# ── Analysis side: the MONEY thread (P&L backtest via avt_trader + grounded Q&A) ──

class AnalysisBacktestRequest(BaseModel):
    name: str
    direction: str = "up"
    require: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    interval: str = "5m"
    max_sessions: int = 120


@router.post("/analysis/backtest")
def analysis_backtest(req: AnalysisBacktestRequest):
    """Start a background P&L backtest (replays matching sessions through avt_trader — structural stop +
    asymmetric exit, honest R). Returns live status immediately; poll status for the result."""
    from intelligence.strategy_analysis import start_backtest
    try:
        return start_backtest(req.name, req.direction, req.require, req.exclude, req.interval, req.max_sessions)
    except Exception as e:
        logger.error(f"analysis backtest failed to start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="backtest failed to start")


@router.get("/analysis/backtest/status/{name}")
def analysis_backtest_status(name: str):
    """Live status of a running backtest, or the last stored ledger if none is running."""
    from intelligence.strategy_analysis import backtest_status
    return backtest_status(name)


class ApplyRefinementRequest(BaseModel):
    name: str
    op: str                          # require | exclude
    tags: list[str] = Field(default_factory=list)
    reason: str = ""


@router.post("/analysis/apply-refinement")
def analysis_apply_refinement(req: ApplyRefinementRequest):
    """Apply a finding as a versioned signature change (provisional, logged) — the finding→improvement loop.
    Returns the new signature; the caller then re-runs the backtest to validate the money effect."""
    from intelligence.strategy_analysis import apply_refinement
    try:
        return apply_refinement(req.name, req.op, req.tags, req.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"apply refinement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="apply failed")


@router.get("/analysis/verdict/{name}")
def analysis_verdict(name: str):
    """After an applied refinement re-runs, the HONEST held/didn't-hold verdict vs the baseline (judged on
    expectancy). Resolves the pending hypothesis. {} if nothing pending."""
    from intelligence.strategy_analysis import evaluate_refinement
    try:
        return evaluate_refinement(name) or {}
    except Exception as e:
        logger.error(f"verdict failed: {e}", exc_info=True)
        return {}


@router.get("/analysis/history/{name}")
def analysis_history(name: str):
    """The strategy's version history + applied-refinement (hypothesis) log — the accumulated improvements."""
    from intelligence.taught_strategies import load_session
    s = load_session(name) or {}
    return {"versions": s.get("versions", []), "hypotheses": s.get("hypotheses", [])}


@router.get("/analysis/discoveries/{name}")
def analysis_discoveries(name: str):
    """Grounded research findings from the strategy's stored backtest trades — ranked observations (each
    computed from real trades, with evidence n) + the full attribution for drill-down. The 'what I found'."""
    from intelligence.strategy_analysis import discover_strategy
    try:
        return discover_strategy(name)
    except Exception as e:
        logger.error(f"discoveries failed: {e}", exc_info=True)
        return {"ready": False, "n": 0, "discoveries": [], "attribution": {}}


class AnalysisAskRequest(BaseModel):
    question: str
    name: str = ""
    direction: str = "up"
    require: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


@router.post("/analysis/ask")
def analysis_ask(req: AnalysisAskRequest):
    """Answer a money/hypothesis question grounded in the strategy's real backtest ledger (honest about
    R-multiples being a disciplined simulated result, not brokerage P&L)."""
    from intelligence.strategy_analysis import answer_analysis
    try:
        return {"answer": answer_analysis(req.question, req.name, req.direction, req.require, req.exclude)}
    except Exception as e:
        logger.error(f"analysis ask failed: {e}", exc_info=True)
        return {"answer": "I couldn't answer that just now."}


class TeachAskRequest(BaseModel):
    question: str
    direction: str = "up"
    require: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


@router.post("/teach/ask")
def teach_ask(req: TeachAskRequest):
    """Answer a question about the strategy/stats — grounded in the real numbers, honest that these are
    RECOGNITION stats (did the move run), not profit/loss."""
    from intelligence.teaching_service import answer_question
    try:
        return {"answer": answer_question(req.question, req.direction, req.require, req.exclude)}
    except Exception as e:
        logger.error(f"teach ask failed: {e}", exc_info=True)
        return {"answer": "I couldn't answer that just now."}


class TeachReviewSigRequest(BaseModel):
    direction: str = "up"
    require: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


@router.post("/teach/review-signature")
def teach_review_signature(req: TeachReviewSigRequest):
    """Review a STORED signature (no text) — the current understanding (validation + matched examples +
    refinements). Used on REOPEN so a strategy shows what the system knows, not a replay of raw words."""
    from intelligence.teaching_service import review_signature
    try:
        return review_signature(req.direction, req.require, req.exclude).to_dict()
    except Exception as e:
        logger.error(f"review signature failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="review failed")


@router.get("/teach/sessions")
def teach_sessions():
    """The strategy-sessions list (= My Strategies) for the sidebar."""
    from intelligence.taught_strategies import list_sessions
    return {"sessions": list_sessions()}


@router.get("/teach/session/{name}")
def teach_session(name: str):
    """Reopen one strategy's full teaching session (signature + conversation + examples + versions)."""
    from intelligence.taught_strategies import load_session
    s = load_session(name)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@router.get("/strategies")
def list_taught_strategies():
    """The user's taught strategies, each with its VALIDATED recognition stats off the episodic library
    (money-move rate vs base, lift, validated?, per-stock + by-phase) — so the partner points only at
    edges that earned it. `vocab` = the eye's tag vocabulary the UI builds strategies from."""
    from intelligence.taught_strategies import strategies_with_stats, TAG_VOCAB, EXAMPLE_TEMPLATES
    try:
        # `strategies` = what the USER has taught (empty until they do); `examples` = inert one-click
        # starting points the UI may offer (NOT taught, never auto-stored); `vocab` = the tag palette.
        return {"strategies": strategies_with_stats(), "examples": EXAMPLE_TEMPLATES, "vocab": TAG_VOCAB}
    except Exception as e:
        logger.error(f"list strategies failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="failed to load strategies")


@router.post("/strategies")
def teach_strategy(req: TaughtStrategyRequest):
    """Teach (add/update) a strategy = a signature over the eye's vocabulary → instantly graded."""
    from intelligence.taught_strategies import add_strategy, library_stats, load_strategies
    try:
        add_strategy(req.name, req.direction, req.require, req.exclude)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"teach strategy failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="failed to save strategy")
    s = next((x for x in load_strategies() if x.name.lower() == req.name.strip().lower()), None)
    return {"ok": True, "stats": library_stats(s).to_dict() if s else None}


@router.delete("/strategies/{name}")
def forget_strategy(name: str):
    from intelligence.taught_strategies import remove_strategy
    return {"removed": remove_strategy(name)}


@router.get("/proactive")
def proactive_scan(symbols: str = "", as_of: str | None = None):
    """Run ONE proactive pass over `symbols` (comma-separated) → PUBLISH partner alerts onto the live
    feed (event bus → /ws/session → LiveAlertsPage/chat) AND return them. The frontend can poll this,
    and the background ProactiveLoop calls the same path. Sync def → threadpooled (heavy narration)."""
    from intelligence.proactive_loop import run_once
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="provide ?symbols=A,B,C")
    try:
        alerts = run_once(syms, as_of=as_of)
    except Exception as e:
        logger.error(f"proactive scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="proactive scan failed")
    return {"published": len(alerts), "alerts": [a.to_dict() for a in alerts]}


@router.get("/trade-review")
def trade_review_endpoint(symbol: str, direction: str, entry: str, exit: str = "",
                          entry_price: float = 0.0, exit_price: float = 0.0, timeframe: str = "5m"):
    """EYE-grounded trade review (the Learning-Journal core): what the eye SAW the moment you entered
    (with/against its read, its stance + exit-trigger) and at your exit (move still alive vs rolled
    over vs turned). `entry`/`exit` = ISO timestamps of your clicks. Works ad-hoc (no DB). Sync def."""
    from intelligence.trade_review import review_trade
    try:
        tr = review_trade(symbol, direction, entry, exit or None,
                          entry_price or None, exit_price or None, interval=timeframe)
    except Exception as e:
        logger.error(f"trade review failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="trade review failed")
    if not tr.ok:
        raise HTTPException(status_code=404, detail=tr.note)
    return tr.to_dict()


@router.get("/partner/{symbol}")
def partner_read_endpoint(symbol: str, timeframe: str = "5m", as_of: str | None = None,
                          position: str = "", trades_today: int = 0, pnl: float = 0.0):
    """THE UNIFIED PARTNER (Co-Planner + COACH merged) for a symbol: one brain read → coach voice +
    matched taught-strategy (with validated stats) + a position-aware co-planner ACTION. `position`
    = long/short (if in a trade) → manage it; else evaluate the setup. (User's taught strategies are a
    follow-on wire — v1 runs without them, so strategy_match is empty.) Sync def → threadpooled."""
    from agents.partner import partner_read, PartnerContext, Position
    from agents.coach import TraderContext
    from intelligence.taught_strategies import load_strategies
    pos = (Position(direction="up" if position.lower() in ("long", "up") else "down")
           if position.lower() in ("long", "short", "up", "down") else None)
    trader = (TraderContext(todays_trade_count=trades_today, current_pnl=pnl)
              if (trades_today or pnl or position) else None)
    try:
        pr = partner_read(PartnerContext(symbol, position=pos, trader=trader, strategies=load_strategies()),
                          interval=timeframe, as_of=as_of)
    except Exception as e:
        logger.error(f"partner read failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="partner read failed")
    if not pr.ok:
        raise HTTPException(status_code=404, detail=pr.note)
    return pr.to_dict()


@router.get("/coach/{symbol}")
def coach_read(symbol: str, timeframe: str = "5m", as_of: str | None = None,
               direction: str = "", trades_today: int = 0, pnl: float = 0.0):
    """THE COACH face for a symbol — the partner voice over the brain (cognition_service): stance +
    mechanism + grounded why + EXIT (the invalidation) + calibrated disagreement (vs `direction`) +
    discipline nudge (`trades_today`, `pnl`). Returns the structured CoachRead plus a rendered `speak`
    string for direct display. Sync def → threadpooled (heavy narration)."""
    from agents.coach import coach, TraderContext
    trader = (TraderContext(intended_direction=direction, todays_trade_count=trades_today, current_pnl=pnl)
              if (direction or trades_today or pnl) else None)
    try:
        c = coach(symbol, interval=timeframe, as_of=as_of, trader=trader)
    except Exception as e:
        logger.error(f"coach read failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="coach read failed")
    if not c.ok:
        raise HTTPException(status_code=404, detail=c.note)
    return {**c.to_dict(), "speak": c.speak()}


@router.get("/cognition/{symbol}")
def cognition(symbol: str, timeframe: str = "5m", as_of: str | None = None):
    """THE BRAIN'S READ (the single Cognition Service seam) for a symbol at a moment: eye state +
    staged Candidate→Money-Move + episodic recall + reasoner VERDICT (mechanism · consequence ·
    INVALIDATION, on Llama). This is the canonical cognition endpoint every face (COACH, co-planner,
    AVT, this page) calls; it supersedes the GEN-1 /evaluate path (kept until the frontend migrates).
    Defined as a SYNC def on purpose — FastAPI runs it in a threadpool, so the heavy narration does not
    block the event loop."""
    from intelligence.cognition_service import read as _read
    try:
        r = _read(symbol, interval=timeframe, as_of=as_of)
    except Exception as e:
        logger.error(f"cognition read failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="cognition read failed")
    if not r.ok:
        raise HTTPException(status_code=404, detail=r.note)
    return r.to_dict()


@router.post("/alerts")
async def get_alerts(
    request: AlertsRequest,
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """Get active setup alerts across watchlist."""
    try:
        alerts = await service.get_setup_alerts(
            symbols=request.symbols,
            trader_id=trader_id,
            min_priority=request.min_priority,
        )
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Alert fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Alert fetch failed")


@router.get("/suitability/{symbol}")
async def get_environment_suitability(
    symbol: str,
    timeframe: str = Query(default="5m"),
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Get environment suitability score with per-strategy breakdown.

    Quick dashboard endpoint showing which strategies are favorable
    in current conditions.
    """
    try:
        result = await service.get_environment_suitability(
            symbol=symbol,
            trader_id=trader_id,
            timeframe=timeframe,
        )
        return result
    except Exception as e:
        logger.error(f"Suitability check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Suitability check failed")


@router.get("/profile")
async def get_intelligence_profile(
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Get the trader's intelligence profile.

    Returns the system's deep model of the trader's style, risk
    preferences, timing, emotional patterns, and performance context.
    """
    try:
        profile = await service.get_trader_profile(trader_id)
        return profile.to_dict()
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Profile fetch failed")


@router.get("/market-overview")
async def get_market_overview(
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
):
    """
    Get a live snapshot of key Indian market indices + VIX.

    Returns prices and % change for NIFTY, BANKNIFTY, FINNIFTY, SENSEX, INDIA VIX
    and a short market bias summary (bullish / bearish / neutral + reasoning).

    FNO traders use this to understand underlying direction before placing trades
    on constituent stocks.  This endpoint does NOT require a symbol input —
    it always returns all tracked indices.
    """
    from ingestion.market_data_service import get_market_data_service
    from datetime import datetime, timezone

    KEY_INDICES = [
        ("NIFTY",     "Nifty 50",          "NSE"),
        ("BANKNIFTY", "Bank Nifty",         "NSE"),
        ("FINNIFTY",  "Nifty Fin Service",  "NSE"),
        ("SENSEX",    "BSE Sensex",         "BSE"),
        ("INDIA VIX", "India VIX",          "NSE"),
    ]

    mds = get_market_data_service()
    index_data = []

    for symbol, display_name, exchange in KEY_INDICES:
        entry: dict = {
            "symbol": symbol,
            "name": display_name,
            "exchange": exchange,
            "ltp": None,
            "change": None,
            "change_pct": None,
            "prev_close": None,
        }
        if mds and mds.has_live:
            try:
                quote = mds.get_market_quote(symbol)
                if quote.get("ltp"):
                    entry["ltp"] = round(quote["ltp"], 2)
                    prev = quote.get("close") or quote.get("prev_close")
                    if prev and prev > 0:
                        entry["prev_close"] = round(prev, 2)
                        entry["change"] = round(quote["ltp"] - prev, 2)
                        entry["change_pct"] = round((quote["ltp"] - prev) / prev * 100, 2)
            except Exception as exc:
                logger.warning("Market overview: failed to fetch %s — %s", symbol, exc)
        index_data.append(entry)

    # Derive simple market bias from Nifty + BankNifty
    bias = "neutral"
    bias_reasons = []
    nifty_row = next((r for r in index_data if r["symbol"] == "NIFTY"), None)
    bank_row   = next((r for r in index_data if r["symbol"] == "BANKNIFTY"), None)
    vix_row    = next((r for r in index_data if r["symbol"] == "INDIA VIX"), None)

    if nifty_row and nifty_row.get("change_pct") is not None:
        pct = nifty_row["change_pct"]
        if pct > 0.5:
            bias = "bullish"
            bias_reasons.append(f"Nifty up {pct:.2f}%")
        elif pct < -0.5:
            bias = "bearish"
            bias_reasons.append(f"Nifty down {abs(pct):.2f}%")
        else:
            bias_reasons.append(f"Nifty flat ({pct:+.2f}%)")

    if bank_row and bank_row.get("change_pct") is not None:
        bpct = bank_row["change_pct"]
        if bpct > 1.0:
            bias_reasons.append(f"Bank Nifty strong +{bpct:.2f}%")
        elif bpct < -1.0:
            bias_reasons.append(f"Bank Nifty weak {bpct:.2f}%")

    if vix_row and vix_row.get("ltp") is not None:
        vix = vix_row["ltp"]
        if vix > 20:
            bias_reasons.append(f"VIX elevated at {vix:.1f} — high volatility, tighten sizing")
        elif vix < 13:
            bias_reasons.append(f"VIX low at {vix:.1f} — calm conditions, breakouts reliable")
        else:
            bias_reasons.append(f"VIX normal at {vix:.1f}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_bias": bias,
        "bias_summary": " | ".join(bias_reasons) if bias_reasons else "Insufficient data",
        "indices": index_data,
        "fno_note": (
            "These are spot index prices. FNO positions trade on Futures/Options contracts "
            "that track these underlyings — use these prices to gauge market direction, "
            "not for direct entry/exit."
        ),
    }


@router.get("/cognitive-alerts")
async def get_cognitive_alerts(
    service: RealtimeIntelligenceService = Depends(get_intelligence_service),
    trader_id: str = Depends(get_current_trader_id),
):
    """
    Retrieve recent cognitive support alerts for the trader.

    Currently this endpoint returns an empty list (stub); later it will
    surface stop‑widen, oversize, volatility and other real‑time alerts
    generated during live trading.
    """
    try:
        alerts = await service.get_recent_cognitive_alerts(trader_id)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Cognitive alert fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cognitive alert fetch failed")


# ─── WebSocket: Real-Time Alert Feed ─────────────────────────────


@router.websocket("/ws/alerts/{trader_id}")
async def intelligence_alert_feed(
    websocket: WebSocket,
    trader_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket feed for real-time setup alerts.

    Pushes alerts as they're detected during scanning:
    - Setup forming alerts
    - Setup triggered alerts
    - Regime change alerts
    - Structure break alerts

    Client sends:
      {"action": "subscribe", "symbols": ["AAPL", "NVDA", "TSLA"]}
      {"action": "unsubscribe", "symbols": ["TSLA"]}

    Server pushes:
      {"type": "alert", "data": {...}}
      {"type": "environment_update", "data": {...}}
      {"type": "scan_result", "data": {...}}
    """
    await websocket.accept()
    logger.info(f"Intelligence WebSocket connected for trader {trader_id}")

    subscribed_symbols: set[str] = set()
    scan_task: Optional[asyncio.Task] = None

    # Initialize service
    from config.settings import settings
    from strategy_engine.strategy_registry import StrategyRegistry
    from strategy_engine.conditions.registry import ConditionRegistry
    from intelligence.price_fetcher import get_price_fetcher

    import asyncio

    registry = StrategyRegistry()
    registry.initialize()          # load YAML templates
    condition_registry = ConditionRegistry()
    price_fetcher = get_price_fetcher(settings)

    service = RealtimeIntelligenceService(
        db=db,
        strategy_registry=registry,
        condition_registry=condition_registry,
        price_fetcher=price_fetcher,
    )

    # Register alert callback
    async def push_alert(alert: dict):
        try:
            await websocket.send_json({
                "type": "alert",
                "data": alert,
            })
        except Exception:
            pass

    service.on_alert(push_alert)

    # Background scanning loop
    async def scan_loop():
        while True:
            try:
                for symbol in list(subscribed_symbols):
                    try:
                        result = await service.scan_setups(
                            symbol=symbol,
                            trader_id=trader_id,
                            force_refresh=True,
                        )

                        # Push scan result summary
                        if result.top_setup and result.top_setup.readiness_score > 0.50:
                            await websocket.send_json({
                                "type": "scan_result",
                                "data": {
                                    "symbol": symbol,
                                    "top_setup": result.top_setup.to_dict(),
                                    "alert_count": len(result.alerts),
                                },
                            })

                    except Exception as e:
                        logger.warning(f"Scan loop error for {symbol}: {e}")

                await asyncio.sleep(15)  # Scan every 15 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(30)

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                })
                continue

            action = data.get("action")

            if action == "subscribe":
                symbols = data.get("symbols", [])
                subscribed_symbols.update(symbols)

                # Start scanning if not already running
                if scan_task is None or scan_task.done():
                    scan_task = asyncio.create_task(scan_loop())

                await websocket.send_json({
                    "type": "subscribed",
                    "data": {"symbols": list(subscribed_symbols)},
                })

                # Immediately send current state for new symbols
                for symbol in symbols:
                    try:
                        env = await service.get_environment(symbol)
                        await websocket.send_json({
                            "type": "environment_update",
                            "data": env.to_dict(),
                        })
                    except Exception:
                        pass

            elif action == "unsubscribe":
                symbols = data.get("symbols", [])
                subscribed_symbols -= set(symbols)

                if not subscribed_symbols and scan_task:
                    scan_task.cancel()
                    scan_task = None

                await websocket.send_json({
                    "type": "unsubscribed",
                    "data": {"symbols": list(subscribed_symbols)},
                })

            elif action == "evaluate":
                # Real-time discretionary query via WebSocket
                query = data.get("query", "")
                symbol = data.get("symbol", "")

                if query and symbol:
                    await websocket.send_json({
                        "type": "evaluating",
                        "data": {"symbol": symbol},
                    })

                    try:
                        assessment = await service.evaluate_discretionary(
                            query=query,
                            symbol=symbol,
                            trader_id=trader_id,
                            direction=data.get("direction"),
                            todays_trade_count=data.get("todays_trade_count", 0),
                            current_pnl=data.get("current_pnl", 0.0),
                        )

                        await websocket.send_json({
                            "type": "evaluation",
                            "data": assessment.to_dict(),
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": f"Evaluation failed: {str(e)}"},
                        })

    except WebSocketDisconnect:
        logger.info(f"Intelligence WebSocket disconnected for trader {trader_id}")
    except Exception as e:
        logger.error(f"Intelligence WebSocket error: {e}", exc_info=True)
    finally:
        if scan_task and not scan_task.done():
            scan_task.cancel()