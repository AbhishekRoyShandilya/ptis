"""
avt.replay — historical replay of AVT over the lake (the clean-edge backfill).

AVT normally runs FORWARD (live scanner loop). That means the strategy-research
record — and therefore the edge-decay matrix — would start empty the day a user
teaches a strategy and fill only slowly. This driver replays the user's TAUGHT
strategies over the lake's history (2022→today) so edge-decay has depth
immediately.

How it stays honest (point-in-time, no lookahead):
  • The engine's clock is driven by the current BAR's timestamp (engine.set_clock),
    so every virtual trade is stamped at its real historical time — and
    marketdata.strategy_research then freezes the regime/features as-of THAT time.
  • ReplayMarketData.get_candles only returns bars at/▸before the cursor; signals
    are generated on a bar's close, entries fill at that close, and exits resolve
    on LATER bars (the engine skips a position's own entry bar).
  • OHLC-aware fills: stops/targets fill against each bar's low/high at the
    stop/target price (conservative: stop-before-target if both touched).

TWO MODES (distinct scientific experiments — never conflated):
  • BENCHMARK (default): FROZEN strategy v1.0, self-improver OFF, no lifecycle
    gate. Answers "does the strategy have edge, and where?" — the canonical source
    for the edge matrix AND per-(strategy,stock) ranking initialization. Market is
    the only moving part, so edge-decay is actually measurable.
  • LEARNING (later): adaptive (self-improver ON, v1.0→v1.1→…). Answers "can AVT
    improve the strategy?" — a separate research tool, NEVER a ranking source.
    Not built yet (requires a simulated clock in self_improver); raises clearly.

LOOKAHEAD SAFETY (benchmark): anything used at simulated date D was knowable at D.
  • FIXED universe (Nifty100/FNO, defined once) — NOT today's matched watchlist,
    which would inject today's stock-selection knowledge into 2022.
  • Point-in-time candles (ReplayMarketData cursor); FROZEN v1.0 (no self-improver);
    StockSelectionRanker's variant-perf cache stays empty (only the self-improver
    fills it). So no live performance leaks backward.

Each strategy is replayed at ITS OWN timeframe over the SAME fixed universe (so
per-strategy edge is independent and comparable). Writes AVT virtual trades to the
DB exactly like the live engine; the CLI then chains:
    strategy_research → edge_matrix → strategy_symbol_perf   (edge + ranking init)

Usage:
    python -m avt.replay --trader <uuid> [--from 2022-01-01] [--to today]
        [--strategy <sko_id>] [--universe nifty100|fno] [--max-symbols N]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_INTRADAY_TFS = {"1m", "5m", "15m", "1h"}


def _norm_tf(v: str) -> str:
    v = str(v).strip().lower()
    table = {
        "1m": "1m", "1min": "1m", "1minute": "1m", "minute": "1m",
        "5m": "5m", "5min": "5m", "5minute": "5m",
        "15m": "15m", "15min": "15m", "15minute": "15m",
        "1h": "1h", "60m": "1h", "60min": "1h", "60minute": "1h", "hour": "1h",
        "1d": "1d", "1day": "1d", "day": "1d", "daily": "1d", "eod": "1d",
    }
    return table.get(v, v if v in {"1m", "5m", "15m", "1h", "1d"} else "5m")


def _strategy_timeframe(sko: Dict[str, Any]) -> str:
    """Best-effort native timeframe of a taught strategy; defaults to 5m (live AVT default)."""
    for src in (sko.get("time_preferences") or {}, sko.get("risk_parameters") or {},
                sko.get("scoring_config") or {}):
        if not isinstance(src, dict):
            continue
        for k in ("timeframe", "primary_timeframe", "entry_timeframe", "tf", "interval"):
            v = src.get(k)
            if v:
                return _norm_tf(str(v))
    return "5m"


def _candles_as_dicts(df) -> List[Dict[str, Any]]:
    """Lake DataFrame → list of OHLCV dicts (sorted by ts) for ReplayMarketData."""
    if df is None or df.empty:
        return []
    df = df.sort_values("ts")
    out: List[Dict[str, Any]] = []
    for ts, o, h, l, c, v in zip(
        df["ts"], df["open"], df["high"], df["low"], df["close"], df["volume"]
    ):
        out.append({
            "ts": ts,
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(v or 0),
        })
    return out


async def _replay_one_strategy(
    db_factory: Callable,
    sko: Dict[str, Any],
    symbols: List[str],
    tf: str,
    start: datetime,
    end: datetime,
    trader_id: Optional[str],
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Replay one taught strategy over its symbols at its timeframe; persist trades."""
    from avt.engine import AutonomousVirtualTrader, ReplayMarketData
    from ske.signal_generator import SignalGenerator
    from avt.stock_ranker import StockSelectionRanker
    from marketdata.candle_store import get_candle_store

    sid = str(sko.get("id") or sko.get("short_code"))
    store = get_candle_store()
    market = ReplayMarketData()

    # Load candles per symbol + build a unified, sorted bar timeline.
    timeline: set = set()
    loaded = 0
    for sym in symbols:
        try:
            candles = _candles_as_dicts(store.get(sym, tf, start, end))
        except Exception as e:
            logger.warning("replay: candle load failed %s/%s: %s", sym, tf, e)
            continue
        if len(candles) < 30:
            continue
        market.load_candles(sym, candles)
        timeline.update(c["ts"] for c in candles)
        loaded += 1
    if loaded == 0 or not timeline:
        return {"strategy": sid, "tf": tf, "symbols": 0, "bars": 0, "trades": 0,
                "note": "no candles in lake for this strategy's symbols/timeframe"}

    bars = sorted(timeline)

    # Build a single-strategy replay engine (NOT registered in the live registry).
    market_for_ranker = market
    engine = AutonomousVirtualTrader(
        strategies=[sko],
        trader_id=trader_id,
        market_data=market,
        signal_generator=SignalGenerator(),
        stock_ranker=StockSelectionRanker(market_data=market_for_ranker),
        watchlist=symbols,
    )
    engine.set_strategy_watchlists({sid: symbols})

    # Inject the replay clock (current bar timestamp) + intraday-close policy.
    clock_holder: Dict[str, Any] = {"t": bars[0]}
    engine.set_clock(lambda: clock_holder["t"],
                     force_intraday_close=(tf in _INTRADAY_TFS))

    # Walk the timeline.
    n = len(bars)
    for i, t in enumerate(bars):
        clock_holder["t"] = t
        market.seek(t)
        try:
            await engine.run_scan_cycle()
        except Exception as e:
            logger.debug("replay cycle error @ %s: %s", t, e)
        if progress and (i % 5000 == 0):
            progress(f"  {sko.get('short_code', sid)} [{tf}]: bar {i}/{n} "
                     f"({t}) — open={len(engine.portfolio.open_positions)} "
                     f"closed={len(engine.portfolio.closed_trades)}")

    # Force-close anything still open at the last bar (so every trade has an outcome).
    last_ts = bars[-1]
    for pos in list(engine.portfolio.open_positions):
        bar = market.get_current_bar(pos.symbol)
        last_close = float(bar.get("close")) if bar else pos.entry_price
        engine.portfolio.close_position(
            trade=pos, exit_price=last_close, exit_time=last_ts, exit_type="replay_end",
        )

    # Persist all closed virtual trades to the DB (same table the live engine uses).
    saved = _persist_trades(db_factory, engine.portfolio.closed_trades)
    return {"strategy": sid, "short_code": sko.get("short_code"), "tf": tf,
            "symbols": loaded, "bars": n, "trades": saved}


def _persist_trades(db_factory: Callable, trades: List[Any]) -> int:
    """Bulk-save closed virtual trades via the AVT repository (one commit)."""
    if not trades:
        return 0
    from avt.repository import AVTTradeRepository
    db = db_factory()
    if hasattr(db, "__next__"):
        db = next(db)
    saved = 0
    try:
        repo = AVTTradeRepository(db)
        for t in trades:
            try:
                repo.save(t)
                saved += 1
            except Exception as e:
                logger.debug("replay persist failed for %s: %s", getattr(t, "id", "?"), e)
        db.commit()
    finally:
        db.close()
    return saved


def _fixed_universe(universe: str = "nifty100", max_symbols: int = 150) -> List[str]:
    """The FIXED replay universe — NOT the user's live watchlist (no hindsight).

    'nifty100' = Nifty50 + Next50 (broad, bounded); 'fno' = full FNO list.
    Deterministic + the same for every strategy, so per-strategy edge/ranking is
    comparable and contains zero stock-selection lookahead.
    """
    from automation.fno_universe import FNOUniverseFetcher
    f = FNOUniverseFetcher()
    if universe == "fno":
        syms = f.get_symbols() or (f.get_nifty50() + f.get_nifty_next50())
    else:  # nifty100 (default)
        syms = f.get_nifty50() + f.get_nifty_next50()
    deduped: List[str] = []
    for s in syms:
        if s and s not in deduped:
            deduped.append(s)
    return deduped[:max_symbols]


async def replay_user(
    db_factory: Callable,
    trader_id: Optional[str],
    start: datetime,
    end: datetime,
    only_strategy: Optional[str] = None,
    universe: str = "nifty100",
    max_symbols: int = 150,
    mode: str = "benchmark",
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Replay ALL of a user's taught strategies (BENCHMARK mode) → persist AVT trades.

    mode="benchmark" (default): frozen v1.0, no learning — the canonical edge +
    ranking source. mode="learning" is a separate research tool (adaptive, needs a
    simulated clock in self_improver) and is NOT YET IMPLEMENTED.
    """
    if mode == "learning":
        raise NotImplementedError(
            "Learning (adaptive) replay is specified but not built yet — it needs a "
            "simulated clock in avt/self_improver.py so the 3-week proof window can "
            "elapse in replay (see Phase 3 of the AVT replay plan). Use mode='benchmark' "
            "for edge + ranking; learning replay is a separate research tool and never "
            "a ranking source."
        )
    if mode != "benchmark":
        raise ValueError(f"unknown replay mode {mode!r} (expected 'benchmark' or 'learning')")

    from avt.integrations import _load_tradeable_skos

    skos = _load_tradeable_skos(db_factory, trader_id)
    if only_strategy:
        skos = [s for s in skos if str(s.get("id")) == only_strategy
                or s.get("short_code") == only_strategy]
    if not skos:
        return {"strategies": 0, "results": [],
                "note": "no taught/compiled strategies for this user — teach one first"}

    # FIXED universe for ALL strategies (lookahead-safe; not the live watchlist).
    symbols = _fixed_universe(universe, max_symbols)
    if progress:
        progress(f"fixed replay universe: {len(symbols)} symbols ({universe})")

    results = []
    for sko in skos:
        sid = str(sko.get("id"))
        tf = _strategy_timeframe(sko)
        if progress:
            progress(f"▶ replaying {sko.get('short_code', sid)} [{tf}] over {len(symbols)} symbols…")
        res = await _replay_one_strategy(
            db_factory, sko, symbols, tf, start, end, trader_id, progress,
        )
        results.append(res)
        if progress:
            progress(f"✓ {res.get('short_code', sid)}: {res.get('trades', 0)} virtual trades "
                     f"from {res.get('bars', 0)} bars × {res.get('symbols', 0)} symbols")
    return {"strategies": len(skos), "results": results, "universe": universe,
            "universe_size": len(symbols)}


# ── Background job runner (frontend-initiated, user-scoped) ──────────────
# The replay is a USER action ("I taught my strategies — build my baseline"),
# not a dev CLI. The API triggers it via start_replay_job; because it's CPU-heavy
# and bar-by-bar it runs in a background THREAD (its own asyncio loop) so the API
# event loop never blocks, with a per-user status registry the frontend polls.
# One replay per user at a time. (Multi-user scale → move to a task queue.)

_REPLAY_JOBS: Dict[str, Dict[str, Any]] = {}
_REPLAY_LOCK = threading.Lock()
_PROGRESS_KEEP = 60


def _job_key(trader_id: Optional[str]) -> str:
    return str(trader_id) if trader_id is not None else "__default__"


def replay_job_status(trader_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Current replay job status for a user (or None if never run)."""
    with _REPLAY_LOCK:
        st = _REPLAY_JOBS.get(_job_key(trader_id))
        return dict(st) if st else None


def start_replay_job(
    trader_id: Optional[str],
    db_factory: Callable,
    *,
    start: datetime,
    end: datetime,
    only_strategy: Optional[str] = None,
    universe: str = "nifty100",
    max_symbols: int = 150,
    mode: str = "benchmark",
) -> Dict[str, Any]:
    """Start a benchmark replay for one user in a background thread. Idempotent:
    if one is already running for this user, returns its current status."""
    key = _job_key(trader_id)
    with _REPLAY_LOCK:
        existing = _REPLAY_JOBS.get(key)
        if existing and existing.get("state") == "running":
            return dict(existing)
        status = {
            "state": "running", "message": "starting…",
            "started_at": datetime.utcnow().isoformat(), "finished_at": None,
            "params": {"from": start.isoformat(), "to": end.isoformat(),
                       "universe": universe, "max_symbols": max_symbols,
                       "mode": mode, "strategy": only_strategy},
            "progress": [], "results": None, "error": None,
        }
        _REPLAY_JOBS[key] = status

    def _update(msg: str) -> None:
        with _REPLAY_LOCK:
            st = _REPLAY_JOBS.get(key)
            if st is not None:
                st["message"] = msg
                st["progress"] = (st["progress"] + [msg])[-_PROGRESS_KEEP:]

    def _run() -> None:
        try:
            stats = asyncio.run(replay_user(
                db_factory, trader_id, start, end, only_strategy=only_strategy,
                universe=universe, max_symbols=max_symbols, mode=mode, progress=_update,
            ))
            total = sum(r.get("trades", 0) for r in stats.get("results", []))
            agg: Dict[str, Any] = {}
            if total > 0:
                _update("building strategy_research + edge/symbol matrices…")
                from marketdata.strategy_research import build_from_avt
                from analytics.edge_matrix import recompute as rc_edge
                from analytics.strategy_symbol_perf import recompute as rc_sym
                db = db_factory()
                if hasattr(db, "__next__"):
                    db = next(db)
                try:
                    sr = build_from_avt(db, trader_id=trader_id)
                finally:
                    db.close()
                agg = {"strategy_research": sr,
                       "edge_rows": rc_edge(db_factory),
                       "symbol_rows": rc_sym(db_factory)}
                # If this user's AVT is already live, refresh its ranking baseline now
                # so the just-built per-stock edge takes effect without a restart.
                try:
                    from avt.engine import get_avt
                    eng = get_avt(trader_id)
                    if eng is not None and getattr(eng, "_stock_ranker", None) is not None:
                        rdb = db_factory()
                        if hasattr(rdb, "__next__"):
                            rdb = next(rdb)
                        try:
                            eng._stock_ranker.load_symbol_performance(rdb, trader_id)
                        finally:
                            rdb.close()
                except Exception as e:
                    logger.debug("post-replay ranker refresh skipped: %s", e)
            with _REPLAY_LOCK:
                st = _REPLAY_JOBS.get(key)
                if st is not None:
                    st["state"] = "done"
                    st["message"] = f"done — {total} virtual trades, ranking baseline built"
                    st["finished_at"] = datetime.utcnow().isoformat()
                    st["results"] = {**stats, "total_trades": total, **agg}
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            logger.exception("replay job failed for %s", key)
            with _REPLAY_LOCK:
                st = _REPLAY_JOBS.get(key)
                if st is not None:
                    st["state"] = "error"
                    st["error"] = str(e)
                    st["message"] = f"error: {e}"
                    st["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_run, daemon=True, name=f"avt-replay-{key[:8]}").start()
    return replay_job_status(trader_id)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Replay AVT over lake history → clean strategy edge")
    p.add_argument("--trader", default=None, help="trader_id whose strategies to replay")
    p.add_argument("--from", dest="frm", default="2022-01-01")
    p.add_argument("--to", dest="to", default=date.today().isoformat())
    p.add_argument("--strategy", default=None, help="restrict to one sko_id / short_code")
    p.add_argument("--mode", default="benchmark", choices=["benchmark", "learning"],
                   help="benchmark=frozen v1.0 (edge+ranking); learning=adaptive (not yet built)")
    p.add_argument("--universe", default="nifty100", choices=["nifty100", "fno"],
                   help="FIXED replay universe (default nifty100); not the live watchlist")
    p.add_argument("--max-symbols", type=int, default=150)
    p.add_argument("--no-edge-matrix", action="store_true",
                   help="skip building strategy_research + edge/symbol matrices afterwards")
    args = p.parse_args(argv)

    start = datetime.fromisoformat(args.frm)
    end = datetime.fromisoformat(args.to)

    from core.database import SessionLocal
    stats = asyncio.run(replay_user(
        SessionLocal, args.trader, start, end,
        only_strategy=args.strategy, universe=args.universe, max_symbols=args.max_symbols,
        mode=args.mode, progress=lambda m: print(m, flush=True),
    ))
    total = sum(r.get("trades", 0) for r in stats.get("results", []))
    print(f"\nAVT replay complete: {total} virtual trades across "
          f"{stats.get('strategies', 0)} strategies")
    if stats.get("note"):
        print(f"note: {stats['note']}")

    if not args.no_edge_matrix and total > 0:
        print("\nBuilding strategy_research + edge/symbol matrices from the new AVT trades…")
        from marketdata.strategy_research import build_from_avt
        from analytics.edge_matrix import recompute as recompute_edge
        from analytics.strategy_symbol_perf import recompute as recompute_symbol
        db = SessionLocal()
        try:
            sr = build_from_avt(db, trader_id=args.trader)
        finally:
            db.close()
        print(f"strategy_research: {sr['captured']} samples from {sr['avt_trades']} AVT trades")
        ne = recompute_edge(SessionLocal)
        print(f"edge_matrix: upserted {ne} (trader, strategy, context) rows")
        ns = recompute_symbol(SessionLocal)
        print(f"strategy_symbol_perf: upserted {ns} (trader, strategy, symbol, …) rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
