"""
marketdata.backfill — one-time backfill + incremental update for the OHLCV lake.

Usage:
  python -m marketdata.backfill --instruments
  python -m marketdata.backfill --symbol RELIANCE [--interval 1m|1d|both] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
  python -m marketdata.backfill --all   [--limit N]
  python -m marketdata.backfill --update

Backfilling a symbol just calls CandleStore.get(...) over the full range — the
store fetches only the missing tail and writes Parquet, so re-runs are idempotent
and `--update` naturally fetches just new bars (end > stored_max).

Set PTIS_ENV=test to target the test lake (./data/test/lake). Requires a working
Upstox token (or Angel One / yfinance fallback) to actually fetch candles.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from marketdata import instruments, lake
from marketdata.candle_store import get_candle_store

logger = logging.getLogger(__name__)

DEFAULT_1M_START = date(2022, 1, 1)   # Upstox v3 floor for 1-min
DEFAULT_1D_START = date(2000, 1, 1)   # daily floor

INDEX_LIST = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "INDIA VIX"]


def backfill_symbol(symbol: str, interval: str, frm: date, to: date) -> int:
    store = get_candle_store()
    df = store.get(symbol, interval, frm, to)
    n = len(df)
    logger.info("backfill %-14s %-3s %s→%s : %d rows in lake", symbol, interval, frm, to, n)
    return n


def backfill_one(symbol: str, which: str, to: date, refetch_tail_days: int = 0) -> None:
    # On incremental updates, re-open a rolling recent window so bars that
    # published AFTER a day was marked covered (e.g. Friday's, fetched over the
    # weekend as 0-bars) get caught — see lake.forget_coverage_after.
    if refetch_tail_days > 0:
        cutoff = to - timedelta(days=refetch_tail_days)
        if which in ("1d", "both"):
            lake.forget_coverage_after(symbol, "1d", cutoff)
        if which in ("1m", "both"):
            lake.forget_coverage_after(symbol, "1m", cutoff)
    if which in ("1d", "both"):
        backfill_symbol(symbol, "1d", DEFAULT_1D_START, to)
    if which in ("1m", "both"):
        backfill_symbol(symbol, "1m", DEFAULT_1M_START, to)


def universe() -> list[str]:
    syms: list[str] = []
    try:
        from automation.fno_universe import FNOUniverseFetcher
        syms = list(FNOUniverseFetcher().get_symbols() or [])
    except Exception as e:
        logger.warning("FNO universe unavailable (%s) — indices only", e)
    # Indices first, then equities, de-duped preserving order
    seen, ordered = set(), []
    for s in INDEX_LIST + syms:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            ordered.append(s)
    return ordered


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="OHLCV lake backfill")
    p.add_argument("--instruments", action="store_true", help="refresh the instrument dimension")
    p.add_argument("--symbol", help="backfill a single symbol")
    p.add_argument("--interval", default="both", choices=["1m", "1d", "both"])
    p.add_argument("--from", dest="frm", help="start date YYYY-MM-DD (override default)")
    p.add_argument("--to", dest="to", help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--all", action="store_true", help="backfill the FNO universe + indices")
    p.add_argument("--update", action="store_true", help="incremental: fetch only new tail bars")
    p.add_argument("--setup", action="store_true",
                   help="ONE-SHOT: refresh instruments + backfill the whole universe (1d + 1m)")
    p.add_argument("--limit", type=int, default=0, help="cap number of symbols (with --all/--setup)")
    p.add_argument("--features", action="store_true",
                   help="populate the FEATURE STORE (compute+store features) for the universe (or --symbol)")
    p.add_argument("--feature-intervals", default="",
                   help="comma-list of intervals for --features (default: 5m,15m,1h,1d)")
    args = p.parse_args(argv)

    # Connect a broker so the lake's lazy fetch has a live source. The app does this at startup, but the
    # backfill runs standalone (MDS was None → 0 fetch → stale lake). Angel One auto-logs-in via TOTP
    # (24h session, no daily manual step) and serves fresh historical candles.
    try:
        from ingestion.market_data_service import init_market_data_service
        _mds = init_market_data_service()
        if _mds is not None:
            logger.info("backfill: market data connected (angel=%s upstox=%s)",
                        bool(getattr(_mds, "_angel", None)), bool(getattr(_mds, "_upstox", None)))
        else:
            logger.warning("backfill: NO market data source connected — will only read existing lake")
    except Exception as _e:
        logger.warning("backfill: market data init failed: %s", _e)

    to = date.fromisoformat(args.to) if args.to else date.today()

    # --setup = instruments + full universe backfill in one command.
    if args.setup:
        args.instruments = True
        args.all = True

    if args.instruments:
        n = instruments.refresh_instruments()
        print(f"instruments: {n} contracts written")

    if args.symbol:
        if args.frm:
            frm = date.fromisoformat(args.frm)
            backfill_symbol(args.symbol, "1m" if args.interval == "both" else args.interval, frm, to)
        else:
            backfill_one(args.symbol, args.interval, to)

    if args.all or args.update:
        syms = universe()
        if args.limit:
            syms = syms[: args.limit]
        logger.info("%s %d symbols up to %s", "update" if args.update else "backfill", len(syms), to)
        # --update re-opens the last ~10 days (≈7 trading days) per symbol so
        # late-published bars self-heal; --all backfills from scratch (no trim).
        tail = 10 if args.update else 0
        import time as _time
        for i, s in enumerate(syms, 1):
            try:
                backfill_one(s, args.interval, to, refetch_tail_days=tail)
            except Exception as e:
                logger.warning("backfill failed for %s: %s", s, e)
            _time.sleep(0.9)   # throttle — Angel One rate-limits bursts ("exceeding access rate");
            #                    each symbol = several requests, so keep ~1s between symbols.
            if i % 25 == 0:
                logger.info("… %d/%d symbols done", i, len(syms))

    # Feature Store population — after candles exist. `--features --all` does the whole universe;
    # `--features --symbol X` does one. Idempotent; re-runnable. (project_brain/08)
    if args.features:
        from marketdata import features as F
        ivs = [s.strip() for s in args.feature_intervals.split(",") if s.strip()] or None
        frm = date.fromisoformat(args.frm) if args.frm else None
        if args.symbol:
            syms = [args.symbol]
        else:
            syms = universe()
            if args.limit:
                syms = syms[: args.limit]
        logger.info("features: backfilling %d symbol(s) × %s up to %s",
                    len(syms), ivs or F.FEATURE_INTERVALS, to)
        rows = F.backfill_features(syms, intervals=ivs, start=frm, to=to)
        print(f"features: {rows} rows written to the store")

    if not any([args.instruments, args.symbol, args.all, args.update, args.setup, args.features]):
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
