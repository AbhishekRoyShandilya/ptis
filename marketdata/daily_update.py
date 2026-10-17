"""
marketdata.daily_update — the nightly market-data + intelligence refresh.

Run by Windows Task Scheduler (update_market_data.bat) after market close on
trading days. In order:

  1. CANDLES — refresh the instrument dimension + self-healing tail-fetch of new
     1-min/daily bars for the whole universe (marketdata.backfill --update).
  2. CORPORA — recompute the RECENT TAIL of the analog/pattern corpora so the
     forward intelligence (analog_engine / state_transitions) reflects the newest
     bars and recently-incomplete forward outcomes get finalised. Tiered by
     decision horizon: 1d/1h/15m for the full universe; 5m for index products only.
     Only the tail is rewritten (features computed on a warm-up window via
     patterns.build_corpus(write_from=...)), so it's minutes, not a full rebuild.
  3. MATRICES — best-effort recompute of the edge matrix + per-stock ranking from
     whatever clean AVT trades exist (no-op until a baseline is built).

Idempotent + resumable: every step dedupes/heals, so a failed run is fixed by the
next. Usage:  python -m marketdata.daily_update [--no-candles] [--no-corpora] [--no-matrices]
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Per-timeframe (load_days, write_days): load enough warm-up for the 252-bar
# indicators (ATR percentile, EMA50) before the tail we actually rewrite.
_TF_WINDOWS = {
    "1d":  (450, 60),
    "1h":  (140, 45),
    "15m": (90, 45),
    "5m":  (60, 30),
}
_GLOBAL_TFS = ["1d", "1h", "15m"]          # full universe
_INDEX_TFS = {"5m": ["NIFTY", "BANKNIFTY", "FINNIFTY"]}   # index products only


def _refresh_corpus(tf: str, symbols: list[str], to: date) -> tuple[int, int]:
    from marketdata import patterns
    load, write = _TF_WINDOWS[tf]
    start, write_from = to - timedelta(days=load), to - timedelta(days=write)
    total = errors = 0
    for s in symbols:
        try:
            total += patterns.build_and_store(s, tf, start, to, write_from=write_from)
        except Exception as e:
            errors += 1
            logger.warning("corpus refresh failed %s/%s: %s", s, tf, e)
    logger.info("corpus %s: +%d tail rows (%s..%s), %d errors", tf, total, write_from, to, errors)
    return total, errors


def run(update_candles: bool = True, rebuild_corpora: bool = True,
        recompute_matrices: bool = True) -> None:
    from marketdata import backfill
    to = date.today()

    # 1. CANDLES (self-healing tail fetch; see lake.forget_coverage_after)
    if update_candles:
        logger.info("daily_update: refreshing instruments + candle tail …")
        backfill.main(["--instruments", "--update"])

    # 2. CORPORA (recent tail only)
    if rebuild_corpora:
        syms = backfill.universe()
        for tf in _GLOBAL_TFS:
            _refresh_corpus(tf, syms, to)
        for tf, idx in _INDEX_TFS.items():
            _refresh_corpus(tf, idx, to)

    # 3. MATRICES (best-effort — no-op until clean AVT trades exist)
    if recompute_matrices:
        try:
            from core.database import SessionLocal
            from analytics.edge_matrix import recompute as rc_edge
            from analytics.strategy_symbol_perf import recompute as rc_sym
            ne, ns = rc_edge(SessionLocal), rc_sym(SessionLocal)
            logger.info("matrices: edge=%d rows, symbol=%d rows", ne, ns)
        except Exception as e:
            logger.warning("matrix recompute skipped: %s", e)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="PTIS nightly market-data + intelligence refresh")
    p.add_argument("--no-candles", action="store_true", help="skip the candle tail-fetch")
    p.add_argument("--no-corpora", action="store_true", help="skip the analog-corpus refresh")
    p.add_argument("--no-matrices", action="store_true", help="skip edge/symbol matrix recompute")
    a = p.parse_args(argv)
    run(update_candles=not a.no_candles,
        rebuild_corpora=not a.no_corpora,
        recompute_matrices=not a.no_matrices)
    print("daily_update: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
