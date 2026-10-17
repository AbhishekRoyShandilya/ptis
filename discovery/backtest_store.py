"""
Persist backtest results so they can be served.

`BacktestResult` is produced in memory and, until now, discarded. That is fine
for a research script reading its own output, and useless for a product: a
dashboard cannot filter a Python object, secure it per user, or refresh from it.

Results are written into two tables. A run header records what the backtest
covered — window, symbols, parameters, cost model — and the trades hang off it.
Two consequences worth knowing:

  * A result can always be traced to the conditions that produced it. Several of
    the measurement errors in this project's history came from a figure that had
    been separated from its assumptions.
  * Re-running a strategy does not blend with what it replaced. The new run is
    marked current and the previous one is retained, so the dashboard moves and
    the audit trail survives.

Usage:

    from discovery.backtest_store import persist_backtest

    run_id = persist_backtest(
        results,
        trader_id="abhishek",
        strategy_code="MER",
        window=(start, end),
        symbols=["RELIANCE", "INFY"],
        interval="5m",
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: Intraday session boundaries (IST). These match the analytics layer, so a
#: session label means the same thing in the warehouse as it does in a report.
_SESSIONS = [
    ("OPEN", 9 * 60 + 15, 9 * 60 + 45),
    ("MORNING", 9 * 60 + 45, 11 * 60 + 30),
    ("MIDDAY", 11 * 60 + 30, 13 * 60 + 30),
    ("AFTERNOON", 13 * 60 + 30, 15 * 60),
    ("CLOSE", 15 * 60, 15 * 60 + 30),
]


def _session_of(time_of_day: Optional[str]) -> Optional[str]:
    """Map an 'HH:MM' string to a session bucket."""
    if not time_of_day or ":" not in str(time_of_day):
        return None
    try:
        hour, minute = str(time_of_day).split(":")[:2]
        minutes = int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return None
    for name, start, end in _SESSIONS:
        if start <= minutes < end:
            return name
    return "UNKNOWN"


def _get(obj: Any, *names: str, default=None):
    """Read the first attribute that exists, tolerating schema drift.

    The backtester's result shape has changed before and will again. A field
    that has been renamed should leave a null in one column, not abort the run.
    """
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _jsonable(value):
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return value


def _json_text(value) -> Optional[str]:
    return None if value is None else json.dumps(value)


def summarise(rows: Sequence[dict]) -> dict:
    """Headline stats for the run header.

    Denormalised deliberately: a scorecard should not have to aggregate the
    whole trade table to draw a single row.
    """
    r_values = [r["r_net"] for r in rows if r.get("r_net") is not None]
    if not r_values:
        return {"trade_count": len(rows)}

    wins = [v for v in r_values if v > 0]
    losses = [v for v in r_values if v < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    # Worst peak-to-trough decline of the cumulative R curve.
    cumulative = peak = max_drawdown = 0.0
    for value in r_values:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)

    return {
        "trade_count": len(rows),
        "win_rate": len(wins) / len(r_values),
        "expectancy_r": sum(r_values) / len(r_values),
        "total_r": sum(r_values),
        "max_drawdown_r": max_drawdown,
        "profit_factor": (gross_profit / gross_loss) if gross_loss else None,
    }


def to_row(result: Any, trader_id: str, strategy_code: str, run_id) -> dict:
    """Flatten one BacktestResult into a table row."""
    match = getattr(result, "match", None)

    symbol = _get(match, "symbol", default="") if match else ""
    entry_ts = _get(match, "date", "entry_ts") if match else None
    time_of_day = (_get(match, "time_of_day") if match else None) \
        or _get(result, "time_of_day")

    return {
        "run_id": run_id,
        "trader_id": trader_id,
        "strategy_code": strategy_code,
        "symbol": str(symbol)[:32],
        "direction": _get(result, "direction"),
        "entry_ts": entry_ts,
        "exit_ts": _get(result, "exit_ts"),
        "time_of_day": str(time_of_day)[:8] if time_of_day else None,
        "session": _session_of(time_of_day),
        "hold_bars": _get(result, "hold_bars"),
        "hold_minutes": _get(result, "hold_minutes"),
        # r_multiple is already net of costs; the gross figure is kept beside
        # it so friction can be measured without re-deriving anything.
        "r_net": _get(result, "r_multiple"),
        "r_gross": _get(result, "r_multiple_gross"),
        "cost_r": _get(result, "cost_r"),
        "pnl_pct": _get(result, "pnl_pct"),
        "won": bool(_get(result, "won", default=False)),
        "mae_r": _get(result, "max_adverse_excursion"),
        "mfe_r": _get(result, "max_favorable_excursion"),
        "exit_reason": _get(result, "exit_reason") or None,
        "regime_at_entry": _get(result, "regime_at_entry") or None,
        "htf_regime": _get(result, "htf_regime", "htf_trend_1h") or None,
        "regime_confidence": _get(result, "regime_confidence"),
        "regime_probs": _jsonable(_get(result, "regime_probs_at_entry")),
        "cost_breakdown": _jsonable(_get(result, "cost_breakdown")),
        "similarity": _get(match, "similarity") if match else None,
    }


def build_rows(results: Iterable[Any], trader_id: str,
               strategy_code: str, run_id) -> list[dict]:
    """Convert results to rows, skipping only what cannot be read at all."""
    rows: list[dict] = []
    skipped = 0
    for result in results:
        try:
            rows.append(to_row(result, trader_id, strategy_code, run_id))
        except Exception as exc:
            skipped += 1
            logger.debug("unreadable backtest result skipped: %s", exc)
    if skipped:
        logger.warning("%s backtest result(s) could not be flattened", skipped)
    return rows


_INSERT_RUN = """
INSERT INTO backtest_runs (
    id, trader_id, strategy_code, strategy_version,
    window_start, window_end, symbols, interval, params, cost_model,
    trade_count, win_rate, expectancy_r, total_r, max_drawdown_r,
    profit_factor, is_current, status
) VALUES (
    :id, :trader, :code, :version,
    :wstart, :wend, CAST(:symbols AS JSONB), :interval,
    CAST(:params AS JSONB), :cost_model,
    :trade_count, :win_rate, :expectancy_r, :total_r, :max_drawdown_r,
    :profit_factor, TRUE, 'complete'
)
"""

_INSERT_TRADES = """
INSERT INTO backtest_trades (
    run_id, trader_id, strategy_code, symbol, direction,
    entry_ts, exit_ts, time_of_day, session, hold_bars, hold_minutes,
    r_net, r_gross, cost_r, pnl_pct, won, mae_r, mfe_r, exit_reason,
    regime_at_entry, htf_regime, regime_confidence,
    regime_probs, cost_breakdown, similarity
) VALUES (
    :run_id, :trader_id, :strategy_code, :symbol, :direction,
    :entry_ts, :exit_ts, :time_of_day, :session, :hold_bars, :hold_minutes,
    :r_net, :r_gross, :cost_r, :pnl_pct, :won, :mae_r, :mfe_r, :exit_reason,
    :regime_at_entry, :htf_regime, :regime_confidence,
    CAST(:regime_probs AS JSONB), CAST(:cost_breakdown AS JSONB), :similarity
)
"""


def persist_backtest(
    results: Iterable[Any],
    trader_id: str,
    strategy_code: str,
    *,
    strategy_version: str | None = None,
    window: tuple[datetime, datetime] | None = None,
    symbols: Sequence[str] | None = None,
    interval: str | None = None,
    params: dict | None = None,
    cost_model: str | None = None,
    db=None,
) -> str:
    """Write a completed backtest and return its run id.

    Any previous run for the same (trader, strategy) is marked not-current in
    the same transaction, so a dashboard reading `is_current` never sees two
    runs at once and never sees none.
    """
    from sqlalchemy import text

    from core.database import SessionLocal

    run_id = uuid.uuid4()
    rows = build_rows(results, trader_id, strategy_code, run_id)
    stats = summarise(rows)

    session = db or SessionLocal()
    owns_session = db is None
    try:
        session.execute(
            text(
                "UPDATE backtest_runs SET is_current = FALSE "
                "WHERE trader_id = :trader AND strategy_code = :code "
                "AND is_current = TRUE"
            ),
            {"trader": trader_id, "code": strategy_code},
        )

        session.execute(text(_INSERT_RUN), {
            "id": run_id,
            "trader": trader_id,
            "code": strategy_code,
            "version": strategy_version,
            "wstart": window[0] if window else None,
            "wend": window[1] if window else None,
            "symbols": _json_text(list(symbols) if symbols else None),
            "interval": interval,
            "params": _json_text(params),
            "cost_model": cost_model,
            "trade_count": stats.get("trade_count", 0),
            "win_rate": stats.get("win_rate"),
            "expectancy_r": stats.get("expectancy_r"),
            "total_r": stats.get("total_r"),
            "max_drawdown_r": stats.get("max_drawdown_r"),
            "profit_factor": stats.get("profit_factor"),
        })

        if rows:
            session.execute(text(_INSERT_TRADES), [_prepare(r) for r in rows])

        session.commit()
        logger.info("backtest run %s stored: %s trades for %s/%s",
                    run_id, len(rows), trader_id, strategy_code)
        return str(run_id)
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def _prepare(row: dict) -> dict:
    out = dict(row)
    out["regime_probs"] = _json_text(out.get("regime_probs"))
    out["cost_breakdown"] = _json_text(out.get("cost_breakdown"))
    return out


def current_run(trader_id: str, strategy_code: str, db=None):
    """The run a dashboard should be reading for this strategy."""
    from sqlalchemy import text

    from core.database import SessionLocal

    session = db or SessionLocal()
    try:
        return session.execute(
            text(
                "SELECT * FROM backtest_runs "
                "WHERE trader_id = :trader AND strategy_code = :code "
                "AND is_current = TRUE ORDER BY created_at DESC LIMIT 1"
            ),
            {"trader": trader_id, "code": strategy_code},
        ).mappings().first()
    finally:
        if db is None:
            session.close()


def list_runs(trader_id: str, strategy_code: str | None = None, db=None):
    """Run history, newest first — the audit trail behind any current figure."""
    from sqlalchemy import text

    from core.database import SessionLocal

    clause = "WHERE trader_id = :trader"
    params: dict[str, Any] = {"trader": trader_id}
    if strategy_code:
        clause += " AND strategy_code = :code"
        params["code"] = strategy_code

    session = db or SessionLocal()
    try:
        return session.execute(
            text(
                "SELECT id, strategy_code, strategy_version, trade_count, "
                "win_rate, expectancy_r, total_r, max_drawdown_r, "
                "profit_factor, is_current, created_at "
                f"FROM backtest_runs {clause} ORDER BY created_at DESC"
            ),
            params,
        ).mappings().all()
    finally:
        if db is None:
            session.close()
