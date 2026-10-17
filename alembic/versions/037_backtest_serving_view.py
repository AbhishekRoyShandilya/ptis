"""Serving view for backtest analytics

WHY: a report should not encode which run is current, or how a run header joins
to its trades. Putting that in a view means the dashboard's query is
`SELECT * FROM v_backtest_serving WHERE trader_id = ...` and nothing more, and
the definition of "current" can change without touching the report.

This is the contract between the engine and the BI layer. Power BI points at
this view in DirectQuery, or imports it on a refresh; either way it never sees
a superseded run.

Revision ID: 037
Revises: 036
Create Date: 2026-09-01
"""
from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


SERVING_VIEW = """
CREATE OR REPLACE VIEW v_backtest_serving AS
SELECT
    t.id                    AS trade_id,
    t.run_id,
    t.trader_id,
    t.strategy_code,
    r.strategy_version,
    r.cost_model,
    r.window_start,
    r.window_end,

    t.symbol,
    t.direction,
    t.entry_ts,
    t.exit_ts,
    t.entry_ts::date        AS trade_date,
    t.time_of_day,
    t.session,
    t.hold_bars,
    t.hold_minutes,

    t.r_net,
    t.r_gross,
    t.cost_r,
    t.pnl_pct,
    t.won,
    t.mae_r,
    t.mfe_r,
    t.exit_reason,

    t.regime_at_entry,
    t.htf_regime,
    t.regime_confidence,
    t.similarity,

    -- Holding-time buckets, so "how long should this be held" is answerable
    -- without every report re-inventing the same CASE expression.
    CASE
        WHEN t.hold_minutes IS NULL      THEN 'Unknown'
        WHEN t.hold_minutes <= 15        THEN '0-15 min'
        WHEN t.hold_minutes <= 30        THEN '16-30 min'
        WHEN t.hold_minutes <= 60        THEN '31-60 min'
        WHEN t.hold_minutes <= 120       THEN '1-2 hours'
        ELSE 'Over 2 hours'
    END                     AS hold_bucket,
    CASE
        WHEN t.hold_minutes IS NULL THEN 99
        WHEN t.hold_minutes <= 15   THEN 1
        WHEN t.hold_minutes <= 30   THEN 2
        WHEN t.hold_minutes <= 60   THEN 3
        WHEN t.hold_minutes <= 120  THEN 4
        ELSE 5
    END                     AS hold_bucket_order
FROM backtest_trades t
JOIN backtest_runs   r ON t.run_id = r.id
WHERE r.is_current = TRUE
  AND r.status = 'complete';
"""

# Per-strategy scorecard. Reads the run header rather than aggregating the
# trade table, which is why the header carries denormalised stats at all.
SCORECARD_VIEW = """
CREATE OR REPLACE VIEW v_backtest_scorecard AS
SELECT
    r.trader_id,
    r.strategy_code,
    r.strategy_version,
    r.trade_count,
    r.win_rate,
    r.expectancy_r,
    r.total_r,
    r.max_drawdown_r,
    r.profit_factor,
    r.window_start,
    r.window_end,
    r.cost_model,
    r.created_at            AS last_backtest_at,
    -- A thin backtest should not be presented beside a deep one without
    -- the difference being visible.
    (r.trade_count >= 100)  AS sample_adequate
FROM backtest_runs r
WHERE r.is_current = TRUE
  AND r.status = 'complete';
"""

# Which regime, session and holding time a strategy actually works in — the
# question the product exists to answer, defined once.
CONTEXT_VIEW = """
CREATE OR REPLACE VIEW v_backtest_context AS
SELECT
    trader_id,
    strategy_code,
    regime_at_entry,
    session,
    hold_bucket,
    MIN(hold_bucket_order)                            AS hold_bucket_order,
    COUNT(*)                                          AS trades,
    AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)          AS win_rate,
    AVG(r_net)                                        AS expectancy_r,
    SUM(r_net)                                        AS total_r,
    AVG(mae_r)                                        AS avg_mae_r,
    AVG(mfe_r)                                        AS avg_mfe_r,
    (COUNT(*) >= 30)                                  AS sample_adequate
FROM v_backtest_serving
GROUP BY trader_id, strategy_code, regime_at_entry, session, hold_bucket;
"""


def upgrade() -> None:
    op.execute(SERVING_VIEW)
    op.execute(SCORECARD_VIEW)
    op.execute(CONTEXT_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_backtest_context")
    op.execute("DROP VIEW IF EXISTS v_backtest_scorecard")
    op.execute("DROP VIEW IF EXISTS v_backtest_serving")
