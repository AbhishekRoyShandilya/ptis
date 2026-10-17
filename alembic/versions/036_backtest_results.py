"""Persist backtest results so analytics can be served per strategy and per user

WHY: BacktestResult is produced as an in-memory dataclass and discarded. The only
durable performance records are the pre-aggregated rollups in 033/034, so nothing
downstream can answer trade-level questions ("which regime did this strategy lose
in, and at what holding time") without re-running the backtest.

This adds the missing serving layer, in two tables rather than one:

  backtest_runs    one row per execution. Carries the parameters and window the
                   run used, so a result can always be traced to the conditions
                   that produced it. `is_current` marks the run a dashboard
                   should read, which is what stops a re-run silently blending
                   with the results it replaced.

  backtest_trades  one row per simulated trade, keyed to its run.

Three columns make this servable where a file is not: strategy_code (joins to the
strategy dimension), trader_id (row-level security in a multi-user report), and
run_id (isolation between runs).

Revision ID: 036
Revises: 035
Create Date: 2026-09-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "backtest_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("trader_id", sa.String(50), nullable=False),
        sa.Column("strategy_code", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.String(16), nullable=True),
        # What the run covered, so a figure can be traced to its conditions.
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("symbols", JSONB, nullable=True),
        sa.Column("interval", sa.String(8), nullable=True),
        sa.Column("params", JSONB, nullable=True),
        sa.Column("cost_model", sa.String(64), nullable=True),
        # Headline stats, denormalised so a scorecard does not have to
        # aggregate the trade table on every page load.
        sa.Column("trade_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=True),
        sa.Column("expectancy_r", sa.Float, nullable=True),
        sa.Column("total_r", sa.Float, nullable=True),
        sa.Column("max_drawdown_r", sa.Float, nullable=True),
        sa.Column("profit_factor", sa.Float, nullable=True),
        # Only one run per (trader, strategy) is what the dashboard reads.
        sa.Column("is_current", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("status", sa.String(16), nullable=False, server_default="complete"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_backtest_runs_lookup", "backtest_runs",
                    ["trader_id", "strategy_code", "is_current"])

    op.create_table(
        "backtest_trades",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("run_id", UUID(as_uuid=True),
                  sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
                  nullable=False),
        # Denormalised onto the trade so the fact table can be read without
        # a join, which is what a DirectQuery report needs.
        sa.Column("trader_id", sa.String(50), nullable=False),
        sa.Column("strategy_code", sa.String(64), nullable=False),

        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(8), nullable=True),
        sa.Column("entry_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_of_day", sa.String(8), nullable=True),
        sa.Column("session", sa.String(16), nullable=True),

        sa.Column("hold_bars", sa.Integer, nullable=True),
        sa.Column("hold_minutes", sa.Integer, nullable=True),

        # Net is the headline; gross and cost are kept beside it so the
        # friction question can be asked without re-deriving anything.
        sa.Column("r_net", sa.Float, nullable=True),
        sa.Column("r_gross", sa.Float, nullable=True),
        sa.Column("cost_r", sa.Float, nullable=True),
        sa.Column("pnl_pct", sa.Float, nullable=True),
        sa.Column("won", sa.Boolean, nullable=True),

        sa.Column("mae_r", sa.Float, nullable=True),
        sa.Column("mfe_r", sa.Float, nullable=True),
        sa.Column("exit_reason", sa.String(24), nullable=True),

        sa.Column("regime_at_entry", sa.String(32), nullable=True),
        sa.Column("htf_regime", sa.String(32), nullable=True),
        sa.Column("regime_confidence", sa.Float, nullable=True),
        sa.Column("regime_probs", JSONB, nullable=True),
        sa.Column("cost_breakdown", JSONB, nullable=True),

        sa.Column("similarity", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_backtest_trades_run", "backtest_trades", ["run_id"])
    op.create_index("ix_backtest_trades_serving", "backtest_trades",
                    ["trader_id", "strategy_code", "entry_ts"])
    op.create_index("ix_backtest_trades_regime", "backtest_trades",
                    ["strategy_code", "regime_at_entry"])


def downgrade() -> None:
    op.drop_index("ix_backtest_trades_regime", table_name="backtest_trades")
    op.drop_index("ix_backtest_trades_serving", table_name="backtest_trades")
    op.drop_index("ix_backtest_trades_run", table_name="backtest_trades")
    op.drop_table("backtest_trades")
    op.drop_index("ix_backtest_runs_lookup", table_name="backtest_runs")
    op.drop_table("backtest_runs")
