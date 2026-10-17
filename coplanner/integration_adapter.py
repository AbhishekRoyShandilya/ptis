# coplanner/integration_adapter.py
"""
ADAPTER LAYER — Bridges existing PTIS modules to Co-Planner.

PURPOSE:
  - Convert watchlist/scanner output → SetupData
  - Convert trade journal entries → TradeRecord
  - Build structured LLM context from engine output
  - Feed ENGINE results into LLM prompt so LLM explains, not computes

RULE: Engine computes → LLM explains. NEVER the reverse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from coplanner.models import SetupData, TradeRecord

logger = logging.getLogger(__name__)


class WatchlistAdapter:
    """
    Converts watchlist_analyzer / setup_scanner output into SetupData
    objects that ConvictionEngine can score.
    """

    @staticmethod
    def from_watchlist_result(
        watchlist_output: dict,
        regime_builder: Any = None,
    ) -> SetupData:
        """
        Convert a SINGLE watchlist/scanner result to SetupData.

        Handles multiple field naming conventions so this works
        regardless of upstream format.
        """
        current_regime = "unknown"
        if regime_builder:
            try:
                current_regime = str(regime_builder.get_current_regime())
            except Exception:
                pass

        return SetupData(
            symbol=watchlist_output.get(
                "symbol",
                watchlist_output.get("ticker",
                watchlist_output.get("stock", "")),
            ),
            strategy_name=watchlist_output.get(
                "strategy",
                watchlist_output.get("strategy_name",
                watchlist_output.get("signal_type", "")),
            ),
            timeframe=watchlist_output.get(
                "timeframe",
                watchlist_output.get("tf", "1h"),
            ),
            pattern_data=watchlist_output.get(
                "pattern",
                watchlist_output.get("pattern_data",
                watchlist_output.get("features", {})),
            ),
            cluster_id=watchlist_output.get(
                "cluster_id",
                watchlist_output.get("cluster", None),
            ),
            entry_price=float(watchlist_output.get(
                "entry",
                watchlist_output.get("entry_price", 0),
            )),
            stop_loss=float(watchlist_output.get(
                "stop_loss",
                watchlist_output.get("sl",
                watchlist_output.get("stoploss", 0)),
            )),
            target_price=float(watchlist_output.get(
                "target",
                watchlist_output.get("target_price",
                watchlist_output.get("tp", 0)),
            )),
            current_volume=float(watchlist_output.get(
                "volume",
                watchlist_output.get("current_volume",
                watchlist_output.get("vol", 0)),
            )),
            avg_volume=float(watchlist_output.get(
                "avg_volume",
                watchlist_output.get("average_volume", 0),
            )),
            current_regime=current_regime,
            pattern_score=float(watchlist_output.get(
                "score",
                watchlist_output.get("pattern_score",
                watchlist_output.get("signal_strength", 0)),
            )),
            watchlist_meta=watchlist_output,
        )

    @staticmethod
    def from_watchlist_batch(
        watchlist_results: list,
        regime_builder: Any = None,
    ) -> List[SetupData]:
        """Convert a BATCH of watchlist results to SetupData list."""
        setups: List[SetupData] = []
        for result in watchlist_results:
            try:
                setup = WatchlistAdapter.from_watchlist_result(result, regime_builder)
                if setup.symbol:
                    setups.append(setup)
            except Exception as e:
                logger.warning(
                    "Failed to convert watchlist result: %s. Skipping: %s",
                    e, result.get("symbol", "unknown"),
                )
        return setups


class TradeJournalAdapter:
    """
    Converts trade journal entries into TradeRecord objects
    that StrategyRegimeMatrix can consume.
    """

    @staticmethod
    def from_journal_entry(journal_entry: dict) -> TradeRecord:
        """Convert a single trade journal entry to TradeRecord."""
        # Normalize outcome
        raw_outcome = str(journal_entry.get(
            "outcome",
            journal_entry.get("result",
            journal_entry.get("status", "")),
        )).lower().strip()

        if raw_outcome in ("win", "profit", "winner", "w", "tp_hit", "target_hit"):
            outcome = "win"
        elif raw_outcome in ("loss", "lose", "loser", "l", "sl_hit", "stop_hit"):
            outcome = "loss"
        else:
            outcome = "unknown"

        # Compute actual RR if not provided
        actual_rr = journal_entry.get("rr", journal_entry.get("actual_rr", 0))
        if not actual_rr:
            entry = float(journal_entry.get("entry_price", journal_entry.get("entry", 0)))
            exit_p = float(journal_entry.get("exit_price", journal_entry.get("exit", 0)))
            sl = float(journal_entry.get("stop_loss", journal_entry.get("sl", 0)))
            if entry and sl and exit_p:
                risk = abs(entry - sl)
                if risk > 0:
                    actual_rr = round(abs(exit_p - entry) / risk, 2)
                    if (exit_p < entry and entry < journal_entry.get("target", entry + 1)):
                        actual_rr = -actual_rr

        return TradeRecord(
            trade_id=str(journal_entry.get("id", journal_entry.get("trade_id", ""))),
            symbol=journal_entry.get("symbol", journal_entry.get("ticker", "")),
            strategy_name=journal_entry.get("strategy", journal_entry.get("strategy_name", "")),
            timeframe=journal_entry.get("timeframe", journal_entry.get("tf", "")),
            regime_at_entry=journal_entry.get("regime", journal_entry.get("regime_at_entry", "unknown")),
            entry_price=float(journal_entry.get("entry_price", journal_entry.get("entry", 0))),
            exit_price=float(journal_entry.get("exit_price", journal_entry.get("exit", 0))),
            stop_loss=float(journal_entry.get("stop_loss", journal_entry.get("sl", 0))),
            target_price=float(journal_entry.get("target_price", journal_entry.get("target", 0))),
            entry_time=TradeJournalAdapter._safe_datetime(journal_entry.get("entry_time", "")),
            exit_time=TradeJournalAdapter._safe_datetime(journal_entry.get("exit_time", "")),
            outcome=outcome,
            actual_rr=float(actual_rr or 0),
            volume_at_entry=float(journal_entry.get("volume", journal_entry.get("volume_at_entry", 0))),
            cluster_id=journal_entry.get("cluster_id", journal_entry.get("cluster", None)),
        )

    @staticmethod
    def from_journal_batch(entries: list) -> List[TradeRecord]:
        """Convert batch of journal entries."""
        records: List[TradeRecord] = []
        for entry in entries:
            try:
                records.append(TradeJournalAdapter.from_journal_entry(entry))
            except Exception as e:
                logger.warning("Failed to convert journal entry: %s", e)
        return records

    @staticmethod
    def _safe_datetime(val: Any) -> datetime:
        if isinstance(val, datetime):
            return val
        if isinstance(val, str) and val:
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%d-%m-%Y %H:%M:%S",
            ]:
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
        return datetime.now(timezone.utc)


class CoPlanEngineAdapter:
    """
    Adapts CoPlannerEngine output for the existing LLM-based coplan_engine.py.

    Feeds ENGINE results into the LLM prompt so LLM EXPLAINS instead of COMPUTING.
    """

    def __init__(self, engine):
        """
        Args:
            engine: CoPlannerEngine instance (or dict with sub-modules)
        """
        if isinstance(engine, dict):
            self._conviction = engine.get("conviction")
            self._regime_matrix = engine.get("regime_matrix")
            self._rr_enforcer = engine.get("rr_enforcer")
            self._trust_layer = engine.get("trust_layer")
            self._classifier = engine.get("quality_classifier")
        else:
            self._conviction = getattr(engine, "_conviction", None)
            self._regime_matrix = getattr(engine, "_regime_matrix", None)
            self._rr_enforcer = getattr(engine, "_rr_enforcer", None)
            self._trust_layer = getattr(engine, "_trust_layer", None)
            self._classifier = getattr(engine, "_classifier", None)

    def get_engine_analysis_for_llm(
        self,
        watchlist_results: list,
        regime_builder: Any = None,
    ) -> Dict[str, Any]:
        """
        Takes watchlist results, runs through engine,
        returns structured data for LLM to explain.
        """
        setups = WatchlistAdapter.from_watchlist_batch(watchlist_results, regime_builder)

        if not setups:
            return {
                "engine_analysis": None,
                "llm_context": "No valid setups found in watchlist scan.",
                "recommendation": {"action": "NO_SETUPS", "message": "Nothing to analyze."},
                "warnings": [],
            }

        # Score each setup
        scored_setups: List[Dict[str, Any]] = []
        regime_warnings: List[Dict[str, Any]] = []
        all_warnings: List[Dict[str, Any]] = []

        for setup in setups:
            if self._conviction:
                conv = self._conviction.compute(setup)
                scored = {
                    "symbol": setup.symbol,
                    "strategy": setup.strategy_name,
                    "conviction_score": conv.conviction_score,
                    "conviction_level": conv.conviction_level.value,
                    "components": conv.components,
                    "warnings": conv.warnings,
                    "sizing_modifier": conv.sizing_modifier,
                    "verdict": conv.verdict,
                }
                scored_setups.append(scored)

                for w in conv.warnings:
                    all_warnings.append({"symbol": setup.symbol, **w})

            if self._regime_matrix:
                advice = self._regime_matrix.should_avoid(setup.strategy_name, setup.current_regime)
                if advice.avoid:
                    rw = {
                        "symbol": setup.symbol,
                        "strategy": setup.strategy_name,
                        "regime_warning": advice.reason,
                        "alternatives": [
                            {"strategy": a["strategy"], "win_rate": a["win_rate"]}
                            for a in advice.alternatives
                        ],
                    }
                    regime_warnings.append(rw)
                    all_warnings.append(rw)

        scored_setups.sort(key=lambda x: x.get("conviction_score", 0), reverse=True)

        # Build recommendation
        recommendation: Dict[str, Any] = {"action": "NO_SETUPS", "message": "No actionable setups."}
        if scored_setups:
            top = scored_setups[0]
            if top["conviction_score"] >= 0.70:
                recommendation = {
                    "action": "TAKE",
                    "primary": top["symbol"],
                    "message": f"{top['symbol']} — conviction {top['conviction_score']:.2f}",
                }
            elif top["conviction_score"] >= 0.55:
                recommendation = {
                    "action": "CAUTIOUS",
                    "primary": top["symbol"],
                    "message": f"{top['symbol']} — moderate conviction, reduced size.",
                }
            else:
                recommendation = {
                    "action": "SKIP_ALL",
                    "message": "No high-conviction setup today. Best to wait.",
                }

        llm_context = self._build_llm_context(scored_setups, regime_warnings, recommendation)

        return {
            "engine_analysis": {"scored_setups": scored_setups, "recommendation": recommendation},
            "llm_context": llm_context,
            "recommendation": recommendation,
            "warnings": all_warnings,
        }

    def _build_llm_context(
        self,
        scored_setups: List[Dict[str, Any]],
        regime_warnings: List[Dict[str, Any]],
        recommendation: Dict[str, Any],
    ) -> str:
        """Build structured context string for LLM."""
        lines = [
            "=" * 50,
            "CO-PLANNER ENGINE ANALYSIS",
            "=" * 50, "",
            f"Total setups scanned: {len(scored_setups)}",
            f"Actionable: {sum(1 for s in scored_setups if s.get('conviction_score', 0) >= 0.55)}",
            "",
        ]

        if scored_setups:
            lines.append("RANKED SETUPS:")
            lines.append("-" * 40)
            for i, scored in enumerate(scored_setups, 1):
                lines.append(f"Rank {i}: {scored['symbol']} ({scored['strategy']})")
                lines.append(f"  Conviction: {scored['conviction_score']:.2f} ({scored['conviction_level']})")
                components = scored.get("components", {})
                lines.append("  Components:")
                for k, v in components.items():
                    lines.append(f"    {k}: {v:.2f}")
                lines.append(f"  Sizing: {scored.get('sizing_modifier', 0):.0%}")
                lines.append(f"  Verdict: {scored.get('verdict', '')}")
                if scored.get("warnings"):
                    lines.append("  Warnings:")
                    for w in scored["warnings"]:
                        lines.append(f"    {w.get('message', w)}")
                lines.append("")

        if regime_warnings:
            lines.append("REGIME WARNINGS:")
            lines.append("-" * 40)
            for rw in regime_warnings:
                lines.append(f"  {rw['symbol']} ({rw['strategy']}): {rw['regime_warning']}")
                if rw.get("alternatives"):
                    for alt in rw["alternatives"]:
                        lines.append(f"    -> {alt['strategy']} (WR: {alt['win_rate']:.0%})")
            lines.append("")

        lines.append("RECOMMENDATION:")
        lines.append("-" * 40)
        lines.append(f"  Action: {recommendation.get('action', 'UNKNOWN')}")
        lines.append(f"  {recommendation.get('message', '')}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("NOTE: These are ENGINE-computed values.")
        lines.append("Explain them to the user clearly. Do NOT recompute or override the scores.")
        lines.append("=" * 50)

        return "\n".join(lines)

    def get_data_quality_for_llm(self) -> Optional[str]:
        """Get data quality context for LLM to communicate to user."""
        if not self._trust_layer:
            return None
        try:
            report = self._trust_layer.get_data_quality_report()
            return (
                f"Data Quality: Grade {report.get('execution_grade', 'N/A')} "
                f"({report.get('clean_ratio', 0):.0%} clean, "
                f"{report.get('emotional_ratio', 0):.0%} emotional). "
                f"{report.get('message', '')}"
            )
        except Exception:
            return None

    def get_modified_llm_prompt(
        self, original_prompt: str, engine_context: str,
    ) -> str:
        """Modify existing LLM prompt to include engine analysis."""
        injection = f"""
IMPORTANT -- ENGINE ANALYSIS BELOW
The following analysis was computed by the Co-Planner Engine.
Your role is to EXPLAIN these findings to the user in a clear,
actionable way. Do NOT override or recompute any scores.

{engine_context}

Based on the above engine analysis, provide your explanation.
Focus on:
1. Which setup is best and WHY (use the component scores)
2. What to avoid and WHY (use the warnings)
3. Regime-specific advice (use the regime data)
4. Clear action: take / skip / wait
"""
        return f"{original_prompt}\n\n{injection}"
