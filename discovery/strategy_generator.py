# discovery/strategy_generator.py
"""
Stage 6 — Strategy DNA Generator

Takes the complete statistical findings from stages 3-5 and generates
a full Strategy DNA YAML file, either via:
  1. LLM mode — sends findings to Claude for intelligent YAML generation
  2. Template mode — fills a structured template with backtest stats

The generated YAML matches the schema in strategy_engine/strategy_dna/schema.py
and can be loaded by StrategyDNALoader.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from discovery.models import (
    BacktestResult,
    CorrelationReport,
    DiscoveryReport,
    GeneratedStrategy,
    PatternFingerprint,
    PatternQuery,
    RegimeBreakdown,
    TimeSlotBreakdown,
)

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """
    Generates Strategy DNA YAML from discovery findings.

    Usage:
        gen = StrategyGenerator(llm_client=my_llm)  # or None for template mode
        strategy = await gen.generate(report)
    """

    def __init__(self, llm_client: Any = None):
        self._llm = llm_client

    async def generate(self, report: DiscoveryReport) -> GeneratedStrategy:
        """Generate a complete Strategy DNA from the discovery report."""
        if self._llm:
            try:
                return await self._generate_with_llm(report)
            except Exception as e:
                logger.warning("LLM generation failed, using template: %s", e)

        return self._generate_from_template(report)

    # ── LLM-powered generation ────────────────────────────────────────

    async def _generate_with_llm(self, report: DiscoveryReport) -> GeneratedStrategy:
        findings = self._format_findings_for_llm(report)

        prompt = f"""You are a quantitative strategy designer for PTIS (Personal Trading Intelligence System).

Based on the following statistical pattern discovery findings, generate a COMPLETE Strategy DNA YAML file.

═══ DISCOVERY FINDINGS ═══
{findings}

═══ REQUIREMENTS ═══
Generate a YAML file matching this EXACT structure (all fields required):

name: <snake_case, unique>
display_name: <human-readable>
short_code: <3-5 chars>
version: 1
category: <momentum|mean_reversion|breakout|trend|scalp|swing>
timeframe_primary: "{report.query.timeframe if report.query else '15m'}"
timeframe_confirmation: "5m"
typical_hold: "<estimated from median hold time>"
description: >
  <2-3 sentence description based on the pattern discovered>

entry_logic:
  narrative: >
    <detailed entry logic based on what the pattern actually does>
  setup_conditions:
    - name: "<condition 1>"
      description: "<what to look for>"
      importance: critical
      what_good_looks_like: "<ideal scenario>"
      what_bad_looks_like: "<bad scenario>"
    <at least 4 conditions based on the fingerprint dimensions>
  a_plus_conditions:
    - "<condition that makes this A+ setup>"
  skip_conditions:
    - "<when to skip this pattern>"

execution:
  entry_trigger: "<specific trigger>"
  entry_timing: "<timing guidance>"
  common_mistakes_i_make:
    - "<common error>"

risk_management:
  stop_loss: "<ATR-based stop description>"
  stop_description: "<detailed stop logic>"
  stop_common_mistakes: ["<mistake>"]
  position_sizing: "<sizing method>"
  risk_per_trade: "<based on win rate and avg R>"
  max_risk: "2%"
  size_adjustments:
    trending_up: "<based on regime performance>"
    ranging: "<based on regime performance>"
  target_primary: "<primary target>"
  target_secondary: "<secondary target>"
  target_method: "<R-multiple based>"

exit_rules:
  profit_taking:
    - rule: "<rule>"
      action: "<action>"
  time_based:
    - rule: "<time rule>"
      action: "<action>"
  common_mistakes: ["<mistake>"]

market_conditions:
  works_best:
    <for each regime with win rate > 55%>
    - regime: "<regime>"
      notes: "<why it works, from the findings>"
      expected_win_rate: "<actual win rate from data>"
      recommendation: "<sizing recommendation>"
  works_poorly:
    <for each regime with win rate < 50%>
    - regime: "<regime>"
      notes: "<why it fails>"
  works_best_sessions:
    <from time-of-day breakdown>
    - session: "<time slot>"
      notes: "<performance>"
  works_poorly_sessions:
    <avoid time slots>
  regime_sizing:
    <map each regime to: full_size, standard_size, half_size, quarter_size, skip>

psychology:
  emotional_traps:
    - trap: "<trap>"
      description: "<detail>"
      solution: "<solution>"
      frequency: sometimes
  ideal_mental_state: ["patient", "systematic", "evidence-based"]
  do_not_trade_when:
    - "<condition to avoid>"

relationships:
  fallback_strategies: []
  conflicts_with: []
  pairs_well_with: []
  regime_alternatives: {{}}

personal_insights:
  strengths: ["<from the data>"]
  weaknesses: ["<from the data>"]
  edge_conditions: ["<when this pattern has a clear edge>"]
  learned_statistics:
    discovery_date: "{datetime.now().strftime('%Y-%m-%d')}"
    total_instances: {report.total_matches}
    overall_win_rate: {report.overall_win_rate:.1%}
    overall_avg_r: {report.overall_avg_r:.2f}
    profit_factor: {report.overall_profit_factor:.2f}

checklist:
  - "<pre-trade check item>"

tags: ["discovered", "pattern-matched", "<category>"]

Return ONLY the YAML content, no markdown fences, no explanations before or after."""

        from agents.prompts import STRATEGY_GENERATOR as _SYS
        response = await self._llm.generate(prompt, system_prompt=_SYS, caller="discovery:strategy_gen")
        yaml_text = response if isinstance(response, str) else str(response)

        # Clean up: remove markdown fences if present
        yaml_text = re.sub(r'^```ya?ml\s*', '', yaml_text, flags=re.MULTILINE)
        yaml_text = re.sub(r'^```\s*$', '', yaml_text, flags=re.MULTILINE)
        yaml_text = yaml_text.strip()

        # Validate it parses
        try:
            yaml_dict = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Generated YAML is invalid: {e}")

        name = yaml_dict.get("name", "discovered_pattern")
        display = yaml_dict.get("display_name", name.replace("_", " ").title())
        code = yaml_dict.get("short_code", name[:4].upper())
        category = yaml_dict.get("category", "momentum")

        return GeneratedStrategy(
            strategy_name=name,
            display_name=display,
            short_code=code,
            category=category,
            yaml_content=yaml_text,
            yaml_dict=yaml_dict,
            confidence=self._compute_confidence(report),
            llm_reasoning=(
                f"Generated from {report.total_matches} historical matches "
                f"with {report.overall_win_rate:.0%} win rate and "
                f"{report.overall_avg_r:.1f}R average across "
                f"{len(report.top_stocks)} stocks."
            ),
        )

    # ── Template-based generation ─────────────────────────────────────

    def _generate_from_template(self, report: DiscoveryReport) -> GeneratedStrategy:
        """Fill a structured template with backtest statistics."""
        q = report.query or PatternQuery()
        corr = report.correlation
        stats = report

        # Derive strategy name from pattern type + symbol
        base_name = q.pattern_type.value if q.pattern_type else "discovered_pattern"
        name = f"{base_name}_{q.timeframe}" if q.timeframe else base_name
        display = name.replace("_", " ").title()
        code = base_name[:3].upper() + q.timeframe.replace("m", "").replace("h", "H")

        # Build regime sizing map
        regime_sizing = {}
        if corr:
            for rb in corr.regime_breakdown:
                if rb.count < 5:
                    regime_sizing[rb.regime] = "skip"
                elif rb.win_rate >= 0.65:
                    regime_sizing[rb.regime] = "full_size"
                elif rb.win_rate >= 0.55:
                    regime_sizing[rb.regime] = "standard_size"
                elif rb.win_rate >= 0.45:
                    regime_sizing[rb.regime] = "half_size"
                else:
                    regime_sizing[rb.regime] = "skip"

        # Build works_best / works_poorly
        works_best = []
        works_poorly = []
        if corr:
            for rb in corr.regime_breakdown:
                entry = {
                    "regime": rb.regime,
                    "notes": rb.why_works or rb.why_fails or "See backtest data",
                    "expected_win_rate": f"{rb.win_rate:.0%}",
                    "recommendation": regime_sizing.get(rb.regime, "standard_size"),
                }
                if rb.win_rate >= 0.55:
                    works_best.append(entry)
                else:
                    works_poorly.append(entry)

        # Time sessions
        best_sessions = []
        poor_sessions = []
        if corr:
            for tb in corr.time_breakdown:
                session = {"session": tb.slot, "notes": f"WR: {tb.win_rate:.0%}, Avg R: {tb.avg_r:.1f}"}
                if tb.win_rate >= 0.55 and tb.count >= 5:
                    best_sessions.append(session)
                elif tb.win_rate < 0.45 and tb.count >= 5:
                    poor_sessions.append(session)

        # Determine risk sizing from win rate
        if stats.overall_win_rate >= 0.65:
            risk_per_trade = "1.0%"
        elif stats.overall_win_rate >= 0.55:
            risk_per_trade = "0.75%"
        else:
            risk_per_trade = "0.5%"

        yaml_dict = {
            "name": name,
            "display_name": display,
            "short_code": code,
            "version": 1,
            "category": self._infer_category(q),
            "timeframe_primary": q.timeframe,
            "timeframe_confirmation": self._infer_confirmation_tf(q.timeframe),
            "typical_hold": f"{stats.median_hold_minutes} minutes" if stats.median_hold_minutes else "15-60 minutes",
            "description": (
                f"Discovered pattern: {q.pattern_type.value.replace('_', ' ')} "
                f"on {q.timeframe} timeframe. Based on {stats.total_matches} "
                f"historical occurrences across FNO stocks with "
                f"{stats.overall_win_rate:.0%} win rate and {stats.overall_avg_r:.1f}R average."
            ),
            "entry_logic": {
                "narrative": (
                    f"Look for {q.pattern_type.value.replace('_', ' ')} pattern "
                    f"on {q.timeframe} chart. The pattern requires volume confirmation "
                    f"(surge above 20-bar average) and momentum aligned with trend. "
                    f"Enter on candle close with strong body ratio."
                ),
                "setup_conditions": [
                    {
                        "name": "Volume surge",
                        "description": "Breakout bar volume > 1.5x 20-bar average",
                        "importance": "critical",
                        "what_good_looks_like": "Clear volume spike visible on chart",
                        "what_bad_looks_like": "Flat or declining volume",
                    },
                    {
                        "name": "Momentum confirmation",
                        "description": "Close-to-close move > 1 ATR",
                        "importance": "critical",
                        "what_good_looks_like": "Strong body candle, little wick against direction",
                        "what_bad_looks_like": "Doji or spinning top on breakout",
                    },
                    {
                        "name": "Consolidation base",
                        "description": "At least 3-5 bars of tight range before breakout",
                        "importance": "important",
                        "what_good_looks_like": "Visible compression in price range",
                        "what_bad_looks_like": "Choppy, no clear consolidation",
                    },
                    {
                        "name": "Trend alignment",
                        "description": "EMA20 slope in the direction of breakout",
                        "importance": "important",
                        "what_good_looks_like": "Rising EMA for longs, falling for shorts",
                        "what_bad_looks_like": "Counter-trend breakout attempt",
                    },
                ],
                "a_plus_conditions": [
                    "Volume > 2x average on breakout bar",
                    f"Time of day in golden window ({corr.best_time_window if corr else '10:00-11:30'})",
                    f"Market regime is {corr.best_regime if corr else 'trending'}",
                    "Multiple timeframe confluence",
                ],
                "skip_conditions": [
                    f"Market regime is {corr.worst_regime if corr else 'ranging'}",
                    "Volume below average on breakout",
                    "Counter-trend to EMA20 slope",
                    "Already extended (RSI > 80 for longs, < 20 for shorts)",
                    "Within 30 minutes of market close",
                ],
            },
            "execution": {
                "entry_trigger": "Candle close beyond consolidation with volume confirmation",
                "entry_timing": f"Enter on close of breakout candle during {corr.best_time_window if corr else '10:00-11:30'} IST",
                "common_mistakes_i_make": [
                    "Entering before candle closes (anticipation)",
                    "Chasing after missing the initial entry",
                    "Ignoring regime context",
                ],
            },
            "risk_management": {
                "stop_loss": "1.5 ATR below entry (longs) or above (shorts)",
                "stop_description": "ATR-based stop placed beyond consolidation structure",
                "stop_common_mistakes": [
                    "Moving stop to breakeven too early",
                    "Widening stop when price pulls back",
                ],
                "position_sizing": "Fixed fractional based on ATR stop distance",
                "risk_per_trade": risk_per_trade,
                "max_risk": "2%",
                "size_adjustments": regime_sizing,
                "target_primary": f"{stats.overall_avg_r:.1f}R (based on backtest average)",
                "target_secondary": "Trail stop after 1.5R using ATR trail",
                "target_method": "R-multiple based with ATR trailing",
            },
            "exit_rules": {
                "profit_taking": [
                    {"rule": "Take 50% at 1.5R", "action": "Move stop to breakeven on remainder"},
                    {"rule": f"Take remaining at {stats.overall_avg_r:.1f}R or trail", "action": "Full exit"},
                ],
                "time_based": [
                    {"rule": f"Exit if not at 1R within {stats.median_hold_minutes or 30} minutes", "action": "Close at market"},
                    {"rule": "Exit by 15:15 IST regardless", "action": "Market close exit"},
                ],
                "common_mistakes": [
                    "Holding past target hoping for more",
                    "Not taking partial profits",
                ],
            },
            "market_conditions": {
                "works_best": works_best,
                "works_poorly": works_poorly,
                "works_best_sessions": best_sessions,
                "works_poorly_sessions": poor_sessions,
                "works_best_additional": [],
                "works_poorly_additional": [],
                "regime_sizing": regime_sizing,
            },
            "psychology": {
                "emotional_traps": [
                    {
                        "trap": "FOMO on missed entry",
                        "description": "Chasing after price has already moved 1R+ from entry zone",
                        "solution": "Accept the miss. Wait for next setup — they occur regularly.",
                        "frequency": "happens_often",
                    },
                    {
                        "trap": "Doubt on valid setup",
                        "description": "Hesitating when all conditions are met",
                        "solution": "Trust the data. This pattern has been validated across {stats.total_matches} instances.",
                        "frequency": "sometimes",
                    },
                ],
                "ideal_mental_state": ["patient", "systematic", "evidence-based", "disciplined"],
                "do_not_trade_when": [
                    "Feeling rushed or pressured",
                    "After 2 consecutive losses (take a break)",
                    f"In {corr.worst_regime if corr else 'ranging'} regime",
                ],
            },
            "relationships": {
                "fallback_strategies": [],
                "conflicts_with": [],
                "pairs_well_with": [],
                "regime_alternatives": {},
            },
            "personal_insights": {
                "strengths": [
                    f"High win rate ({stats.overall_win_rate:.0%}) across {stats.total_matches} instances",
                    f"Positive expectancy ({stats.overall_avg_r:.1f}R average)",
                    f"Clear regime preference ({corr.best_regime if corr else 'trending'})",
                ],
                "weaknesses": [
                    f"Poor in {corr.worst_regime if corr else 'ranging'} regime",
                    "Requires volume confirmation (may miss quiet breakouts)",
                ],
                "edge_conditions": [
                    f"Regime is {corr.best_regime if corr else 'trending'}",
                    f"Time is {corr.best_time_window if corr else '10:00-11:30'} IST",
                    "Volume > 1.5x average",
                ],
                "learned_statistics": {
                    "discovery_date": datetime.now().strftime("%Y-%m-%d"),
                    "total_instances": stats.total_matches,
                    "overall_win_rate": f"{stats.overall_win_rate:.1%}",
                    "overall_avg_r": f"{stats.overall_avg_r:.2f}",
                    "profit_factor": f"{stats.overall_profit_factor:.2f}",
                    "median_hold_minutes": stats.median_hold_minutes,
                    "avg_mae": f"{stats.avg_mae:.2f}R",
                },
            },
            "checklist": [
                "Volume surge confirmed (> 1.5x average)?",
                "Consolidation base visible (3+ bars)?",
                "Trend alignment (EMA20 slope)?",
                f"Regime check ({corr.best_regime if corr else 'trending'})?",
                f"Time window ({corr.best_time_window if corr else '10:00-11:30'} IST)?",
                "Stop placement calculated (1.5 ATR)?",
                "Position size within risk limits?",
            ],
            "tags": [
                "discovered",
                "pattern-matched",
                q.pattern_type.value if q.pattern_type else "custom",
                q.timeframe,
                f"wr-{int(stats.overall_win_rate * 100)}pct",
            ],
        }

        yaml_text = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return GeneratedStrategy(
            strategy_name=name,
            display_name=display,
            short_code=code,
            category=yaml_dict["category"],
            yaml_content=yaml_text,
            yaml_dict=yaml_dict,
            confidence=self._compute_confidence(report),
            llm_reasoning=(
                f"Template-generated from {stats.total_matches} instances. "
                f"Win rate: {stats.overall_win_rate:.0%}, Avg R: {stats.overall_avg_r:.1f}, "
                f"Best regime: {corr.best_regime if corr else 'N/A'}, "
                f"Best time: {corr.best_time_window if corr else 'N/A'}."
            ),
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _format_findings_for_llm(self, report: DiscoveryReport) -> str:
        """Format the report as a structured text block for LLM consumption."""
        q = report.query or PatternQuery()
        corr = report.correlation

        lines = [
            f"PATTERN: {q.pattern_type.value} on {q.symbol} {q.timeframe}",
            f"DESCRIPTION: {q.description}",
            f"",
            f"BACKTEST SUMMARY ({report.total_matches} matches across FNO universe):",
            f"  Win Rate:        {report.overall_win_rate:.1%}",
            f"  Average R:       {report.overall_avg_r:.2f}",
            f"  Profit Factor:   {report.overall_profit_factor:.2f}",
            f"  Median Hold:     {report.median_hold_minutes} minutes",
            f"  Avg MAE:         {report.avg_mae:.2f}R",
            f"",
            f"TOP STOCKS:",
        ]
        for stock in report.top_stocks[:10]:
            lines.append(
                f"  {stock['symbol']:15s} {stock['instances']:3d} instances | "
                f"WR: {stock['win_rate']:.0%} | Avg R: {stock['avg_r']:.1f}"
            )

        if corr:
            lines.append(f"\nREGIME BREAKDOWN:")
            for rb in corr.regime_breakdown:
                lines.append(
                    f"  {rb.regime:20s} WR: {rb.win_rate:.0%} | Avg R: {rb.avg_r:.1f} | "
                    f"N={rb.count}"
                )
                if rb.why_works:
                    lines.append(f"    WHY WORKS: {rb.why_works}")
                if rb.why_fails:
                    lines.append(f"    WHY FAILS: {rb.why_fails}")

            lines.append(f"\nTIME-OF-DAY BREAKDOWN:")
            for tb in corr.time_breakdown:
                lines.append(
                    f"  {tb.slot:15s} WR: {tb.win_rate:.0%} | Avg R: {tb.avg_r:.1f} | "
                    f"N={tb.count} {tb.label}"
                )

            lines.append(f"\nBEST REGIME: {corr.best_regime}")
            lines.append(f"WORST REGIME: {corr.worst_regime}")
            lines.append(f"BEST TIME: {corr.best_time_window}")

        return "\n".join(lines)

    def _compute_confidence(self, report: DiscoveryReport) -> float:
        """Score confidence in the discovered strategy (0–1)."""
        score = 0.0

        # More matches = more confidence
        if report.total_matches >= 500:
            score += 0.25
        elif report.total_matches >= 200:
            score += 0.20
        elif report.total_matches >= 50:
            score += 0.10

        # Higher win rate = more confidence
        if report.overall_win_rate >= 0.65:
            score += 0.25
        elif report.overall_win_rate >= 0.55:
            score += 0.15
        elif report.overall_win_rate >= 0.50:
            score += 0.05

        # Positive avg R
        if report.overall_avg_r >= 2.0:
            score += 0.20
        elif report.overall_avg_r >= 1.5:
            score += 0.15
        elif report.overall_avg_r >= 1.0:
            score += 0.10

        # Profit factor
        if report.overall_profit_factor >= 2.5:
            score += 0.15
        elif report.overall_profit_factor >= 1.5:
            score += 0.10

        # Low MAE
        if report.avg_mae <= 0.5:
            score += 0.15
        elif report.avg_mae <= 1.0:
            score += 0.10

        return min(score, 1.0)

    def _infer_category(self, query: PatternQuery) -> str:
        mapping = {
            "momentum_breakout": "momentum",
            "volume_breakout": "breakout",
            "pullback_entry": "trend",
            "reversal": "mean_reversion",
            "gap_play": "momentum",
            "range_breakout": "breakout",
            "vwap_reclaim": "mean_reversion",
            "squeeze_breakout": "breakout",
            "trend_continuation": "trend",
            "custom": "momentum",
        }
        return mapping.get(query.pattern_type.value, "momentum")

    def _infer_confirmation_tf(self, primary_tf: str) -> str:
        mapping = {
            "1m": "1m",
            "3m": "1m",
            "5m": "1m",
            "15m": "5m",
            "30m": "15m",
            "1h": "15m",
            "4h": "1h",
            "1d": "4h",
        }
        return mapping.get(primary_tf, "5m")
