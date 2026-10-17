"""
Last-Week Mistake Summarizer
=============================
Queries BehavioralEvent and OverrideEvent over the last 7 days and
returns a concise, ranked list of the trader's top recurring mistake patterns.

Output is plain text, injected directly into the co-planner system prompt
so the LLM can say:
  "Last week you skipped your stop 3 times — that cost you 4.2R total.
   Don't do it again today."

Design:
- Pure SQL — no LLM, no inference, no hallucination
- Runs synchronously (uses sync DB session, same as CoPlannerContextAssembler)
- Returns top 3 patterns ranked by frequency × severity
- Fails silently — returns empty string if DB is unavailable
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# How many days to look back
LOOKBACK_DAYS = 7

# Human-readable labels for enum values
_OVERRIDE_LABELS: dict[str, str] = {
    "STOP_LOSS_MOVED":     "moved stop loss wider",
    "STOP_LOSS_IGNORED":   "ignored stop loss",
    "POSITION_OVERSIZED":  "oversized position",
    "ENTRY_EARLY":         "entered before confirmation",
    "ENTRY_LATE":          "chased entry after signal",
    "TARGET_MOVED":        "moved target prematurely",
    "RULE_DEVIATION":      "deviated from rules",
    "REVENGE_TRADE":       "revenge traded after loss",
}

_BEHAVIORAL_LABELS: dict[str, str] = {
    "REVENGE_TRADE":       "revenge traded after a loss",
    "FOMO_ENTRY":          "entered due to FOMO",
    "OVERTRADING":         "overtraded (too many trades)",
    "TILT":                "traded while in tilt",
    "PANIC_EXIT":          "panic-exited before stop",
    "EARLY_EXIT":          "exited too early, left gains behind",
    "HESITATION":          "hesitated on valid signals",
    "LOSS_AVERSION":       "kept losers too long",
    "CONFIRMATION_BIAS":   "ignored conflicting signals",
}


class LastWeekMistakeSummarizer:
    """
    Aggregates last 7 days of mistakes from DB and ranks them by impact.
    Runs synchronously — pass a sync SQLAlchemy session.
    """

    def __init__(self, db_session: Any) -> None:
        self._db = db_session

    def summarize(self, trader_id: str) -> str:
        """
        Returns a multi-line string listing the top recurring mistakes,
        ready to be injected into the co-planner system prompt.

        Returns "" if no data or DB error.
        """
        since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

        override_patterns = self._query_overrides(trader_id, since)
        behavioral_patterns = self._query_behavioral(trader_id, since)
        day_patterns = self._query_overtrade_days(trader_id, since)

        all_patterns: list[dict] = []
        all_patterns.extend(override_patterns)
        all_patterns.extend(behavioral_patterns)
        all_patterns.extend(day_patterns)

        if not all_patterns:
            return ""

        # Rank by score (frequency × avg_severity)
        all_patterns.sort(key=lambda p: p["score"], reverse=True)
        top = all_patterns[:3]

        lines = [f"## Last {LOOKBACK_DAYS}-Day Mistake Patterns"]
        for i, p in enumerate(top, 1):
            lines.append(f"  {i}. {p['description']}")

        return "\n".join(lines)

    # ── Override patterns ─────────────────────────────────────────

    def _query_overrides(self, trader_id: str, since: datetime) -> list[dict]:
        """Group override_events by type, count, and sum their severity."""
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT
                        oe.override_type,
                        COUNT(*) AS cnt,
                        AVG(oe.severity_score) AS avg_sev,
                        SUM(CASE WHEN t.pnl IS NOT NULL THEN t.pnl ELSE 0 END) AS total_pnl_impact
                    FROM override_events oe
                    JOIN trades t ON t.id = oe.trade_id
                    WHERE oe.detected_at >= :since
                      AND (t.trader_id = :tid OR t.trader_id IS NULL)
                    GROUP BY oe.override_type
                    ORDER BY COUNT(*) DESC
                    LIMIT 5
                """),
                {"since": since, "tid": trader_id},
            )
            rows = result.fetchall()
            patterns = []
            for row in rows:
                otype, cnt, avg_sev, pnl_impact = row
                otype_str = str(otype.value if hasattr(otype, "value") else otype)
                label = _OVERRIDE_LABELS.get(otype_str, otype_str.lower().replace("_", " "))
                pnl_str = ""
                if pnl_impact and abs(pnl_impact) > 0:
                    pnl_str = f", cost ₹{abs(pnl_impact):,.0f}"
                desc = (
                    f"You {label} {int(cnt)}x in the last week"
                    f" (avg severity {round((avg_sev or 0) * 10, 1)}/10{pnl_str})"
                )
                patterns.append({
                    "description": desc,
                    "score": int(cnt) * float(avg_sev or 0),
                })
            return patterns
        except Exception as e:
            logger.debug("Override pattern query failed: %s", e)
            return []

    # ── Behavioral patterns ───────────────────────────────────────

    def _query_behavioral(self, trader_id: str, since: datetime) -> list[dict]:
        """Group behavioral_events by type, negative ones only."""
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT
                        be.event_type,
                        COUNT(*) AS cnt,
                        AVG(be.severity_score) AS avg_sev,
                        AVG(CASE WHEN be.outcome_r IS NOT NULL THEN be.outcome_r ELSE 0 END) AS avg_outcome_r
                    FROM behavioral_events be
                    WHERE be.detected_at >= :since
                      AND be.is_positive = false
                    GROUP BY be.event_type
                    ORDER BY COUNT(*) DESC
                    LIMIT 5
                """),
                {"since": since},
            )
            rows = result.fetchall()
            patterns = []
            for row in rows:
                etype, cnt, avg_sev, avg_r = row
                etype_str = str(etype.value if hasattr(etype, "value") else etype)
                label = _BEHAVIORAL_LABELS.get(etype_str, etype_str.lower().replace("_", " "))
                r_str = ""
                if avg_r and abs(avg_r) > 0.05:
                    r_str = f", avg outcome {avg_r:+.1f}R"
                desc = (
                    f"You {label} {int(cnt)}x"
                    f" (severity {round((avg_sev or 0) * 10, 1)}/10{r_str})"
                )
                patterns.append({
                    "description": desc,
                    "score": int(cnt) * float(avg_sev or 0),
                })
            return patterns
        except Exception as e:
            logger.debug("Behavioral pattern query failed: %s", e)
            return []

    # ── Day-of-week overtrading ───────────────────────────────────

    def _query_overtrade_days(self, trader_id: str, since: datetime) -> list[dict]:
        """Detect days where trade count was consistently high."""
        try:
            from sqlalchemy import text
            result = self._db.execute(
                text("""
                    SELECT
                        TO_CHAR(entry_time AT TIME ZONE 'Asia/Kolkata', 'Day') AS day_name,
                        COUNT(*) AS total_trades,
                        COUNT(DISTINCT DATE(entry_time AT TIME ZONE 'Asia/Kolkata')) AS trading_days,
                        AVG(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) AS avg_pnl
                    FROM trades
                    WHERE entry_time >= :since
                      AND (trader_id = :tid OR trader_id IS NULL)
                    GROUP BY TO_CHAR(entry_time AT TIME ZONE 'Asia/Kolkata', 'Day')
                    HAVING COUNT(*) > 3
                    ORDER BY COUNT(*) / NULLIF(COUNT(DISTINCT DATE(entry_time AT TIME ZONE 'Asia/Kolkata')), 0) DESC
                    LIMIT 2
                """),
                {"since": since, "tid": trader_id},
            )
            rows = result.fetchall()
            patterns = []
            for row in rows:
                day_name, total, days, avg_pnl = row
                if not days or days == 0:
                    continue
                avg_per_day = total / days
                if avg_per_day < 4:
                    continue  # Not remarkable
                pnl_note = ""
                if avg_pnl and avg_pnl < -200:
                    pnl_note = f" with avg ₹{avg_pnl:,.0f} P&L"
                desc = (
                    f"You averaged {avg_per_day:.1f} trades on {day_name.strip()}s"
                    f" (possible overtrading{pnl_note})"
                )
                patterns.append({
                    "description": desc,
                    "score": avg_per_day * 0.3,  # lower weight than explicit mistakes
                })
            return patterns
        except Exception as e:
            logger.debug("Day pattern query failed: %s", e)
            return []
