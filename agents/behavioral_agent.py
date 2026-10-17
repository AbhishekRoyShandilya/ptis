"""
Behavioral Agent (Autonomous) — Orchestrator
==============================================

Pipelines all the pieces:
  ingest trade history (from `trades` table — already populated by trade_sync)
  → replay each trade with TradeReplayEngine
  → match each trade to strategy with TradeStrategyMatcher
  → link causes with CausalLinker
  → persist trade_replays + populate StrategyAttribution
  → synthesize per-trader BehaviorProfile (capped to 1-3 strengths + 1-3 weaknesses)
  → return summary

Triggered by:
  - Manual via API: POST /api/v1/behavioral-agent/run
  - On broker linking: initial batch
  - On TRADE_CLOSED event: incremental (single trade)
  - Weekly cron: full recompute (refreshes patterns, captures regime shifts)

Real-money safety:
  - Read-only and advisory throughout — never trades
  - Insights capped at 1-3 (per partner's "no lecture system" warning)
  - Idempotent — re-run safely; existing replays reused unless force=True
  - Per-trader scoping — never mixes users' data
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Insight cap (per partner's strict "1-3 max" rule) ──────────────


MAX_STRENGTHS = 3
MAX_WEAKNESSES = 3


# ── Output dataclasses ─────────────────────────────────────────────


@dataclass
class BehavioralRunSummary:
    """Result returned by BehavioralAgent.run() — for status endpoint + logs."""
    trader_id: str
    trades_processed: int
    replays_created: int
    replays_reused: int
    matches_attributed: int
    causes_linked: int
    profile_id: str
    biggest_issue_pattern: Optional[str]
    started_at: str
    finished_at: str
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


# ── Orchestrator ───────────────────────────────────────────────────


class BehavioralAgent:
    """
    The Behavioral Agent. One instance per process; reuse across runs.

    Usage:
        agent = BehavioralAgent(db_session_factory, market_data_service)
        summary = await agent.run(trader_id="...", since_days=90)
    """

    def __init__(
        self,
        db_session_factory: Any,
        market_data_service: Any,
        replay_engine: Any = None,
        matcher: Any = None,
        linker: Any = None,
        clusterer: Any = None,
    ):
        self._db_factory = db_session_factory
        self._mds = market_data_service

        # Lazy-import to avoid circular deps + heavy modules at import time
        from behavioral.trade_replay import TradeReplayEngine
        from behavioral.trade_matcher import TradeStrategyMatcher
        from behavioral.causal_linker import CausalLinker

        self.replay_engine = replay_engine or TradeReplayEngine(market_data_service)
        self.matcher = matcher or TradeStrategyMatcher()
        self.linker = linker or CausalLinker()

        # Optional clusterer (for archetype tag) — graceful if sklearn missing
        if clusterer is None:
            try:
                from ml.behavior_clustering import BehaviorClusterer
                clusterer = BehaviorClusterer()
            except Exception as e:
                logger.debug("BehaviorClusterer unavailable: %s", e)
                clusterer = None
        self.clusterer = clusterer

    # ── Public API ──

    def run(
        self,
        trader_id: str,
        since_days: int = 90,
        force_replay: bool = False,
    ) -> BehavioralRunSummary:
        """
        Run the full pipeline for a trader.

        Args:
            trader_id: trader UUID (string form)
            since_days: only process trades from last N days (default 90)
            force_replay: if True, re-replay even trades that already have a TradeReplay row

        Returns:
            BehavioralRunSummary with counts + biggest issue pattern
        """
        start_time = datetime.now(timezone.utc)
        errors: List[str] = []

        replays_created = 0
        replays_reused = 0
        matches_attributed = 0
        causes_linked = 0
        biggest_issue_pattern: Optional[str] = None
        profile_id: str = ""

        db = self._db_factory()
        try:
            trades = self._load_trades(db, trader_id, since_days)
            logger.info("BA: loaded %d trades for trader %s (since %dd)",
                        len(trades), trader_id, since_days)

            # The user's own taught strategies — so trades are matched only
            # against THIS trader's strategies, never another user's.
            user_strategy_ids = self._load_user_strategy_ids(db, trader_id)

            replay_results: List[Tuple[Any, Any]] = []  # (trade, replay)
            match_results: List[Tuple[Any, Any]] = []   # (trade, match_result)
            cause_results: List[Tuple[Any, Any]] = []   # (trade, explanation)

            for trade in trades:
                try:
                    # Step 1: replay (or reuse existing)
                    replay, was_reused = self._get_or_replay(db, trade, force=force_replay)
                    if was_reused:
                        replays_reused += 1
                    else:
                        replays_created += 1
                    replay_results.append((trade, replay))

                    # Step 2: match
                    declared = getattr(trade, "declared_strategy_id", None)
                    match = self.matcher.match(
                        replay, declared_strategy_id=declared,
                        allowed_strategy_ids=user_strategy_ids,
                    )
                    if match.best_match_id:
                        self._persist_attribution(db, trade, match)
                        matches_attributed += 1
                    match_results.append((trade, match))

                    # Step 3: link causes
                    pnl = getattr(trade, "pnl", None)
                    rmult = getattr(trade, "r_multiple", None)
                    explanation = self.linker.link(
                        replay, match, declared, trade_pnl=pnl, trade_r_multiple=rmult,
                    )
                    if explanation.causes:
                        causes_linked += 1
                    cause_results.append((trade, explanation))
                except Exception as e:
                    logger.warning("BA: per-trade processing failed for %s: %s",
                                   getattr(trade, "id", "?"), e)
                    errors.append(f"trade {getattr(trade, 'id', '?')}: {e}")

            # Step 4: synthesize profile (cap to 1-3 strengths + weaknesses)
            try:
                profile_payload = self._synthesize(
                    trader_id=trader_id,
                    trades=trades,
                    replays=replay_results,
                    matches=match_results,
                    causes=cause_results,
                )
                profile_id = self._persist_profile(db, trader_id, profile_payload, len(trades))
                if profile_payload.get("biggest_issue"):
                    biggest_issue_pattern = profile_payload["biggest_issue"].get("pattern")
            except Exception as e:
                logger.error("BA: synthesis failed: %s", e, exc_info=True)
                errors.append(f"synthesis: {e}")

            db.commit()
        except Exception as e:
            logger.error("BA: run failed: %s", e, exc_info=True)
            errors.append(f"top-level: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

        end_time = datetime.now(timezone.utc)
        return BehavioralRunSummary(
            trader_id=trader_id,
            trades_processed=len(trades) if "trades" in dir() else 0,
            replays_created=replays_created,
            replays_reused=replays_reused,
            matches_attributed=matches_attributed,
            causes_linked=causes_linked,
            profile_id=profile_id,
            biggest_issue_pattern=biggest_issue_pattern,
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            errors=errors,
        )

    # ── Trade loading ──

    def _load_trades(self, db: Any, trader_id: str, since_days: int) -> List[Any]:
        """Load closed trades for trader from last N days."""
        try:
            from core.models import Trade
            from core.enums import TradeStatus
        except ImportError:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        try:
            q = (
                db.query(Trade)
                .filter(Trade.trader_id == trader_id)
                .filter(Trade.status == TradeStatus.CLOSED)
                .filter(Trade.entry_time >= cutoff)
                .order_by(Trade.entry_time.asc())
            )
            return q.all()
        except Exception as e:
            logger.warning("BA: trade load query failed: %s", e)
            return []

    def _load_user_strategy_ids(self, db: Any, trader_id: str) -> List[str]:
        """
        The caller's own active strategy short_codes — used to scope trade↔strategy
        matching so a user's trades are never matched against another user's
        strategies. Empty list means "no scoping" (match falls back to all active).
        """
        try:
            from core.models import StrategyProfile
            rows = (
                db.query(StrategyProfile.short_code)
                .filter(StrategyProfile.trader_id == trader_id)
                .filter(StrategyProfile.status.in_(["active", "learning"]))
                .all()
            )
            return [r[0] for r in rows if r[0]]
        except Exception as e:
            logger.warning("BA: user strategy load failed: %s", e)
            return []

    # ── Replay (with cache reuse) ──

    def _get_or_replay(self, db: Any, trade: Any, force: bool = False) -> Tuple[Any, bool]:
        """Return (ReplayedTrade, was_reused). Reuses existing TradeReplay row if present."""
        from core.models import TradeReplay as TradeReplayORM

        trade_id_str = str(trade.id)

        if not force:
            existing = db.query(TradeReplayORM).filter_by(trade_id=trade.id).first()
            if existing is not None:
                # Hydrate ReplayedTrade dataclass from payload
                replay = self._hydrate_replay(trade, existing)
                return replay, True

        # Fresh replay
        replay = self.replay_engine.replay_trade(trade)
        # Persist
        try:
            obs = replay.observations
            row = db.query(TradeReplayORM).filter_by(trade_id=trade.id).first()
            if row is None:
                row = TradeReplayORM(
                    id=trade_id_str,
                    trade_id=trade.id,
                )
                db.add(row)
            row.tier = replay.tier
            row.symbol = replay.symbol
            row.entry_time = replay.entry_time
            row.exit_time = replay.exit_time
            row.regime_at_entry = obs.regime_at_entry or None
            row.regime_at_exit = obs.regime_at_exit or None
            row.trend_alignment = obs.trend_alignment_at_entry or None
            row.peak_r = obs.peak_r_during_trade
            row.final_r = obs.final_r
            row.early_exit_severity = obs.early_exit_severity
            row.move_pct_complete_at_entry = obs.move_pct_complete_at_entry
            row.rsi_at_entry = obs.rsi_at_entry
            row.payload = replay.to_dict()
            row.candles_fetched = replay.candles_fetched
            row.fetch_errors = replay.fetch_errors or None
            row.replayed_at = replay.replayed_at
            db.flush()
        except Exception as e:
            logger.warning("BA: replay persist failed for %s: %s", trade_id_str, e)
        return replay, False

    @staticmethod
    def _hydrate_replay(trade: Any, row: Any) -> Any:
        """Reconstruct ReplayedTrade dataclass from a stored DB row."""
        from behavioral.replay_models import ReplayedTrade, ReplayObservations

        replay = ReplayedTrade(
            trade_id=str(trade.id),
            symbol=row.symbol or getattr(trade, "symbol", ""),
            direction=str(getattr(trade, "direction", "long")).lower().split(".")[-1],
            entry_time=row.entry_time or getattr(trade, "entry_time", None),
            exit_time=row.exit_time,
            entry_price=float(getattr(trade, "entry_price", 0.0) or 0.0),
            exit_price=getattr(trade, "exit_price", None),
            stop_loss=getattr(trade, "stop_loss", None),
            tier=row.tier or "standard",
            replayed_at=row.replayed_at,
        )
        # Lift the observations from row columns (more reliable than parsing payload)
        replay.observations = ReplayObservations(
            move_pct_complete_at_entry=row.move_pct_complete_at_entry,
            rsi_at_entry=row.rsi_at_entry,
            peak_r_during_trade=row.peak_r or 0.0,
            final_r=row.final_r or 0.0,
            early_exit_severity=row.early_exit_severity or 0.0,
            regime_at_entry=row.regime_at_entry or "",
            regime_at_exit=row.regime_at_exit or "",
            regime_changed=bool(
                row.regime_at_entry and row.regime_at_exit
                and row.regime_at_entry != row.regime_at_exit
            ),
            trend_alignment_at_entry=row.trend_alignment or None,
        )
        return replay

    # ── Strategy attribution persistence ──

    @staticmethod
    def _persist_attribution(db: Any, trade: Any, match_result: Any) -> None:
        """Update StrategyAttribution.tier2_* fields with matcher output."""
        try:
            from core.models import StrategyAttribution
        except ImportError:
            return
        try:
            attr = db.query(StrategyAttribution).filter_by(trade_id=trade.id).first()
            if attr is None:
                attr = StrategyAttribution(trade_id=trade.id)
                db.add(attr)
            attr.tier2_best_match_id = match_result.best_match_id
            attr.tier2_matches = [
                {
                    "strategy_id": c.strategy_id,
                    "strategy_name": c.strategy_name,
                    "total_score": c.total_score,
                    "required_score": c.required_score,
                }
                for c in match_result.candidates[:5]
            ]
            db.flush()
        except Exception as e:
            logger.debug("Attribution persist skipped (%s): %s", getattr(trade, "id", "?"), e)

    # ── Synthesis: build the BehaviorProfile payload ──

    def _synthesize(
        self,
        trader_id: str,
        trades: List[Any],
        replays: List[Tuple[Any, Any]],
        matches: List[Tuple[Any, Any]],
        causes: List[Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """
        Synthesize per-trader profile from per-trade pieces.

        Strict cap: max 3 strengths + 3 weaknesses + 1 biggest_issue.
        Patterns ranked by frequency × severity × recency.
        """
        # Aggregate cause counts (negative + positive)
        neg_counter: Counter = Counter()
        pos_counter: Counter = Counter()
        cause_severities: Dict[str, List[float]] = {}
        cause_examples: Dict[str, str] = {}  # pattern → one detail line
        cause_recent_dates: Dict[str, datetime] = {}  # most recent occurrence

        for trade, expl in causes:
            entry_time = getattr(trade, "entry_time", None)
            for c in expl.causes:
                if c.is_positive:
                    pos_counter[c.cause] += 1
                else:
                    neg_counter[c.cause] += 1
                cause_severities.setdefault(c.cause, []).append(c.severity)
                cause_examples.setdefault(c.cause, c.detail)
                if entry_time:
                    if c.cause not in cause_recent_dates or entry_time > cause_recent_dates[c.cause]:
                        cause_recent_dates[c.cause] = entry_time

        n_trades = len(trades)

        # Score patterns: frequency × avg_severity × recency_weight
        def _score_pattern(cause: str, count: int) -> float:
            avg_sev = sum(cause_severities.get(cause, [0])) / max(len(cause_severities.get(cause, [1])), 1)
            freq = count / max(n_trades, 1)
            # Recency weight: 1.0 if last seen ≤ 7d ago, falls off slowly
            recent = cause_recent_dates.get(cause)
            recency = 0.5
            if recent:
                age_days = (datetime.now(timezone.utc) - self._ensure_aware(recent)).days
                recency = max(0.3, 1.0 - age_days / 90.0)
            return freq * avg_sev * recency

        # Top weaknesses (capped)
        ranked_neg = sorted(
            ((cause, count) for cause, count in neg_counter.items()),
            key=lambda kv: _score_pattern(*kv),
            reverse=True,
        )
        weaknesses = []
        for cause, count in ranked_neg[:MAX_WEAKNESSES]:
            avg_sev = sum(cause_severities.get(cause, [0])) / max(len(cause_severities.get(cause, [1])), 1)
            weaknesses.append({
                "pattern": cause,
                "frequency": count,
                "frequency_pct": round(count / max(n_trades, 1) * 100, 1),
                "avg_severity": round(avg_sev, 3),
                "example": cause_examples.get(cause, ""),
            })

        # Top strengths (capped)
        ranked_pos = sorted(
            ((cause, count) for cause, count in pos_counter.items()),
            key=lambda kv: _score_pattern(*kv),
            reverse=True,
        )
        strengths = []
        for cause, count in ranked_pos[:MAX_STRENGTHS]:
            strengths.append({
                "pattern": cause,
                "frequency": count,
                "frequency_pct": round(count / max(n_trades, 1) * 100, 1),
                "example": cause_examples.get(cause, ""),
            })

        # Biggest issue (the single weakness with highest score)
        biggest_issue = None
        if weaknesses:
            top = weaknesses[0]
            # Compute impact text (only loss trades for this pattern)
            losses_with_pattern = [
                t for t, e in causes
                if any(c.cause == top["pattern"] for c in e.causes)
                and (getattr(t, "pnl", 0) or 0) < 0
            ]
            biggest_issue = {
                "pattern": top["pattern"],
                "frequency": top["frequency"],
                "frequency_pct": top["frequency_pct"],
                "loss_count": len(losses_with_pattern),
                "sample_size": n_trades,
                "impact_text": (
                    f"Occurred in {top['frequency']}/{n_trades} trades "
                    f"({top['frequency_pct']}%); {len(losses_with_pattern)} of those lost. "
                    f"{top.get('example', '')}"
                ),
            }

        # Best strategy = strategy with highest win rate among matched strategies
        strategy_outcomes: Dict[str, List[bool]] = {}  # strategy_id → [is_win, ...]
        for trade, match in matches:
            if not match.best_match_id:
                continue
            r = getattr(trade, "r_multiple", None)
            pnl = getattr(trade, "pnl", None)
            is_win = (r and r > 0) or (pnl and pnl > 0)
            strategy_outcomes.setdefault(match.best_match_id, []).append(bool(is_win))

        best_strategy: Optional[str] = None
        best_strategy_wr = 0.0
        for sid, outcomes in strategy_outcomes.items():
            if len(outcomes) < 5:  # need min sample
                continue
            wr = sum(outcomes) / len(outcomes)
            if wr > best_strategy_wr:
                best_strategy_wr = wr
                best_strategy = sid

        # Worst pattern = highest-scored weakness
        worst_pattern = weaknesses[0]["pattern"] if weaknesses else None

        # Causal summary (top 3 negative patterns, light-weight for API)
        causal_summary = [
            {
                "cause": w["pattern"],
                "frequency": w["frequency"],
                "frequency_pct": w["frequency_pct"],
            }
            for w in weaknesses[:3]
        ]

        # Discipline score: simple inverse of weighted negative pattern share
        if n_trades > 0:
            neg_weighted = sum(
                neg_counter.get(c, 0)
                * (sum(cause_severities.get(c, [0])) / max(len(cause_severities.get(c, [1])), 1))
                for c in neg_counter
            )
            discipline_score = round(max(0.0, min(1.0, 1.0 - neg_weighted / (n_trades * 1.5))), 4)
        else:
            discipline_score = None

        # Archetype clustering (optional)
        archetype = None
        if self.clusterer is not None and len(trades) >= 10:
            try:
                arch_result = self.clusterer.cluster(
                    [self._trade_to_cluster_dict(t) for t in trades]
                )
                if arch_result and getattr(arch_result, "clusters", None):
                    # Use the dominant cluster's label
                    main_cluster = max(arch_result.clusters, key=lambda c: c.trade_count)
                    archetype = getattr(main_cluster, "label", None)
            except Exception as e:
                logger.debug("Archetype clustering skipped: %s", e)

        # First/last trade timestamps
        if trades:
            entry_times = [self._ensure_aware(getattr(t, "entry_time", None))
                           for t in trades if getattr(t, "entry_time", None)]
            entry_times = [t for t in entry_times if t]
            first_trade_at = min(entry_times) if entry_times else None
            last_trade_at = max(entry_times) if entry_times else None
        else:
            first_trade_at = None
            last_trade_at = None

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "biggest_issue": biggest_issue,
            "best_strategy": best_strategy,
            "worst_pattern": worst_pattern,
            "discipline_score": discipline_score,
            "causal_summary": causal_summary,
            "archetype": archetype,
            "first_trade_at": first_trade_at,
            "last_trade_at": last_trade_at,
        }

    @staticmethod
    def _trade_to_cluster_dict(trade: Any) -> Dict[str, Any]:
        """Build a dict in the shape ml/behavior_clustering expects."""
        return {
            "discipline_score": getattr(trade, "discipline_score", None),
            "override_count": 0,
            "emotion_pre": str(getattr(trade, "emotion_pre", "neutral")).split(".")[-1],
            "hold_min": 0,
            "risk_pct": getattr(trade, "risk_pct", None) or 1.0,
            "pnl_sign": 1 if (getattr(trade, "pnl", 0) or 0) > 0 else -1,
        }

    @staticmethod
    def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    # ── Profile persistence ──

    def _persist_profile(
        self, db: Any, trader_id: str, payload: Dict[str, Any], n_trades: int,
    ) -> str:
        """Upsert BehaviorProfile row. Returns the profile id."""
        try:
            from core.models import BehaviorProfile
        except ImportError:
            return ""

        now = datetime.now(timezone.utc)
        profile_id = f"bp_{trader_id}"

        try:
            row = db.query(BehaviorProfile).filter_by(trader_id=trader_id).first()
            if row is None:
                row = BehaviorProfile(id=profile_id, trader_id=trader_id)
                db.add(row)

            row.strengths = payload.get("strengths", [])
            row.weaknesses = payload.get("weaknesses", [])
            row.biggest_issue = payload.get("biggest_issue")
            row.best_strategy = payload.get("best_strategy")
            row.worst_pattern = payload.get("worst_pattern")
            row.discipline_score = payload.get("discipline_score")
            row.causal_summary = payload.get("causal_summary", [])
            row.archetype = payload.get("archetype")
            row.trades_analyzed_count = n_trades
            row.first_trade_at = payload.get("first_trade_at")
            row.last_trade_at = payload.get("last_trade_at")
            row.last_analyzed_at = now
            db.flush()
            return row.id
        except Exception as e:
            logger.error("Profile persist failed: %s", e)
            return ""


# ── Module singleton helper ────────────────────────────────────────


_singleton: Optional[BehavioralAgent] = None


def get_behavioral_agent(
    db_session_factory: Optional[Any] = None,
    market_data_service: Optional[Any] = None,
) -> Optional[BehavioralAgent]:
    """Return the module-level singleton; lazy-init on first call."""
    global _singleton
    if _singleton is not None:
        return _singleton
    if db_session_factory is None or market_data_service is None:
        return None
    _singleton = BehavioralAgent(
        db_session_factory=db_session_factory,
        market_data_service=market_data_service,
    )
    return _singleton
