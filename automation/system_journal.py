# automation/system_journal.py
"""
System Journal for Autonomous Virtual Trades
=============================================
Records all autonomous shadow trades (created by the system, not the trader)
in a structured journal for review, comparison, and coaching.

The journal serves three critical purposes:

1. **Transparency**: Every autonomous action is logged with full context
   (signal, conviction, variant, regime) so the trader can audit the system.

2. **Competition layer**: Feeds into Sprint 4's comparison framework —
   "how did the system do vs. the trader this week?"

3. **Learning loop**: Closed entries feed back into the SKE to improve
   signal quality and variant scoring.

Journal entries are stored in the `system_journal_entries` table and
include a lifecycle: created → [updated...] → closed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from core.events import Event, EventType, get_event_bus
from shadow.models import ShadowTrade

logger = logging.getLogger(__name__)


class JournalEntry:
    """
    In-memory representation of a system journal entry.

    Used for serialization and API responses. The actual data
    is persisted via SQLAlchemy ORM or raw JSON storage.
    """

    def __init__(
        self,
        entry_id: str,
        shadow_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: Optional[float],
        position_size: int,
        conviction_score: float,
        conviction_band: str,
        sko_name: str = "",
        matched_variant: Optional[str] = None,
        variant_provenance: Optional[str] = None,
        regime_match_score: float = 0.0,
        status: str = "open",
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        exit_reason: Optional[str] = None,
        pnl: Optional[float] = None,
        r_multiple: Optional[float] = None,
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.entry_id = entry_id
        self.shadow_id = shadow_id
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.conviction_score = conviction_score
        self.conviction_band = conviction_band
        self.sko_name = sko_name
        self.matched_variant = matched_variant
        self.variant_provenance = variant_provenance
        self.regime_match_score = regime_match_score
        self.status = status
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.pnl = pnl
        self.r_multiple = r_multiple
        self.notes = notes
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "shadow_id": self.shadow_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "conviction_score": round(self.conviction_score, 4),
            "conviction_band": self.conviction_band,
            "sko_name": self.sko_name,
            "matched_variant": self.matched_variant,
            "variant_provenance": self.variant_provenance,
            "regime_match_score": round(self.regime_match_score, 4),
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "pnl": round(self.pnl, 2) if self.pnl is not None else None,
            "r_multiple": round(self.r_multiple, 2) if self.r_multiple is not None else None,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class SystemJournal:
    """
    Manages the system's autonomous trade journal.

    Lifecycle:
        1. record_entry()     — when autonomous shadow is created
        2. update_entry()     — price updates, trailing stop changes
        3. close_entry()      — shadow exits (stop/target/time)
        4. get_open_entries()  — currently active system trades
        5. get_history()      — closed entries for analysis
    """

    def __init__(self, db: DBSession):
        self.db = db
        self._event_bus = get_event_bus()

    def record_entry(self, shadow: ShadowTrade) -> JournalEntry:
        """
        Create a journal entry from a newly created autonomous shadow trade.
        """
        config = shadow.config or {}

        entry = JournalEntry(
            entry_id=str(uuid.uuid4()),
            shadow_id=str(shadow.id),
            symbol=shadow.symbol,
            direction=shadow.direction if isinstance(shadow.direction, str) else shadow.direction.value,
            entry_price=shadow.entry_price,
            entry_time=shadow.entry_time or datetime.now(timezone.utc),
            stop_loss=shadow.stop_loss,
            take_profit=shadow.take_profit,
            position_size=int(shadow.position_size),
            conviction_score=config.get("conviction_score", 0.0),
            conviction_band=config.get("conviction_band", "unknown"),
            sko_name=config.get("sko_name", ""),
            matched_variant=config.get("matched_variant"),
            variant_provenance=config.get("variant_provenance"),
            regime_match_score=config.get("regime_match_score", 0.0),
            status="open",
            metadata={
                "sizing_recommendation": config.get("sizing_recommendation", "standard"),
                "variant_conviction_boost": config.get("variant_conviction_boost", 0.0),
                "source": "autonomous_shadow_creator",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Persist to database as JSON in a journal table
        # For now, we store it in the shadow trade's config for simplicity
        shadow_config = dict(shadow.config or {})
        shadow_config["journal_entry_id"] = entry.entry_id
        shadow_config["journal_status"] = "open"
        shadow.config = shadow_config
        self.db.commit()

        logger.info(
            "System journal: OPEN %s %s @ %.2f | conviction=%.2f | variant=%s",
            entry.direction.upper(),
            entry.symbol,
            entry.entry_price,
            entry.conviction_score,
            entry.matched_variant or "none",
        )

        self._event_bus.publish(Event(
            event_type=EventType.LEARNING_SESSION_COMPLETED,  # Closest event type
            source="system_journal",
            data={
                "action": "entry_opened",
                "entry_id": entry.entry_id,
                "symbol": entry.symbol,
                "direction": entry.direction,
                "conviction": entry.conviction_score,
            },
        ))

        return entry

    def close_entry(
        self,
        shadow: ShadowTrade,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        exit_reason: str = "unknown",
    ) -> JournalEntry:
        """
        Close a journal entry when its shadow trade exits.
        """
        config = shadow.config or {}
        now = exit_time or datetime.now(timezone.utc)

        # Compute P&L
        direction = shadow.direction if isinstance(shadow.direction, str) else shadow.direction.value
        size = shadow.position_size or 1

        if direction == "long":
            pnl = (exit_price - shadow.entry_price) * size
        else:
            pnl = (shadow.entry_price - exit_price) * size

        # Compute R-multiple
        risk_per_share = abs(shadow.entry_price - (shadow.original_stop or shadow.stop_loss))
        r_multiple = (pnl / size) / risk_per_share if risk_per_share > 0 else 0.0

        entry = JournalEntry(
            entry_id=config.get("journal_entry_id", str(uuid.uuid4())),
            shadow_id=str(shadow.id),
            symbol=shadow.symbol,
            direction=direction,
            entry_price=shadow.entry_price,
            entry_time=shadow.entry_time or now,
            stop_loss=shadow.stop_loss,
            take_profit=shadow.take_profit,
            position_size=int(size),
            conviction_score=config.get("conviction_score", 0.0),
            conviction_band=config.get("conviction_band", "unknown"),
            sko_name=config.get("sko_name", ""),
            matched_variant=config.get("matched_variant"),
            variant_provenance=config.get("variant_provenance"),
            regime_match_score=config.get("regime_match_score", 0.0),
            status="closed",
            exit_price=exit_price,
            exit_time=now,
            exit_reason=exit_reason,
            pnl=pnl,
            r_multiple=r_multiple,
        )

        # Update shadow config with journal close data
        shadow_config = dict(config)
        shadow_config["journal_status"] = "closed"
        shadow_config["journal_exit_reason"] = exit_reason
        shadow_config["journal_pnl"] = round(pnl, 2)
        shadow_config["journal_r_multiple"] = round(r_multiple, 4)
        shadow_config["journal_closed_at"] = now.isoformat()
        shadow.config = shadow_config
        self.db.commit()

        logger.info(
            "System journal: CLOSE %s %s | exit=%.2f | reason=%s | "
            "P&L=%.2f | R=%.2fR | conviction was %.2f",
            direction.upper(),
            shadow.symbol,
            exit_price,
            exit_reason,
            pnl,
            r_multiple,
            config.get("conviction_score", 0.0),
        )

        return entry

    def get_open_entries(self) -> List[JournalEntry]:
        """Get all currently open autonomous entries."""
        shadows = (
            self.db.query(ShadowTrade)
            .filter(
                ShadowTrade.status == "open",
                ShadowTrade.shadow_type == "autonomous",
            )
            .all()
        )

        entries = []
        for s in shadows:
            config = s.config or {}
            entries.append(JournalEntry(
                entry_id=config.get("journal_entry_id", str(s.id)),
                shadow_id=str(s.id),
                symbol=s.symbol,
                direction=s.direction if isinstance(s.direction, str) else s.direction.value,
                entry_price=s.entry_price,
                entry_time=s.entry_time,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                position_size=int(s.position_size or 0),
                conviction_score=config.get("conviction_score", 0.0),
                conviction_band=config.get("conviction_band", "unknown"),
                sko_name=config.get("sko_name", ""),
                matched_variant=config.get("matched_variant"),
                variant_provenance=config.get("variant_provenance"),
                regime_match_score=config.get("regime_match_score", 0.0),
                status="open",
            ))

        return entries

    def get_history(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> List[JournalEntry]:
        """Get closed autonomous entries for review."""
        query = (
            self.db.query(ShadowTrade)
            .filter(
                ShadowTrade.status.in_(["closed", "stopped", "target_hit"]),
                ShadowTrade.shadow_type == "autonomous",
            )
        )

        if symbol:
            query = query.filter(ShadowTrade.symbol == symbol)

        shadows = query.order_by(ShadowTrade.exit_time.desc()).limit(limit).all()

        entries = []
        for s in shadows:
            config = s.config or {}
            entries.append(JournalEntry(
                entry_id=config.get("journal_entry_id", str(s.id)),
                shadow_id=str(s.id),
                symbol=s.symbol,
                direction=s.direction if isinstance(s.direction, str) else s.direction.value,
                entry_price=s.entry_price,
                entry_time=s.entry_time,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                position_size=int(s.position_size or 0),
                conviction_score=config.get("conviction_score", 0.0),
                conviction_band=config.get("conviction_band", "unknown"),
                sko_name=config.get("sko_name", ""),
                matched_variant=config.get("matched_variant"),
                exit_price=s.exit_price,
                exit_time=s.exit_time,
                exit_reason=config.get("journal_exit_reason", "unknown"),
                pnl=config.get("journal_pnl"),
                r_multiple=config.get("journal_r_multiple"),
                status="closed",
            ))

        return entries

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Aggregate performance of all closed autonomous entries.
        Used for the competition dashboard.
        """
        entries = self.get_history(limit=500)
        if not entries:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_r": 0.0,
                "expectancy_r": 0.0,
                "best_trade_r": 0.0,
                "worst_trade_r": 0.0,
            }

        total = len(entries)
        winners = [e for e in entries if e.pnl and e.pnl > 0]
        losers = [e for e in entries if e.pnl and e.pnl <= 0]
        win_rate = len(winners) / total if total > 0 else 0.0
        total_pnl = sum(e.pnl for e in entries if e.pnl is not None)

        r_multiples = [e.r_multiple for e in entries if e.r_multiple is not None]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

        avg_win_r = (
            sum(r for r in r_multiples if r > 0) / len([r for r in r_multiples if r > 0])
            if any(r > 0 for r in r_multiples) else 0.0
        )
        avg_loss_r = (
            sum(abs(r) for r in r_multiples if r <= 0) / len([r for r in r_multiples if r <= 0])
            if any(r <= 0 for r in r_multiples) else 1.0
        )
        expectancy_r = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

        return {
            "total_trades": total,
            "wins": len(winners),
            "losses": len(losers),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_r": round(avg_r, 4),
            "expectancy_r": round(expectancy_r, 4),
            "best_trade_r": round(max(r_multiples), 2) if r_multiples else 0.0,
            "worst_trade_r": round(min(r_multiples), 2) if r_multiples else 0.0,
        }
