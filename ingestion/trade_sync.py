# ingestion/trade_sync.py
"""
Groww Trade Sync Service — auto-imports trades from Groww broker.

Polls Groww order book every sync_interval seconds, deduplicates by
order_id, and feeds filled orders into the TradePipeline as new trades.

BUY orders → enter_trade (open a new position)
SELL orders → find matching open trade by symbol → close_trade

Architecture:
  TradeSyncService.run()        — asyncio background loop
  TradeSyncService.sync_now()   — manual trigger (from API)
  init_trade_sync_service()     — singleton factory
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

from config.settings import settings

logger = logging.getLogger(__name__)

# Default sync interval (seconds)
DEFAULT_SYNC_INTERVAL = 120
DEFAULT_LOOKBACK_DAYS = 1


class TradeSyncService:
    """
    Background service that polls Groww orders and feeds them into PTIS.
    """

    def __init__(
        self,
        groww_client,
        db_factory,
        sync_interval: int = DEFAULT_SYNC_INTERVAL,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ):
        self._groww = groww_client
        self._db_factory = db_factory
        self._sync_interval = sync_interval
        self._lookback_days = lookback_days

        # In-memory dedup set (order_ids already processed this session)
        self._synced_order_ids: Set[str] = set()

        # User the Groww account belongs to — resolved each cycle so synced
        # trades are attributed to them (str(user.id) or None).
        self._owner_trader_id: Optional[str] = None

        # Stats
        self._running = False
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0
        self._total_synced = 0

    # ── Main Loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Background loop — polls Groww every sync_interval seconds."""
        self._running = True
        logger.info(
            "Trade sync started (interval=%ds, lookback=%dd)",
            self._sync_interval, self._lookback_days,
        )

        while self._running:
            try:
                await self._sync_cycle()
            except Exception as e:
                self._error_count += 1
                logger.error("Trade sync cycle error: %s", e)

            await asyncio.sleep(self._sync_interval)

        logger.info("Trade sync stopped")

    def stop(self) -> None:
        self._running = False

    # ── Sync Cycle ────────────────────────────────────────────────

    async def _sync_cycle(self) -> Dict[str, Any]:
        """
        One sync cycle:
        1. Fetch filled orders from Groww (last N days)
        2. Skip already-synced order_ids
        3. Check DB for duplicate (external_id match)
        4. BUY → enter_trade, SELL → close matching open trade
        """
        loop = asyncio.get_event_loop()
        from_date = date.today() - timedelta(days=self._lookback_days)

        # Resolve the owning user once per cycle. Without one, do NOT create
        # orphaned (trader_id=NULL) trades — every read filters by trader_id,
        # so orphans would be invisible. Orders retry on a later cycle.
        self._owner_trader_id = self._resolve_owner_trader_id()
        if not self._owner_trader_id:
            logger.info(
                "Trade sync: no user to attribute trades to yet — skipping "
                "this cycle (sign up first)."
            )
            return {"synced": 0, "skipped": 0, "errors": 0, "reason": "no_user"}

        # Fetch orders in executor (Groww client is sync)
        try:
            orders = await loop.run_in_executor(
                None, lambda: self._groww.get_orders(from_date=from_date, status="filled")
            )
        except Exception as e:
            logger.warning("Groww order fetch failed: %s", e)
            self._error_count += 1
            return {"synced": 0, "skipped": 0, "errors": 1}

        synced = 0
        skipped = 0
        errors = 0

        for order in orders:
            order_id = str(order.get("order_id", ""))
            if not order_id:
                continue

            # Skip if already processed this session
            if order_id in self._synced_order_ids:
                skipped += 1
                continue

            # Process order
            try:
                result = await loop.run_in_executor(
                    None, self._process_order, order,
                )
                if result == "synced":
                    synced += 1
                    self._total_synced += 1
                elif result == "duplicate":
                    skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning("Order %s sync failed: %s", order_id, e)
                errors += 1

            # Mark as seen regardless
            self._synced_order_ids.add(order_id)

        # Cross-day closes: the intraday order book only has today's leg, so pairing from orders misses a
        # position opened on an EARLIER day and closed now. Recover those from the closed-position record.
        closing_orders = {str(o.get("symbol", "")).upper(): o for o in orders}
        try:
            synced += await loop.run_in_executor(None, self._sync_closed_positions, closing_orders)
        except Exception as e:
            logger.warning("closed-position sync failed: %s", e)

        self._sync_count += 1
        self._last_sync = datetime.now(timezone.utc)
        self._error_count += errors

        if synced > 0:
            logger.info("Trade sync: %d synced, %d skipped, %d errors", synced, skipped, errors)

        return {"synced": synced, "skipped": skipped, "errors": errors}

    def _sync_closed_positions(self, closing_orders: dict) -> int:
        """Build CLOSED trades from round-tripped Groww positions (cross-day closes the order book missed).
        `closing_orders` = {symbol: order} from today's fills — the closing order gives the prior direction
        (sell→long, buy→short) AND the REAL exit timestamp. The ENTRY time is genuinely unavailable (Groww's
        API is intraday-only — historical orders aren't exposed), so entry==exit as a placeholder (flagged in
        notes; the eye-review is skipped for these). P&L uses Groww's authoritative realised_pnl. Deduped by
        a stable (symbol, pnl) external_id. NOTE: to get FULL entry timestamps, keep the app running during
        market hours so the sync captures the BUY same-day → the close then pairs with real entry+exit."""
        from core.models import Trade, TradeStatus
        from core.enums import TradeDirection
        if not self._owner_trader_id:
            return 0
        try:
            closed = self._groww.get_closed_positions()
        except Exception as e:
            logger.debug("closed-position fetch failed: %s", e)
            return 0
        synced = 0
        db = self._get_db()
        try:
            for cp in closed:
                sym = (cp.get("symbol") or "").upper()
                qty = cp.get("quantity") or 0
                rpnl = cp.get("realised_pnl") or 0.0
                prices = sorted(p for p in (cp.get("credit_price") or 0.0, cp.get("debit_price") or 0.0) if p > 0)
                if not sym or qty <= 0 or not rpnl or not prices:
                    continue
                ext_id = f"growwpos:{sym}:{round(float(rpnl), 2)}"     # stable per closed position/day
                if db.query(Trade).filter(Trade.external_id == ext_id).first():
                    continue
                lo, hi = prices[0], prices[-1]
                # The closing order gives the prior direction (SELL→LONG, BUY→SHORT; default LONG) AND the
                # real exit timestamp. Entry time is unavailable, so entry==exit as a placeholder.
                ord_ = closing_orders.get(sym) or {}
                is_long = (str(ord_.get("side", "sell")).lower() == "sell")
                exit_time = ord_.get("filled_at") or datetime.now(timezone.utc)
                if is_long:      # realised = (exit - entry) * qty  → sign fixes which price is which
                    entry, exit_ = (lo, hi) if rpnl >= 0 else (hi, lo)
                    direction = TradeDirection.LONG
                else:            # short: realised = (entry - exit) * qty
                    entry, exit_ = (hi, lo) if rpnl >= 0 else (lo, hi)
                    direction = TradeDirection.SHORT
                db.add(Trade(
                    symbol=sym, direction=direction, status=TradeStatus.CLOSED,
                    entry_price=entry or exit_ or 1.0, exit_price=exit_, position_size=float(qty),
                    entry_time=exit_time, exit_time=exit_time, pnl=float(rpnl),
                    source="groww_position_sync", broker="groww", external_id=ext_id,
                    trader_id=self._owner_trader_id,
                    notes="Cross-day close recovered from Groww position — real EXIT time; entry time "
                          "unavailable (Groww API is intraday-only). P&L is Groww's realised_pnl.",
                ))
                db.commit()
                synced += 1
                logger.info("Synced closed position: %s pnl=%.0f (%s)", sym, rpnl, direction.value)
        finally:
            db.close()
        return synced

    def _process_order(self, order: Dict[str, Any]) -> str:
        """
        Process a single Groww order. Returns "synced", "duplicate", or "skipped".
        """
        from core.models import Trade, TradeStatus
        from core.enums import TradeDirection

        order_id = str(order.get("order_id", ""))
        side = order.get("side", "").lower()
        symbol = order.get("symbol", "").upper()
        price = float(order.get("price", 0))
        qty = int(order.get("qty", 0))
        filled_at = order.get("filled_at")

        if not symbol or price <= 0 or qty <= 0:
            return "skipped"

        db = self._get_db()
        try:
            # Check DB for duplicate
            existing = (
                db.query(Trade)
                .filter(Trade.external_id == order_id)
                .first()
            )
            if existing:
                return "duplicate"

            if side == "buy":
                return self._handle_buy(db, symbol, price, qty, filled_at, order_id)
            elif side == "sell":
                return self._handle_sell(db, symbol, price, qty, filled_at, order_id)
            else:
                return "skipped"

        finally:
            db.close()

    def _handle_buy(
        self, db, symbol: str, price: float, qty: int,
        filled_at: Optional[datetime], order_id: str,
    ) -> str:
        """Create a new trade entry from a BUY order."""
        from core.trade_pipeline import TradePipeline, TradeInput
        from core.enums import TradeDirection

        trade_input = TradeInput(
            symbol=symbol,
            direction=TradeDirection.LONG,
            entry_price=price,
            position_size=qty,
            entry_time=filled_at or datetime.now(timezone.utc),
            source="groww_sync",
            broker="groww",
            external_id=order_id,
            notes="Auto-synced from Groww",
            trader_id=self._owner_trader_id,
        )

        pipeline = TradePipeline()
        result = pipeline.enter_trade(db, trade_input)

        if result.trade_id:
            logger.info("Synced BUY: %s %d @ %.2f (order %s)", symbol, qty, price, order_id)
            return "synced"
        else:
            logger.warning("Trade entry failed for order %s: %s", order_id, result.errors)
            return "skipped"

    def _handle_sell(
        self, db, symbol: str, price: float, qty: int,
        filled_at: Optional[datetime], order_id: str,
    ) -> str:
        """Close a matching open trade from a SELL order."""
        from core.models import Trade, TradeStatus
        from core.trade_pipeline import TradePipeline

        # Find the oldest open trade for this symbol
        open_trade = (
            db.query(Trade)
            .filter(
                Trade.symbol == symbol,
                Trade.status == TradeStatus.OPEN,
            )
            .order_by(Trade.entry_time.asc())
            .first()
        )

        if not open_trade:
            # No matching open trade — log as standalone entry
            logger.info(
                "SELL order %s for %s but no open trade found — skipping (manual close needed)",
                order_id, symbol,
            )
            return "skipped"

        pipeline = TradePipeline()
        result = pipeline.close_trade(
            db=db,
            trade_id=open_trade.id,
            exit_price=price,
            exit_time=filled_at or datetime.now(timezone.utc),
            exit_notes=f"Auto-closed via Groww SELL order {order_id}",
        )

        if result.get("success"):
            logger.info(
                "Synced SELL: %s @ %.2f → closed trade %s (order %s)",
                symbol, price, open_trade.id, order_id,
            )
            # Groww doesn't return charges — estimate brokerage + taxes and
            # convert gross P&L → NET. Store the charge total in `commission`
            # so the breakdown stays transparent (gross = pnl + commission).
            try:
                from ingestion.charges import estimate_charges_for_symbol
                db.refresh(open_trade)
                ch = estimate_charges_for_symbol(
                    open_trade.symbol,
                    open_trade.entry_price or 0.0,
                    open_trade.exit_price or price,
                    open_trade.position_size or qty,
                )
                total = ch.get("total", 0.0)
                if total:
                    open_trade.commission = total
                    if open_trade.pnl is not None:
                        open_trade.pnl = round(open_trade.pnl - total, 2)
            except Exception as e:
                logger.debug("Charge estimate failed for %s: %s", order_id, e)
            # Record external_id on the trade for dedup
            open_trade.external_id = order_id
            db.commit()
            return "synced"
        else:
            logger.warning("Trade close failed for order %s: %s", order_id, result.get("error"))
            return "skipped"

    # ── Manual Trigger ────────────────────────────────────────────

    async def sync_now(self) -> Dict[str, Any]:
        """Manual sync trigger, returns {synced, skipped, errors}."""
        return await self._sync_cycle()

    # ── Status ────────────────────────────────────────────────────

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_count": self._sync_count,
            "total_synced": self._total_synced,
            "error_count": self._error_count,
            "tracked_order_ids": len(self._synced_order_ids),
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _get_db(self):
        """Get a fresh DB session from factory."""
        db = self._db_factory()
        if hasattr(db, "__next__"):
            return next(db)
        return db

    def _resolve_owner_trader_id(self) -> Optional[str]:
        """
        Resolve which user owns the Groww account so synced trades are
        attributed to them. Prefers settings.GROWW_OWNER_USERNAME; otherwise
        the first real (non-dev) user. Returns str(user.id) or None.
        """
        db = self._get_db()
        try:
            from core.models import User
            from config.settings import settings

            owner = None
            username = getattr(settings, "GROWW_OWNER_USERNAME", None)
            if username:
                owner = db.query(User).filter(User.username == username).first()
            if owner is None:
                owner = (
                    db.query(User)
                    .filter(User.username != "dev")
                    .order_by(User.created_at.asc())
                    .first()
                )
            if owner is None:
                owner = db.query(User).order_by(User.created_at.asc()).first()
            return str(owner.id) if owner else None
        except Exception as e:
            logger.warning("Could not resolve Groww owner user: %s", e)
            return None
        finally:
            db.close()


# ══════════════════════════════════════════════════════════════════
#  Singleton
# ══════════════════════════════════════════════════════════════════

_service: Optional[TradeSyncService] = None


def init_trade_sync_service(db_factory) -> Optional[TradeSyncService]:
    """
    Initialize the global TradeSyncService from .env credentials.
    Returns None if Groww is not configured.
    """
    global _service

    if not settings.GROWW_API_KEY:
        logger.info("Groww not configured (GROWW_API_KEY missing) — trade sync unavailable")
        return None

    try:
        from ingestion.broker_adapters.groww import GrowwClient

        client = GrowwClient(
            api_key=settings.GROWW_API_KEY,
            api_secret=settings.GROWW_API_SECRET,
            access_token=settings.GROWW_ACCESS_TOKEN,
        )

        _service = TradeSyncService(
            groww_client=client,
            db_factory=db_factory,
        )
        logger.info("TradeSyncService initialized (Groww)")
        return _service

    except Exception as e:
        logger.warning("TradeSyncService init failed: %s", e)
        return None


def get_trade_sync_service() -> Optional[TradeSyncService]:
    """Get the global TradeSyncService (None if not initialized)."""
    return _service
