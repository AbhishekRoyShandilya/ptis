"""
scripts/seed_test_data.py
─────────────────────────
Seeds the ptis_test database with realistic synthetic data.

Creates:
  - 1 strategy profile (compression_breakout)
  - 45 trades over 60 days (realistic win/loss mix)
  - 1 watchlist with 5 symbols

Run after setup_test_env.py:
  PTIS_ENV=test python scripts/seed_test_data.py

Re-runnable: wipes existing test data and re-seeds fresh.
"""

import os
import sys
import random
import uuid
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

os.environ["PTIS_ENV"] = "test"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)  # reproducible output

SYMBOLS = ["RELIANCE", "MPHASIS", "INFY", "HDFCBANK", "TATAMOTORS", "WIPRO", "SBIN", "AXISBANK"]

WATCHLIST_SYMBOLS = ["RELIANCE", "MPHASIS", "INFY", "HDFCBANK", "TATAMOTORS"]


def wipe_test_data(db, trader_id):
    """Remove all seeded data for this trader so re-runs are clean."""
    from core.models import Trade, StrategyProfile, Watchlist
    from sqlalchemy import text

    trader_uuid = uuid.UUID(trader_id) if isinstance(trader_id, str) else trader_id

    db.query(Trade).filter(Trade.trader_id == trader_uuid).delete(synchronize_session=False)
    db.query(StrategyProfile).filter(StrategyProfile.trader_id == trader_uuid).delete(synchronize_session=False)
    db.query(Watchlist).filter(Watchlist.trader_id == trader_uuid).delete(synchronize_session=False)
    db.commit()
    print("  Previous test data wiped ✓")


def seed_strategy_profile(db, trader_id):
    """Create a strategy profile (StrategyProfile tracks learned strategies)."""
    from core.models import StrategyProfile

    trader_uuid = uuid.UUID(trader_id) if isinstance(trader_id, str) else trader_id

    profile = StrategyProfile(
        id=uuid.uuid4(),
        trader_id=trader_uuid,
        name="Compression Breakout",
        short_code="CMPBR01",
        status="active",
        narrative=(
            "Trades breakouts from 3-5 candle compression phases. "
            "Best in trending regimes (TRENDING, BREAKOUT). "
            "Uses volume surge as confirmation. SL below compression low."
        ),
        entry_criteria=[
            {"criterion": "3-5 narrow candles (compression)", "confidence": 0.85},
            {"criterion": "Volume surge on breakout candle >1.5x avg", "confidence": 0.78},
            {"criterion": "Price closes above resistance", "confidence": 0.90},
        ],
        exit_rules={
            "stop_method": "below_compression_low",
            "typical_stop_atr": 1.2,
            "target_patterns": ["measured_move", "prior_resistance"],
        },
        risk_parameters={
            "typical_risk_pct": 1.0,
            "typical_rr": 2.0,
        },
        regime_preferences={
            "preferred": ["TRENDING", "BREAKOUT"],
            "avoid": ["RANGING", "HIGH_VOLATILITY"],
        },
        confidence_scores={
            "entry": 0.85,
            "exit": 0.65,
            "stop": 0.80,
            "overall": 0.77,
        },
        examples_count=12,
        sessions_count=3,
        tags=["breakout", "momentum", "intraday"],
    )
    db.add(profile)
    db.commit()
    print(f"  Strategy profile seeded: {profile.name} ({profile.short_code}) ✓")
    return profile


def seed_trades(db, trader_id):
    """Create 45 synthetic trades over the last 60 calendar days."""
    from core.models import Trade, TradeDirection, TradeStatus

    trader_uuid = uuid.UUID(trader_id) if isinstance(trader_id, str) else trader_id

    # Realistic per-symbol price ranges (approximate NSE prices)
    price_ranges = {
        "RELIANCE":  (2800, 3100),
        "MPHASIS":   (2200, 2500),
        "INFY":      (1700, 1900),
        "HDFCBANK":  (1600, 1800),
        "TATAMOTORS":(950,  1100),
        "WIPRO":     (500,  580),
        "SBIN":      (820,  950),
        "AXISBANK":  (1100, 1250),
    }

    trades = []
    today = date.today()
    trade_date = today - timedelta(days=60)
    trade_num = 0

    while trade_date <= today and trade_num < 45:
        if trade_date.weekday() >= 5:   # skip weekends
            trade_date += timedelta(days=1)
            continue

        daily_count = random.choices([0, 1, 2], weights=[0.30, 0.50, 0.20])[0]

        for _ in range(daily_count):
            if trade_num >= 45:
                break

            symbol = random.choice(SYMBOLS)
            lo, hi = price_ranges[symbol]
            entry_price = round(random.uniform(lo, hi), 2)

            direction = random.choice([TradeDirection.LONG, TradeDirection.SHORT])
            is_win = random.random() < 0.55   # ~55% win rate

            # Risk: 0.8–1.8% of entry price as SL distance
            sl_pct = random.uniform(0.008, 0.018)
            sl_dist = entry_price * sl_pct
            position_size = round(max(1, 15000 / (entry_price * sl_pct * 100)), 0)

            if direction == TradeDirection.LONG:
                sl_price = round(entry_price - sl_dist, 2)
                exit_price = (
                    round(entry_price + sl_dist * random.uniform(1.5, 3.5), 2)
                    if is_win else
                    round(entry_price - sl_dist * random.uniform(0.8, 1.1), 2)
                )
            else:
                sl_price = round(entry_price + sl_dist, 2)
                exit_price = (
                    round(entry_price - sl_dist * random.uniform(1.5, 3.5), 2)
                    if is_win else
                    round(entry_price + sl_dist * random.uniform(0.8, 1.1), 2)
                )

            sign = 1 if direction == TradeDirection.LONG else -1
            pnl = round((exit_price - entry_price) * position_size * sign, 2)
            r_multiple = round(pnl / (sl_dist * position_size), 2) if sl_dist * position_size else 0

            entry_dt = datetime(
                trade_date.year, trade_date.month, trade_date.day,
                random.randint(9, 14), random.randint(15, 59),
                tzinfo=timezone.utc,
            )
            exit_dt = entry_dt + timedelta(minutes=random.randint(15, 240))

            trade = Trade(
                id=uuid.uuid4(),
                trader_id=trader_uuid,
                symbol=symbol,
                direction=direction,
                status=TradeStatus.CLOSED,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=sl_price,
                position_size=position_size,
                pnl=pnl,
                r_multiple=r_multiple,
                risk_pct=round(sl_pct * 100, 2),
                entry_time=entry_dt,
                exit_time=exit_dt,
                declared_strategy_id="CMPBR01",
                source="seed",
                notes=f"Seeded trade #{trade_num + 1}",
            )
            db.add(trade)
            trades.append(trade)
            trade_num += 1

        trade_date += timedelta(days=1)

    db.commit()

    wins = sum(1 for t in trades if (t.pnl or 0) > 0)
    total_pnl = sum(t.pnl or 0 for t in trades)
    print(
        f"  {len(trades)} trades seeded | "
        f"win rate: {wins/len(trades):.0%} | "
        f"total P&L: ₹{total_pnl:,.0f} ✓"
    )
    return trades


def seed_watchlist(db, trader_id):
    """Create a watchlist with 5 common FNO symbols."""
    from core.models import Watchlist, WatchlistItem

    trader_uuid = uuid.UUID(trader_id) if isinstance(trader_id, str) else trader_id

    wl = Watchlist(
        id=uuid.uuid4(),
        trader_id=trader_uuid,
        name="Test Watchlist",
        watchlist_type="master",
        is_active=True,
        auto_scan=True,
    )
    db.add(wl)
    db.flush()

    scores = [0.88, 0.75, 0.70, 0.65, 0.58]
    reasons = [
        "Compression + volume surge on breakout — strong setup",
        "3-candle compression near support — watch for breakout",
        "EMA trend intact, pullback to 20 EMA in progress",
        "Tight range for 4 candles — coiling before move",
        "Moderate compression — wait for volume confirmation",
    ]

    for i, sym in enumerate(WATCHLIST_SYMBOLS):
        item = WatchlistItem(
            id=uuid.uuid4(),
            watchlist_id=wl.id,
            symbol=sym,
            added_by="seed",
            status="matched",
            match_score=scores[i],
            match_reason=reasons[i],
        )
        db.add(item)

    db.commit()
    print(f"  Watchlist seeded: {', '.join(WATCHLIST_SYMBOLS)} ✓")
    return wl


def main():
    print("=" * 55)
    print("  PTIS Test Data Seeder")
    print(f"  Environment: {os.environ.get('PTIS_ENV', 'prod')} ← must be 'test'")
    print("=" * 55)

    if os.environ.get("PTIS_ENV") != "test":
        print("\nERROR: PTIS_ENV is not set to 'test'.")
        print("Run with:  PTIS_ENV=test python scripts/seed_test_data.py")
        print("Or on Windows:  set PTIS_ENV=test && python scripts/seed_test_data.py")
        sys.exit(1)

    from core.database import SessionLocal
    from core.models import User

    db = SessionLocal()

    # Verify we are on ptis_test, not ptis
    from config.settings import settings
    db_url = settings.DATABASE_URL
    if "ptis_test" not in db_url:
        print(f"\nERROR: DATABASE_URL does not point to ptis_test (got: {db_url})")
        print("Check that .env.test has DATABASE_URL=postgresql://ptis:ptis@localhost:5432/ptis_test")
        db.close()
        sys.exit(1)

    print(f"\nDatabase: {db_url}")
    print()

    # Get dev user
    trader = db.query(User).filter(User.username == "dev").first()
    if not trader:
        print("ERROR: 'dev' user not found.")
        print("Run first:  PTIS_ENV=test python scripts/setup_test_env.py")
        db.close()
        sys.exit(1)

    trader_id = str(trader.id)
    print(f"Trader: dev user (id={trader_id})")
    print()

    print("Wiping previous test data...")
    wipe_test_data(db, trader_id)

    print("Seeding strategy profile...")
    seed_strategy_profile(db, trader_id)

    print("Seeding 45 trades...")
    seed_trades(db, trader_id)

    print("Seeding watchlist...")
    seed_watchlist(db, trader_id)

    db.close()

    print()
    print("=" * 55)
    print("  Done! Test data is ready.")
    print()
    print("  Start test environment:")
    print("    Terminal 1:  start_test.bat")
    print("    Terminal 2:  start_test_frontend.bat")
    print("    Browser:     http://localhost:5174")
    print("=" * 55)


if __name__ == "__main__":
    main()
