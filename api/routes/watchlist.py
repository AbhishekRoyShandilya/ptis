# api/routes/watchlist.py
"""
Watchlist API — master + strategy watchlist management.

Endpoints:
- CRUD for watchlists (master + strategy)
- Add/remove symbols to master watchlist
- Trigger scan → auto-distribute to strategy watchlists
- View strategy watchlist matches
- Continuous learning insights per profile
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from api.auth import get_current_user

from core.database import get_db_dependency as get_db
from core.models import StrategyProfile, Watchlist, WatchlistItem, TradeFeedback

# ── Bundle definitions ──────────────────────────────────────────────
_FNO_NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "NESTLEIND", "TITAN", "ULTRACEMCO",
    "WIPRO", "ADANIENT", "ONGC", "POWERGRID", "NTPC", "COALINDIA", "ADANIPORTS",
    "JSWSTEEL", "TATASTEEL", "TECHM", "BPCL", "DRREDDY", "CIPLA", "APOLLOHOSP",
    "DIVISLAB", "EICHERMOT", "INDUSINDBK", "GRASIM", "HINDALCO",
    "M&M", "TATACONSUM", "SBILIFE", "HDFCLIFE", "BAJAJFINSV", "BRITANNIA",
    "UPL", "HEROMOTOCO", "TATAMOTORS", "VEDL", "BAJAJ-AUTO",
]

_FNO_HIGH_VOLUME = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BAJFINANCE",
    "AXISBANK", "KOTAKBANK", "LT", "BHARTIARTL", "TATAMOTORS", "TATASTEEL",
    "JSWSTEEL", "ONGC", "BPCL", "NTPC", "POWERGRID", "HINDALCO", "M&M",
    "WIPRO", "HCLTECH", "SUNPHARMA", "DRREDDY", "INDUSINDBK", "VEDL",
    "ADANIENT", "ADANIPORTS", "CIPLA", "DIVISLAB",
]

# Stocks with historically strong post-breakout momentum on NSE F&O.
# These tend to trend hard after level breaks — high ADR, institutional interest.
_FNO_MOMENTUM_BREAKOUT = [
    "ANGELONE", "DMART", "TRENT", "DIXON", "TIINDIA",
    "POLYCAB", "CDSL", "MCX", "NAUKRI", "ZOMATO",
    "IRFC", "RVNL", "CAMS", "PVRINOX", "SUPREMEIND",
    "RADICO", "MFSL", "PERSISTENT", "COFORGE", "LTIMINDTREE",
    "GODREJPROP", "KAYNES", "BIKAJI", "HSCL", "JYOTHYLAB",
]

# Key underlying indices — FNO traders monitor these for market direction & regime.
# Spot index prices are used for F&O planning; not directly tradeable as stocks.
_KEY_INDICES = [
    "NIFTY",       # Nifty 50 — primary large-cap benchmark
    "BANKNIFTY",   # Bank Nifty — most liquid F&O index
    "FINNIFTY",    # Nifty Financial Services — bank+NBFC combined
    "SENSEX",      # BSE Sensex — cross-check with Nifty
    "INDIA VIX",   # Volatility index — fear gauge, sizing reference
    "MIDCPNIFTY",  # Nifty Midcap Select — mid-cap breadth
]

_BUNDLE_SYMBOLS = {
    "top50_fno": _FNO_NIFTY50,
    "high_volume_fno": _FNO_HIGH_VOLUME,
    "momentum_breakout": _FNO_MOMENTUM_BREAKOUT,
    "key_indices": _KEY_INDICES,
}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/watchlist", tags=["watchlists"], dependencies=[Depends(get_current_user)])


# ── Request/Response Models ──

class CreateWatchlistRequest(BaseModel):
    name: str
    description: Optional[str] = None
    watchlist_type: str = "master"  # master | strategy
    strategy_profile_id: Optional[str] = None


class AddSymbolsRequest(BaseModel):
    symbols: list[str]


class WatchlistResponse(BaseModel):
    id: str
    name: str
    watchlist_type: str
    strategy_profile_id: Optional[str] = None
    strategy_name: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    auto_scan: bool = True
    item_count: int = 0
    last_scanned_at: Optional[str] = None


class WatchlistItemResponse(BaseModel):
    id: str
    symbol: str
    added_by: str
    status: str
    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    last_analysis: Optional[dict] = None
    added_at: Optional[str] = None


# ── Watchlist CRUD ──

@router.post("", response_model=WatchlistResponse)
def create_watchlist(req: CreateWatchlistRequest, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Create a new watchlist."""
    wl = Watchlist(
        name=req.name,
        description=req.description,
        watchlist_type=req.watchlist_type,
        trader_id=user.id,
    )
    if req.strategy_profile_id:
        wl.strategy_profile_id = UUID(req.strategy_profile_id)

    db.add(wl)
    db.commit()
    db.refresh(wl)

    return _serialize_watchlist(wl)


@router.get("", response_model=list[WatchlistResponse])
def list_watchlists(
    watchlist_type: Optional[str] = None,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List all watchlists, optionally filtered by type."""
    q = db.query(Watchlist).filter(Watchlist.trader_id == user.id, Watchlist.is_active == True)
    if watchlist_type:
        q = q.filter(Watchlist.watchlist_type == watchlist_type)
    return [_serialize_watchlist(wl) for wl in q.all()]


@router.get("/bundles")
def list_bundles(db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """List available symbol bundles for quick-add to a master watchlist."""
    # smart_picks: live symbols from user's Smart Picks watchlist (scored by seeder)
    smart_items: list[str] = []
    smart_wl = db.query(Watchlist).filter(
        Watchlist.trader_id == user.id,
        Watchlist.name == "Smart Picks",
        Watchlist.watchlist_type == "strategy",
        Watchlist.is_active == True,
    ).first()
    if smart_wl:
        rows = (
            db.query(WatchlistItem)
            .filter(
                WatchlistItem.watchlist_id == smart_wl.id,
                WatchlistItem.status.in_(["matched", "active"]),
            )
            .order_by(WatchlistItem.match_score.desc().nullslast())
            .limit(15)
            .all()
        )
        smart_items = [r.symbol for r in rows]

    return [
        {
            "key": "top50_fno",
            "name": "NIFTY 50 FNO",
            "description": "50 most liquid NSE F&O stocks",
            "count": len(_FNO_NIFTY50),
            "preview": _FNO_NIFTY50[:6],
        },
        {
            "key": "high_volume_fno",
            "name": "High Volume F&O",
            "description": "Top 30 F&O stocks by daily trading volume",
            "count": len(_FNO_HIGH_VOLUME),
            "preview": _FNO_HIGH_VOLUME[:6],
        },
        {
            "key": "momentum_breakout",
            "name": "Momentum Breakout",
            "description": "NSE F&O stocks known for strong post-breakout moves — high ADR, trend hard after level breaks",
            "count": len(_FNO_MOMENTUM_BREAKOUT),
            "preview": _FNO_MOMENTUM_BREAKOUT[:6],
        },
        {
            "key": "key_indices",
            "name": "Key Indices",
            "description": "Nifty 50, Bank Nifty, Fin Nifty, Sensex, VIX — underlying prices FNO traders track for direction & regime",
            "count": len(_KEY_INDICES),
            "preview": _KEY_INDICES,
        },
        {
            "key": "smart_picks",
            "name": "Smart Picks",
            "description": "Top FnO stocks scored by volume + momentum + compression",
            "count": len(smart_items),
            "preview": smart_items[:6],
            "live": True,
        },
    ]


@router.post("/{watchlist_id}/add-bundle")
def add_bundle(
    watchlist_id: str,
    bundle: str,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Add a named bundle of symbols to a master watchlist.
    bundle: top50_fno | high_volume_fno | smart_picks
    """
    wl = db.query(Watchlist).filter(
        Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id
    ).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    if wl.watchlist_type != "master":
        raise HTTPException(400, "Bundles can only be added to master watchlists")

    # Smart Picks is OPT-IN and STANDALONE: selecting it RUNS its own scoring
    # (volume + momentum + compression) and populates the dedicated "Smart Picks"
    # watchlist. It is NOT merged into the master and NOT filtered by the user's
    # strategies — it scans on its own conditions. The system never runs this
    # unless the user explicitly adds it here.
    if bundle == "smart_picks":
        from automation.smart_watchlist_seeder import SmartWatchlistSeeder
        seeder = SmartWatchlistSeeder(db)
        try:
            from ingestion.market_data_service import get_market_data_service
            mds = get_market_data_service()
            if mds and mds.has_historical:
                seeder.set_fetcher(mds.fetch_candles_for_scanner)
        except Exception:
            pass
        result = seeder.seed(str(user.id))
        return {"bundle": "smart_picks", "mode": "scored", **(result or {})}

    # Dynamic universe bundles: High Volume + Momentum Breakout are RULES, not
    # frozen lists. Rank the FNO universe by the relevant factor (live, snapshot
    # at add-time) so the names reflect today's actual market — then add them to
    # the master and let the user's strategies pick the qualifiers.
    _DYNAMIC = {"high_volume_fno": "volume", "momentum_breakout": "momentum"}
    if bundle in _DYNAMIC:
        symbols = []
        try:
            from automation.smart_watchlist_seeder import SmartWatchlistSeeder
            from ingestion.market_data_service import get_market_data_service
            mds = get_market_data_service()
            if mds and mds.has_historical:
                seeder = SmartWatchlistSeeder(db)
                seeder.set_fetcher(mds.fetch_candles_for_scanner)
                symbols = seeder.rank_by_factor(_DYNAMIC[bundle], top_n=30)
        except Exception as e:
            logger.warning("Dynamic ranking for '%s' failed: %s", bundle, e)
        if not symbols:
            # Market data unavailable → fall back to the curated static list.
            symbols = _BUNDLE_SYMBOLS.get(bundle, [])

    # Static universe bundles (NIFTY 50, Key Indices) → plain stock fetchers,
    # added to the master and matched against strategies by the background scan.
    elif bundle in _BUNDLE_SYMBOLS:
        symbols = _BUNDLE_SYMBOLS[bundle]
    else:
        raise HTTPException(400, f"Unknown bundle '{bundle}'. Valid: top50_fno, high_volume_fno, momentum_breakout, key_indices, smart_picks")

    added, skipped = [], []
    for symbol in symbols:
        existing = db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == wl.id,
            WatchlistItem.symbol == symbol,
        ).first()
        if existing:
            if existing.status == "filtered_out":
                existing.status = "active"
                added.append(symbol)
            else:
                skipped.append(symbol)
        else:
            db.add(WatchlistItem(
                watchlist_id=wl.id,
                symbol=symbol,
                added_by=f"bundle:{bundle}",
                status="active",
            ))
            added.append(symbol)

    db.commit()

    if added and getattr(wl, "auto_scan", True):
        _trigger_background_scan(wl.id)

    return {"bundle": bundle, "added": len(added), "skipped": len(skipped), "symbols": added}


@router.get("/{watchlist_id}", response_model=WatchlistResponse)
def get_watchlist(watchlist_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Get a single watchlist."""
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    return _serialize_watchlist(wl)


@router.delete("/{watchlist_id}")
def delete_watchlist(watchlist_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Soft-delete a watchlist."""
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    wl.is_active = False
    db.commit()
    return {"status": "deleted"}


# ── Symbol Management ──

@router.get("/{watchlist_id}/items", response_model=list[WatchlistItemResponse])
def list_items(
    watchlist_id: str,
    status: Optional[str] = None,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List all items in a watchlist."""
    # Verify ownership via watchlist
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    q = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == UUID(watchlist_id)
    )
    if status:
        q = q.filter(WatchlistItem.status == status)
    else:
        q = q.filter(WatchlistItem.status.in_(["active", "matched"]))

    return [_serialize_item(item) for item in q.all()]


@router.post("/{watchlist_id}/items")
def add_symbols(
    watchlist_id: str,
    req: AddSymbolsRequest,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Add symbols to a watchlist."""
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")

    added = []
    skipped = []
    for symbol in req.symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue

        existing = db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == wl.id,
            WatchlistItem.symbol == symbol,
        ).first()

        if existing:
            if existing.status == "filtered_out":
                existing.status = "active"
                added.append(symbol)
            else:
                skipped.append(symbol)
        else:
            item = WatchlistItem(
                watchlist_id=wl.id,
                symbol=symbol,
                added_by="user",
                status="active",
            )
            db.add(item)
            added.append(symbol)

    db.commit()

    # Auto-trigger scan in background when new symbols are added to a master watchlist
    if added and wl.watchlist_type == "master" and getattr(wl, "auto_scan", True):
        _trigger_background_scan(wl.id)

    return {"added": added, "skipped": skipped}


def _trigger_background_scan(watchlist_id) -> None:
    """Fire-and-forget background thread that runs scan_and_distribute. NO-OP unless smart-pick/setup scanning is
    enabled (PTIS_SETUP_SCANNER=true) — OFF by default, user-actioned (the user explicitly rescans via the UI)."""
    import os
    if os.environ.get("PTIS_SETUP_SCANNER", "false").lower() != "true":
        logger.debug("Auto-scan skipped for watchlist %s (setup scanner off by default)", watchlist_id)
        return
    import threading
    from core.database import SessionLocal
    from strategy_learning.watchlist_analyzer import WatchlistAnalyzer

    def _run():
        _db = SessionLocal()
        try:
            analyzer = WatchlistAnalyzer(_db)
            # Inject live candle fetcher if market data service is available
            try:
                from ingestion.market_data_service import get_market_data_service
                mds = get_market_data_service()
                if mds and mds.has_historical:
                    analyzer.set_fetcher(mds.fetch_candles_for_scanner)
            except Exception:
                pass
            results = analyzer.scan_and_distribute(watchlist_id)
            logger.info("Auto-scan complete for watchlist %s: %s", watchlist_id, {
                k: v.get("total_matched", 0) for k, v in results.items()
                if isinstance(v, dict)
            })
        except Exception as e:
            logger.warning("Background watchlist scan failed: %s", e)
        finally:
            _db.close()

    t = threading.Thread(target=_run, daemon=True, name=f"wl-scan-{watchlist_id}")
    t.start()
    logger.info("Auto-scan triggered for watchlist %s", watchlist_id)


@router.delete("/{watchlist_id}/items/{symbol}")
def remove_symbol(
    watchlist_id: str,
    symbol: str,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Remove a symbol from a watchlist."""
    # Verify ownership
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    item = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == UUID(watchlist_id),
        WatchlistItem.symbol == symbol.upper(),
    ).first()

    if not item:
        raise HTTPException(404, "Symbol not found in watchlist")

    item.status = "archived"
    db.commit()
    return {"status": "removed", "symbol": symbol.upper()}


# ── Smart Seed ──

@router.post("/smart-seed")
def smart_seed(db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """
    Auto-populate the user's 'Smart Picks' watchlist with top FNO stocks
    scored by volume, momentum, and range compression.
    Safe to call repeatedly — idempotent.
    """
    from automation.smart_watchlist_seeder import SmartWatchlistSeeder

    seeder = SmartWatchlistSeeder(db)

    # Inject live candle fetcher if market data service is available
    try:
        from ingestion.market_data_service import get_market_data_service
        mds = get_market_data_service()
        if mds and mds.has_historical:
            seeder.set_fetcher(mds.fetch_candles_for_scanner)
    except Exception:
        pass

    result = seeder.seed(str(user.id))
    return result


# ── Scan & Distribute ──

@router.post("/{watchlist_id}/scan")
def scan_watchlist(watchlist_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Scan a master watchlist and distribute stocks to strategy watchlists."""
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")
    if wl.watchlist_type != "master":
        raise HTTPException(400, "Can only scan master watchlists")

    from strategy_learning.watchlist_analyzer import WatchlistAnalyzer
    analyzer = WatchlistAnalyzer(db)
    results = analyzer.scan_and_distribute(wl.id)
    return results


@router.get("/{watchlist_id}/filter")
def filter_by_strategy(
    watchlist_id: str,
    strategy_name: Optional[str] = None,
    min_score: float = 0.0,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Filter watchlist symbols by strategy match.

    Pass ?strategy_name=compression_breakout to get only symbols that match
    that strategy, sorted by match_score descending.
    Pass ?min_score=0.6 to filter by minimum match confidence.

    Returns a flat list of symbols with their match scores and reasons —
    useful for building a focused trade shortlist from a broad watchlist.
    """
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")

    # Find all strategy watchlists for this user
    strategy_wls = db.query(Watchlist).filter(
        Watchlist.trader_id == user.id,
        Watchlist.watchlist_type == "strategy",
        Watchlist.is_active == True,
    ).all()

    if strategy_name:
        # Filter to watchlists whose profile name or strategy_profile.strategy_name matches
        strategy_wls = [
            swl for swl in strategy_wls
            if (swl.strategy_profile and (
                (swl.strategy_profile.name or "").lower() == strategy_name.lower()
                or strategy_name.lower() in (swl.strategy_profile.name or "").lower()
            )) or strategy_name.lower() in swl.name.lower()
        ]

    if not strategy_wls:
        return {
            "watchlist_id": watchlist_id,
            "strategy_name": strategy_name,
            "results": [],
            "message": "No strategy watchlists found" + (f" matching '{strategy_name}'" if strategy_name else ""),
        }

    # Collect matched symbols across all matching strategy watchlists
    seen: dict[str, dict] = {}
    for swl in strategy_wls:
        profile_name = swl.strategy_profile.name if swl.strategy_profile else swl.name
        matched = db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == swl.id,
            WatchlistItem.status == "matched",
            WatchlistItem.match_score >= min_score,
        ).order_by(WatchlistItem.match_score.desc().nullslast()).all()

        for item in matched:
            if item.symbol not in seen or (item.match_score or 0) > (seen[item.symbol]["match_score"] or 0):
                seen[item.symbol] = {
                    "symbol": item.symbol,
                    "match_score": item.match_score,
                    "match_reason": item.match_reason,
                    "strategy": profile_name,
                    "last_analysis": item.last_analysis,
                }

    results = sorted(seen.values(), key=lambda x: x["match_score"] or 0, reverse=True)
    return {
        "watchlist_id": watchlist_id,
        "strategy_name": strategy_name,
        "min_score": min_score,
        "count": len(results),
        "results": results,
    }


@router.get("/{watchlist_id}/strategy-matches")
def get_strategy_matches(watchlist_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Get all strategy watchlist matches for a master watchlist's symbols."""
    wl = db.query(Watchlist).filter(Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")

    # Get all symbols from this watchlist
    items = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == wl.id,
        WatchlistItem.status.in_(["active", "matched"]),
    ).all()

    symbols = [item.symbol for item in items]

    # Find strategy watchlists that contain these symbols
    strategy_wls = db.query(Watchlist).filter(
        Watchlist.trader_id == user.id,
        Watchlist.watchlist_type == "strategy",
        Watchlist.is_active == True,
    ).all()

    matches = {}
    for swl in strategy_wls:
        profile = swl.strategy_profile
        matched_items = db.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == swl.id,
            WatchlistItem.symbol.in_(symbols),
            WatchlistItem.status == "matched",
        ).all()

        if matched_items:
            matches[profile.name if profile else swl.name] = [
                {
                    "symbol": mi.symbol,
                    "match_score": mi.match_score,
                    "match_reason": mi.match_reason,
                    "analysis": mi.last_analysis,
                }
                for mi in matched_items
            ]

    return matches


# ── Live Prices for Watchlist Items ──

@router.get("/{watchlist_id}/prices")
def get_watchlist_prices(
    watchlist_id: str,
    db: DBSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Get live prices (LTP + change) for all active symbols in a watchlist.
    Uses Upstox for live data, falls back to Angel One LTP.
    """
    wl = db.query(Watchlist).filter(
        Watchlist.id == UUID(watchlist_id), Watchlist.trader_id == user.id
    ).first()
    if not wl:
        raise HTTPException(404, "Watchlist not found")

    items = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == wl.id,
        WatchlistItem.status.in_(["active", "matched"]),
    ).all()

    if not items:
        return {"watchlist_id": str(wl.id), "prices": [], "source": None}

    # Get market data service
    try:
        from ingestion.market_data_service import get_market_data_service
        svc = get_market_data_service()
    except Exception:
        svc = None

    source = None
    prices = []
    for item in items:
        entry = {
            "symbol": item.symbol,
            "ltp": None,
            "change": None,
            "change_pct": None,
            "match_score": item.match_score,
            "status": item.status,
        }

        if svc and svc.has_live:
            source = "upstox"
            try:
                quote = svc.get_market_quote(item.symbol)
                if quote.get("ltp"):
                    entry["ltp"] = quote["ltp"]
                    prev = quote.get("close") or quote.get("prev_close")
                    if prev and prev > 0:
                        entry["change"] = round(quote["ltp"] - prev, 2)
                        entry["change_pct"] = round((quote["ltp"] - prev) / prev * 100, 2)
            except Exception:
                pass

        prices.append(entry)

    return {
        "watchlist_id": str(wl.id),
        "watchlist_name": wl.name,
        "source": source,
        "count": len(prices),
        "prices": prices,
    }


# ── Continuous Learning Insights ──

@router.get("/insights/{profile_id}")
def get_profile_insights(profile_id: str, db: DBSession = Depends(get_db), user=Depends(get_current_user)):
    """Get continuous learning insights for a strategy profile."""
    from strategy_learning.continuous_learner import ContinuousLearner

    profile = db.query(StrategyProfile).filter(StrategyProfile.id == UUID(profile_id), StrategyProfile.trader_id == user.id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")

    learner = ContinuousLearner(db_factory=lambda: iter([db]))
    return learner.get_profile_insights(db, UUID(profile_id))


# ── Serializers ──

def _serialize_watchlist(wl: Watchlist) -> dict:
    profile_name = None
    if wl.strategy_profile:
        profile_name = wl.strategy_profile.name

    return {
        "id": str(wl.id),
        "name": wl.name,
        "watchlist_type": wl.watchlist_type,
        "strategy_profile_id": str(wl.strategy_profile_id) if wl.strategy_profile_id else None,
        "strategy_name": profile_name,
        "description": wl.description,
        "is_active": wl.is_active,
        "auto_scan": wl.auto_scan,
        "item_count": len([i for i in wl.items if i.status in ("active", "matched")]),
        "last_scanned_at": str(wl.last_scanned_at) if wl.last_scanned_at else None,
    }


def _serialize_item(item: WatchlistItem) -> dict:
    return {
        "id": str(item.id),
        "symbol": item.symbol,
        "added_by": item.added_by,
        "status": item.status,
        "match_score": item.match_score,
        "match_reason": item.match_reason,
        "last_analysis": item.last_analysis,
        "added_at": str(item.added_at) if item.added_at else None,
    }
