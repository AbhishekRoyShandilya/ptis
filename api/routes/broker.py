# api/routes/broker.py
"""
Broker integration endpoints — status, connection, portfolio, and market data.

3-broker architecture:
  Angel One  → Historical OHLCV candles + positions/orders/account
  Upstox     → Live market data (LTP, quotes, intraday)
  Groww      → Trade/order/position sync

Portfolio endpoints read from whichever broker is connected (Angel One, Zerodha, or Groww).
Market data endpoints expose historical candles (Angel One) and live quotes (Upstox).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from api.auth import get_current_user
from api.dependencies import get_current_trader_id
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/broker",
    tags=["broker"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/import-pnl")
async def import_pnl_statement(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    trader_id: str = Depends(get_current_trader_id),
) -> Dict[str, Any]:
    """Import a Groww F&O P&L statement (.xlsx / .csv) — the FULL trade history the intraday-only API
    can't provide. Parses the trade-level table (Scrip / Buy&Sell Date+Price / Realized P&L), writes each
    row as a closed trade (deduped). Entry/exit are date-level, so the eye replay is whole-day context."""
    import os
    import tempfile
    from ingestion.groww_pnl_importer import import_pnl_file
    from core.database import SessionLocal

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".xlsx"
    if suffix not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(400, "Upload a Groww P&L statement as .xlsx, .xls, or .csv")
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(data)
        path = tf.name
    db = SessionLocal()
    try:
        stats = import_pnl_file(path, trader_id, db)
    except Exception as e:
        logger.warning("P&L import failed: %s", e)
        raise HTTPException(422, f"Could not parse the statement: {e}")
    finally:
        db.close()
        try:
            os.unlink(path)
        except Exception:
            pass
    return {"ok": True, "file": file.filename, **stats}


# ── Helpers ───────────────────────────────────────────────────────

def _get_portfolio_client():
    """
    Return the best available broker adapter for portfolio operations
    (positions, orders, account). Prefers Angel One → Zerodha → Groww.
    Returns (client, broker_name) or raises 503.
    """
    from ingestion.market_data_service import get_market_data_service

    svc = get_market_data_service()

    # Angel One
    if svc and svc._angel:
        return svc._angel, "angel_one"

    # Zerodha (if wired)
    if svc and getattr(svc, "_zerodha", None):
        return svc._zerodha, "zerodha"

    # Groww via trade sync service
    try:
        from ingestion.trade_sync import get_trade_sync_service
        groww_svc = get_trade_sync_service()
        if groww_svc and groww_svc._groww:
            return groww_svc._groww, "groww"
    except Exception:
        pass

    raise HTTPException(
        status_code=503,
        detail="No broker connected — configure Angel One, Zerodha, or Groww credentials",
    )


def _get_market_svc():
    """Return MarketDataService or raise 503."""
    from ingestion.market_data_service import get_market_data_service
    svc = get_market_data_service()
    if not svc:
        raise HTTPException(503, "Market data service not initialized — check broker env vars")
    return svc


# ══════════════════════════════════════════════════════════════════
#  STATUS & CONNECTION TESTS
# ══════════════════════════════════════════════════════════════════

@router.get("/status")
def broker_status(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Returns configuration and connection status for all brokers."""
    from ingestion.market_data_service import get_market_data_service
    from ingestion.trade_sync import get_trade_sync_service

    svc = get_market_data_service()
    groww_svc = get_trade_sync_service()

    # Angel One (historical candles)
    angel_configured = bool(settings.ANGEL_API_KEY and settings.ANGEL_CLIENT_ID)
    angel_connected = svc is not None and svc._angel is not None

    # Upstox (live market data)
    upstox_configured = bool(settings.UPSTOX_ACCESS_TOKEN)
    upstox_connected = svc is not None and svc.has_live

    # Zerodha (optional)
    zerodha_configured = bool(settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN)
    zerodha_connected = svc is not None and getattr(svc, "_zerodha", None) is not None

    # Groww (trade tracking)
    groww_configured = bool(settings.GROWW_API_KEY)
    groww_connected = groww_svc is not None

    return {
        "angel_one": {
            "configured": angel_configured,
            "connected": angel_connected,
            "role": "Historical Data + Portfolio",
        },
        "upstox": {
            "configured": upstox_configured,
            "connected": upstox_connected,
            "role": "Live Market Data",
        },
        "zerodha": {
            "configured": zerodha_configured,
            "connected": zerodha_connected,
            "role": "Portfolio (optional)",
        },
        "groww": {
            "configured": groww_configured,
            "connected": groww_connected,
            "role": "Trade Tracking",
            "last_sync": groww_svc.status["last_sync"] if groww_svc else None,
            "sync_count": groww_svc.status["sync_count"] if groww_svc else 0,
        },
    }


@router.post("/test/angel")
def test_angel_one(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Test Angel One by fetching a few historical candles for RELIANCE."""
    svc = _get_market_svc()
    if not svc.has_historical:
        raise HTTPException(503, "Angel One not initialized — check ANGEL_* env vars")

    try:
        candles = svc._fetch_candles_raw(
            symbol="RELIANCE",
            interval="1d",
            from_date=date.today() - timedelta(days=5),
            to_date=date.today(),
        )
        if candles:
            last = candles[-1]
            return {
                "status": "connected",
                "test_symbol": "RELIANCE",
                "candles_fetched": len(candles),
                "last_close": last.get("close"),
                "last_date": str(last.get("timestamp", "")),
            }
        return {
            "status": "error",
            "detail": "No candles returned — session may have expired",
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Angel One test failed: {e}")


@router.post("/test/upstox")
def test_upstox(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Test Upstox by fetching RELIANCE LTP."""
    svc = _get_market_svc()
    if not svc.has_live:
        raise HTTPException(503, "Upstox not initialized — check UPSTOX_ACCESS_TOKEN env var")

    try:
        ltp = svc.get_ltp("RELIANCE")
        if ltp and ltp > 0:
            return {"status": "connected", "test_symbol": "RELIANCE", "ltp": ltp}
        return {"status": "error", "detail": "LTP returned None or 0 — token may have expired"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstox test failed: {e}")


@router.post("/test/groww")
def test_groww(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Test Groww connection by fetching account info."""
    from ingestion.trade_sync import get_trade_sync_service

    svc = get_trade_sync_service()
    if not svc:
        raise HTTPException(503, "Groww trade sync not initialized — check GROWW_* env vars")

    try:
        account = svc._groww.get_account_info()
        return {
            "status": "connected",
            "account": {
                "equity": account.get("equity", 0),
                "buying_power": account.get("buying_power", 0),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groww test failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  PORTFOLIO — Positions, Orders, Account
# ══════════════════════════════════════════════════════════════════

@router.get("/positions")
def get_positions(user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Fetch current open positions from the connected broker.
    Returns positions with symbol, quantity, avg_price, pnl, side, ltp.
    """
    client, broker_name = _get_portfolio_client()
    try:
        positions = client.get_positions()
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch positions from {broker_name}: {e}")

    # Enrich with live LTP from Upstox if available
    from ingestion.market_data_service import get_market_data_service
    svc = get_market_data_service()
    if svc and svc.has_live:
        for pos in positions:
            symbol = pos.get("symbol", "")
            if symbol and not pos.get("ltp"):
                try:
                    ltp = svc.get_ltp(symbol)
                    if ltp and ltp > 0:
                        pos["ltp"] = ltp
                except Exception:
                    pass

    return {
        "broker": broker_name,
        "count": len(positions),
        "positions": positions,
    }


@router.get("/orders")
def get_orders(
    status: str = Query("filled", description="Order status filter: filled, open, cancelled, rejected"),
    days: int = Query(1, ge=1, le=90, description="Lookback days"),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Fetch order history from the connected broker.
    Returns orders with symbol, side, qty, price, filled_at, status.
    """
    client, broker_name = _get_portfolio_client()
    from_date = date.today() - timedelta(days=days)
    try:
        orders = client.get_orders(from_date=from_date, status=status)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch orders from {broker_name}: {e}")

    # Serialize datetimes
    for order in orders:
        if isinstance(order.get("filled_at"), datetime):
            order["filled_at"] = order["filled_at"].isoformat()

    return {
        "broker": broker_name,
        "count": len(orders),
        "filter": {"status": status, "from_date": from_date.isoformat()},
        "orders": orders,
    }


@router.get("/account")
def get_account(user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Fetch account balance, margins, and buying power from the connected broker.
    """
    client, broker_name = _get_portfolio_client()
    try:
        account = client.get_account_info()
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch account info from {broker_name}: {e}")

    return {
        "broker": broker_name,
        **account,
    }


# ══════════════════════════════════════════════════════════════════
#  LIVE MARKET DATA — LTP, Quotes, Intraday
# ══════════════════════════════════════════════════════════════════

@router.get("/ltp/{symbol}")
def get_ltp(symbol: str, user=Depends(get_current_user)) -> Dict[str, Any]:
    """Get last traded price for a symbol. Prefers Upstox live, falls back to Angel One/Zerodha."""
    svc = _get_market_svc()

    ltp = svc.get_ltp(symbol.upper())
    if ltp is None:
        raise HTTPException(404, f"Could not fetch LTP for {symbol.upper()}")

    return {"symbol": symbol.upper(), "ltp": ltp}


@router.get("/ltp")
def get_ltp_batch(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. RELIANCE,TCS,INFY"),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Get LTP for multiple symbols at once."""
    svc = _get_market_svc()

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(400, "No valid symbols provided")
    if len(symbol_list) > 50:
        raise HTTPException(400, "Maximum 50 symbols per batch request")

    results = {}
    for sym in symbol_list:
        try:
            ltp = svc.get_ltp(sym)
            results[sym] = ltp
        except Exception:
            results[sym] = None

    return {"prices": results, "count": len(results)}


@router.get("/quote/{symbol}")
def get_quote(symbol: str, user=Depends(get_current_user)) -> Dict[str, Any]:
    """Get full market quote (LTP, OHLC, volume, bid/ask) for a symbol."""
    svc = _get_market_svc()

    quote = svc.get_market_quote(symbol.upper())
    return {"symbol": symbol.upper(), **quote}


@router.get("/intraday/{symbol}")
def get_intraday(
    symbol: str,
    interval: str = Query("5m", description="Candle interval: 1m, 5m, 15m, 30m, 1h"),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """Get today's intraday candles for a symbol via Upstox."""
    svc = _get_market_svc()
    if not svc.has_live:
        raise HTTPException(503, "Live market data not available — Upstox not configured")

    valid_intervals = {"1m", "5m", "15m", "30m", "1h"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {', '.join(sorted(valid_intervals))}")

    candles = svc.get_intraday_candles(symbol.upper(), interval)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "count": len(candles),
        "candles": candles,
    }


# ══════════════════════════════════════════════════════════════════
#  HISTORICAL CANDLES (Angel One)
# ══════════════════════════════════════════════════════════════════

@router.get("/candles/{symbol}")
def get_historical_candles(
    symbol: str,
    interval: str = Query("1d", description="Candle interval: 1m, 5m, 15m, 30m, 1h, 1d"),
    days: int = Query(30, ge=1, le=2000, description="Lookback days from today"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD), overrides days"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD), defaults to today"),
    user=Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV candles from Angel One.
    Used for charting, backtesting, and pattern analysis.
    """
    svc = _get_market_svc()
    if not svc.has_historical:
        raise HTTPException(503, "Historical data not available — Angel One not configured")

    # Parse dates
    try:
        end = date.fromisoformat(to_date) if to_date else date.today()
        start = date.fromisoformat(from_date) if from_date else end - timedelta(days=days)
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")

    if start > end:
        raise HTTPException(400, "from_date must be before to_date")

    candles = svc._fetch_candles_raw(
        symbol=symbol.upper(),
        interval=interval,
        from_date=start,
        to_date=end,
    )

    # Serialize timestamps
    for c in candles:
        if isinstance(c.get("timestamp"), datetime):
            c["timestamp"] = c["timestamp"].isoformat()

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "from_date": start.isoformat(),
        "to_date": end.isoformat(),
        "count": len(candles),
        "candles": candles,
    }


# ══════════════════════════════════════════════════════════════════
#  UPSTOX OAUTH — one-click token refresh
# ══════════════════════════════════════════════════════════════════

@router.get("/upstox/login-url")
def upstox_login_url(user=Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns the Upstox OAuth login URL.
    Frontend opens this URL in a new tab/window.
    After login Upstox redirects to /api/v1/broker/upstox/callback.
    """
    if not settings.UPSTOX_API_KEY:
        raise HTTPException(400, "UPSTOX_API_KEY not configured in .env")

    url = (
        "https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={settings.UPSTOX_API_KEY}"
        f"&redirect_uri={settings.UPSTOX_REDIRECT_URI}"
    )
    return {"login_url": url, "redirect_uri": settings.UPSTOX_REDIRECT_URI}


@router.get("/upstox/callback", response_class=HTMLResponse, include_in_schema=False)
def upstox_callback(code: str = Query(..., description="OAuth code from Upstox")) -> HTMLResponse:
    """
    OAuth callback — exchanges the code for an access token, saves it,
    hot-reloads the running MarketDataService, then closes the browser tab.
    No auth required — this is called by Upstox's redirect.
    """
    if not settings.UPSTOX_API_KEY or not settings.UPSTOX_API_SECRET:
        return HTMLResponse(_oauth_result_page(
            success=False,
            message="UPSTOX_API_KEY and UPSTOX_API_SECRET must both be set in .env"
        ))

    try:
        import httpx
        resp = httpx.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code": code,
                "client_id": settings.UPSTOX_API_KEY,
                "client_secret": settings.UPSTOX_API_SECRET,
                "redirect_uri": settings.UPSTOX_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        resp.raise_for_status()
        token_data = resp.json()
        access_token = token_data.get("access_token")

        if not access_token:
            return HTMLResponse(_oauth_result_page(
                success=False,
                message=f"Upstox did not return an access token: {token_data}"
            ))

        # Persist token
        from ingestion.token_store import save_upstox_token
        save_upstox_token(access_token)

        # Hot-reload running service
        from ingestion.market_data_service import reload_upstox_token
        reload_upstox_token(access_token)

        logger.info("Upstox token refreshed via OAuth callback")
        return HTMLResponse(_oauth_result_page(success=True, message="Upstox connected!"))

    except Exception as e:
        logger.error("Upstox OAuth callback failed: %s", e)
        return HTMLResponse(_oauth_result_page(success=False, message=str(e)))


def _oauth_result_page(success: bool, message: str) -> str:
    """Simple HTML page shown after OAuth redirect — auto-closes the tab."""
    color = "#22c55e" if success else "#ef4444"
    icon = "✓" if success else "✗"
    return f"""<!DOCTYPE html>
<html>
<head><title>Upstox OAuth</title>
<style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
min-height:100vh;margin:0;background:#0f172a;color:#f1f5f9}}
.card{{text-align:center;padding:2rem;border-radius:12px;background:#1e293b;max-width:400px}}
.icon{{font-size:3rem;color:{color}}}
p{{color:#94a3b8;margin-top:.5rem}}</style></head>
<body><div class="card">
<div class="icon">{icon}</div>
<h2>{"Connected" if success else "Failed"}</h2>
<p>{message}</p>
<p style="font-size:.8rem;margin-top:1.5rem">{"This tab will close in 3 seconds..." if success else "Close this tab and try again."}</p>
</div>
{"<script>setTimeout(()=>window.close(),3000)</script>" if success else ""}
</body></html>"""


# ══════════════════════════════════════════════════════════════════
#  TRADE SYNC (Groww)
# ══════════════════════════════════════════════════════════════════

@router.post("/sync/trigger")
async def trigger_sync(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Manually trigger a Groww trade sync cycle."""
    from ingestion.trade_sync import get_trade_sync_service

    svc = get_trade_sync_service()
    if not svc:
        raise HTTPException(503, "Groww trade sync service not initialized")

    result = await svc.sync_now()
    return {"status": "completed", **result}


@router.get("/sync/status")
def sync_status(user=Depends(get_current_user)) -> Dict[str, Any]:
    """Get current trade sync service status."""
    from ingestion.trade_sync import get_trade_sync_service

    svc = get_trade_sync_service()
    if not svc:
        return {
            "running": False,
            "configured": bool(settings.GROWW_API_KEY),
            "detail": "Trade sync service not initialized",
        }

    return svc.status
