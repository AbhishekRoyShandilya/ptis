"""
ingestion.groww_pnl_importer — import a Groww F&O P&L statement (Excel/CSV) → historical trades.

WHY: Groww's API order book is intraday-only, so 3 years of history can't be pulled live. But the P&L /
tax statement (downloadable from the Groww web/app, as .xlsx) has EVERY round-trip trade at TRADE LEVEL:
    Scrip Name | Quantity | Buy Date | Buy Price | Buy Value | Sell Date | Sell Price | Sell Value | Realized P&L
Each row is ONE complete trade (entry = buy leg, exit = sell leg), properly paired, with realised P&L.
This is the right source for the full backlog. Dates are DATE-level (no HH:MM), so the eye-replay at entry
is approximate (whole-day context), but direction / P&L / behavioral patterns are exact.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# The trade-level table's columns, however Groww labels them (case/space-insensitive contains-match).
_COLS = {
    "scrip": ("scrip name", "scrip", "symbol", "instrument"),
    "qty": ("quantity", "qty"),
    "buy_date": ("buy date",),
    "buy_price": ("buy price",),
    "sell_date": ("sell date",),
    "sell_price": ("sell price",),
    "pnl": ("realized p&l", "realised p&l", "realized pnl", "realised pnl", "pnl", "p&l"),
}


def parse_scrip(name: str):
    """'BANKNIFTY 13 APR 23 41500 Call' → (underlying, expiry_date, strike, opt_type 'CE'/'PE'). None on miss.
    Also tolerates the compact broker form (e.g. 'SENSEX2670276400PE') via underlying-only fallback."""
    if not name:
        return None
    m = re.match(r"^\s*([A-Z&\-]+)\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2})\s+([\d.]+)\s+(call|put|ce|pe)\s*$",
                 name.strip(), re.IGNORECASE)
    if m:
        und, dd, mon, yy, strike, typ = m.groups()
        mi = _MONTHS.get(mon.upper())
        if not mi:
            return None
        return (und.upper(), date(2000 + int(yy), mi, int(dd)), float(strike),
                "CE" if typ.upper() in ("CALL", "CE") else "PE")
    # compact fallback: leading letters = underlying, trailing CE/PE
    cm = re.match(r"^([A-Z&\-]+?)\d.*(CE|PE)$", (name or "").upper())
    if cm:
        return cm.group(1), None, None, cm.group(2)
    return None


def _to_date(v) -> Optional[date]:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in ("%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def _num(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def row_to_trade(row: dict, trader_id: str) -> Optional[dict]:
    """One trade-level row (already column-mapped) → a Trade kwargs dict. The buy leg is the ENTRY and the
    sell leg the EXIT (Groww lists bought-then-sold long positions). Direction = LONG; the UNDERLYING bet is
    up (Call) / down (Put) — carried in `underlying`/`opt_type` for the eye. Returns None if unusable."""
    scrip = str(row.get("scrip") or "").strip()
    parsed = parse_scrip(scrip)
    if not parsed:
        return None
    underlying, expiry, strike, opt_type = parsed
    qty = _num(row.get("qty"))
    buy_p = _num(row.get("buy_price"))
    sell_p = _num(row.get("sell_price"))
    pnl = _num(row.get("pnl"))
    bdate = _to_date(row.get("buy_date"))
    sdate = _to_date(row.get("sell_date"))
    if not qty or buy_p is None or bdate is None:
        return None
    # a 0 sell price / missing sell date = an expiry-worthless close; keep it (exit at 0 on expiry/sell date)
    entry_dt = datetime(bdate.year, bdate.month, bdate.day, 9, 15, tzinfo=timezone.utc)
    exit_dt = datetime((sdate or bdate).year, (sdate or bdate).month, (sdate or bdate).day, 15, 30,
                       tzinfo=timezone.utc)
    ext_id = f"growwpnl:{scrip}:{bdate}:{buy_p}:{qty}:{pnl}"   # stable per trade row for de-dup
    # symbol col is varchar(20) → store a COMPACT contract (underlying+strike+type); the eye's
    # underlying_of() still parses the underlying, and the full readable name goes in notes.
    compact = (f"{underlying}{int(strike)}{opt_type}" if strike else f"{underlying}{opt_type}")[:20]
    return {
        "symbol": compact,
        "underlying": underlying,
        "opt_type": opt_type,
        "direction": "LONG",                 # bought options
        "eff_direction": "up" if opt_type == "CE" else "down",   # bet on the underlying
        "entry_price": buy_p,
        "exit_price": sell_p if sell_p is not None else 0.0,
        "position_size": qty,
        "entry_time": entry_dt,
        "exit_time": exit_dt,
        "pnl": pnl,
        "external_id": ext_id,
        "trader_id": trader_id,
        "source": "groww_pnl_import",
        "broker": "groww",
        "notes": (f"{scrip} — imported from Groww P&L statement. "
                  "Entry/exit are DATE-level (no intraday time) — eye replay is whole-day context."),
    }


def _load_frames(path: str):
    """Return a list of raw (header-less) DataFrames — every sheet of an .xlsx, or the single CSV."""
    import pandas as pd
    if path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None, header=None)   # needs openpyxl
        return list(sheets.values())
    return [pd.read_csv(path, header=None)]


def _find_trade_table(df):
    """Locate the TRADE-LEVEL table's header row → (row_idx, {field: col_idx}). Distinguished from the
    contract-level table by having a 'Buy Date' column (contract-level is aggregated, no dates)."""
    for i in range(len(df)):
        cells = [str(c).strip().lower() for c in df.iloc[i].tolist()]
        joined = " | ".join(cells)
        if "buy date" in joined and "sell date" in joined:
            colmap: dict = {}
            for field, names in _COLS.items():
                for j, c in enumerate(cells):
                    if any(n == c for n in names) or any(n in c for n in names):
                        colmap[field] = j
                        break
            if all(k in colmap for k in ("scrip", "qty", "buy_date", "buy_price", "sell_price")):
                return i, colmap
    return None, None


def import_pnl_file(path: str, trader_id: str, db) -> dict:
    """Parse a Groww F&O P&L statement (.xlsx/.csv) and write its TRADE-LEVEL rows as closed Trades.
    Idempotent (dedup by external_id). Returns {imported, skipped, duplicates, errors}."""
    from core.models import Trade, TradeStatus
    from core.enums import TradeDirection
    import pandas as pd

    stats = {"imported": 0, "skipped": 0, "duplicates": 0, "errors": 0, "rows_seen": 0}
    frames = _load_frames(path)
    for df in frames:
        hdr, colmap = _find_trade_table(df)
        if hdr is None:
            continue
        for i in range(hdr + 1, len(df)):
            raw = df.iloc[i].tolist()
            scrip = str(raw[colmap["scrip"]]).strip() if colmap["scrip"] < len(raw) else ""
            if not scrip or scrip.lower() in ("nan", "total", "scrip name"):
                if scrip.lower() == "total":
                    break     # end of the table
                continue
            stats["rows_seen"] += 1
            row = {f: (raw[j] if j < len(raw) else None) for f, j in colmap.items()}
            t = row_to_trade(row, trader_id)
            if not t:
                stats["skipped"] += 1
                continue
            try:
                if db.query(Trade).filter(Trade.external_id == t["external_id"]).first():
                    stats["duplicates"] += 1
                    continue
                db.add(Trade(
                    symbol=t["symbol"], direction=TradeDirection.LONG, status=TradeStatus.CLOSED,
                    entry_price=t["entry_price"], exit_price=t["exit_price"],
                    position_size=t["position_size"], entry_time=t["entry_time"], exit_time=t["exit_time"],
                    pnl=t["pnl"], external_id=t["external_id"], trader_id=trader_id,
                    source=t["source"], broker=t["broker"], notes=t["notes"],
                ))
                db.commit()
                stats["imported"] += 1
            except Exception:
                db.rollback()
                stats["errors"] += 1
    return stats
