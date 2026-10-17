"""
intelligence.watchlist_scan — a BRAIN-backed watchlist scan (the same eye every face uses), so "where is
the big money move building?" is answered from cognition_service, not the old GEN-1 service or LLM blather.

The RECOGNITION/UNDERSTANDING split makes a whole-watchlist scan practical. The full eye read is ~10s/
symbol, so scanning it over every name is impossible; but cheap RECOGNITION (`intelligence.recognize`,
~0.09s/symbol via the BirthTracker) answers "is a move live here?" for ALL names first (peripheral vision)
→ the expensive understanding (`cognition_service.read`) runs ONLY on the few that light up (focal
attention). So we no longer bound to an arbitrary first-N; we look at everything, understand the movers.
Per-symbol read still CACHES (TTL) so repeated asks are instant.
"""
from __future__ import annotations

import time
import logging

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, dict]] = {}   # symbol -> (ts, result)
_TTL = 600                                   # 10 min — a scan is good for a while
# a move is "building/active" when the eye stages it as a candidate/money-move or the state is live/turning
_BUILDING_STAGE = {"CANDIDATE", "MONEY_MOVE"}
_BUILDING_STATE = {"TRANSITION", "ALIVE"}
_PRIO = {"MONEY_MOVE": 4, "CANDIDATE": 3, "ALIVE": 3, "TRANSITION": 2, "STORY_CHANGED": 1}


def scan_symbol(symbol: str, interval: str = "15m") -> dict:
    """One symbol's money-move read off the eye (cached)."""
    now = time.time()
    hit = _CACHE.get(symbol)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    from intelligence import cognition_service as cog
    r = cog.read(symbol, interval)
    out = {"symbol": symbol, "ok": bool(getattr(r, "ok", False)), "state": r.state, "stage": r.stage,
           "direction": r.direction, "bias": r.bias, "read": r.read, "price": r.price,
           "building": (r.stage in _BUILDING_STAGE) or (r.state in _BUILDING_STATE),
           "prio": _PRIO.get(r.stage, 0) or _PRIO.get(r.state, 0)}
    _CACHE[symbol] = (now, out)
    return out


def scan_watchlist(symbols: list[str], limit: int = 8, interval: str = "15m") -> list[dict]:
    """Peripheral vision → focal attention: cheap `recognize()` over EVERY symbol (not a bounded first-N),
    then the full eye read ONLY on the ones recognition flags as live/building — the recognition/
    understanding split. `limit` now caps the expensive UNDERSTANDING reads (of the movers), not the
    breadth of what we look at. Building moves ranked first; falls back to first-`limit` if none are live."""
    from intelligence.recognize import recognize_many
    recs = recognize_many(symbols, interval=interval)          # cheap sweep of the whole list
    movers = [r for r in recs if r.ok and r.worth_understanding]
    focus = [r.symbol for r in movers[:limit]] or symbols[:limit]   # understand the movers (or a floor)
    results = []
    for s in focus:
        try:
            results.append(scan_symbol(s, interval))
        except Exception:
            logger.warning("watchlist scan failed for %s", s, exc_info=True)
    results.sort(key=lambda r: (r.get("building", False), r.get("prio", 0)), reverse=True)
    return results


def render_scan(results: list[dict]) -> str:
    """A context block for the co-planner — the eye's read of each scanned symbol, building moves first."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return ""
    lines = ["## Watchlist Money-Move Scan (PTIS's own eye — building moves first; ground your answer in this)"]
    for r in ok:
        tag = "BUILDING" if r.get("building") else "quiet"
        lines.append(f"- {r['symbol']}: {r['state']} · stage {r['stage'] or '—'} · {r['direction'] or 'no-dir'} "
                     f"· {tag}" + (f" — {r['read']}" if r.get("read") else ""))
    return "\n".join(lines)
