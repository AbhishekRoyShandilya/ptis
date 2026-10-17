"""
intelligence.proactive_monitor — the PROACTIVE layer: surface value WITHOUT the user asking.

The heart of the unified-partner vision: watch the user's WATCHLIST + open POSITIONS through the brain
and raise an alert (→ an auto-generated partner chat) when something deserves attention:
  · a money move BUILDS on a watchlist symbol → "don't miss it" (a fresh recognized money move),
  · an open POSITION needs managing → the eye says it's against you / story changing / exhausting.

Each alert carries the full PARTNER message (coach voice + action), so the chat surface just posts it.
Efficient by design: the expensive cognition only fires per symbol once; the alert TRIGGER gates it.

NOTE (gated): proactivity is only *live* with live candles (Upstox token wired) — on the lake it replays
history. And the per-symbol narration is heavy; a streaming/incremental eye is the future speed-up. The
monitor is a background poller — callers run it on a loop and de-dup with `already_alerted`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ProactiveAlert:
    symbol: str
    kind: str                 # "money-move" / "exhaustion" (watchlist) / "position" (manage)
    urgency: str              # high / medium / low
    headline: str
    message: str              # the full partner speak (coach + action) to post into chat
    as_of: str = ""
    key: str = ""             # symbol|date|attention|stage — for de-dup across polls (transition = new key)
    attention: str = ""       # the recognition attention class that triggered this (event taxonomy)

    def to_dict(self) -> dict:
        return asdict(self)


def _worth_surfacing(r) -> bool:
    """Watchlist trigger: a money move you shouldn't miss — confirmed, or a strong forming transition."""
    if r.stage == "MONEY_MOVE":
        return True
    return r.state in ("TRANSITION", "ALIVE") and r.conviction in ("high", "medium")


def scan_watchlist(symbols, as_of=None, interval: str = "5m", already_alerted: set | None = None):
    """One pass over the watchlist → alerts for freshly-recognised money moves.

    RECOGNITION/UNDERSTANDING SPLIT (not poll-everything): cheap `recognize_many` over the WHOLE list
    first (peripheral vision, ~0.1-0.8s/symbol) → the expensive full `read` (understanding) runs ONLY on
    the names recognition flags as a live move. The recognition ATTENTION class goes in the de-dup key, so
    a TRANSITION (e.g. MONEY_MOVE → EXHAUSTION) is a NEW key and re-fires — edge-driven, not level-polled.
    `already_alerted` persists across passes to suppress repeats of the SAME state."""
    from intelligence.cognition_service import read
    from intelligence.recognize import recognize_many, ATTENTION_EXHAUSTION, ATTENTION_STRATEGY_MATCH
    from agents.partner import partner_from_read, PartnerContext
    # Load the user's taught strategies ONCE per pass so each alert can flag a validated-strategy
    # match ("matches your VALIDATED 'X' — N% become money moves") instead of a generic read.
    try:
        from intelligence.taught_strategies import load_strategies
        strategies = load_strategies()
    except Exception:
        strategies = []
    seen = already_alerted if already_alerted is not None else set()
    # PERIPHERAL VISION: cheap recognition over EVERYTHING → understand only the movers.
    recs = recognize_many(list(symbols), interval=interval, as_of=as_of)
    out = []
    for rec in recs:
        if not (rec.ok and rec.worth_understanding):
            continue
        sym = rec.symbol
        try:
            r = read(sym, interval=interval, as_of=as_of)
        except Exception:
            continue
        if not r.ok:
            continue
        pr = partner_from_read(r, PartnerContext(sym, strategies=strategies))
        # STRATEGY_MATCH producer: when one of the user's OWN validated taught strategies fires, that ALWAYS
        # surfaces (their edge, elevated) — bypasses the generic money-move gate. (Store may be empty until
        # strategies are taught; the pipe is ready — it lights up the moment a validated signature matches.)
        is_strategy = pr.strategy_match.startswith("matches your VALIDATED")
        exhausting = rec.attention == ATTENTION_EXHAUSTION
        if not is_strategy and not _worth_surfacing(r):
            continue
        if is_strategy:
            kind, attn, urg = "strategy-match", ATTENTION_STRATEGY_MATCH, "high"
            headline = pr.strategy_match.split("—")[0].strip()
        elif exhausting:
            kind, attn, urg = "exhaustion", ATTENTION_EXHAUSTION, "high"
            headline = r.read or r.state
        else:
            kind, attn, urg = "money-move", rec.attention, (r.conviction or "low")
            headline = r.read or r.state
        key = f"{sym}|{r.as_of[:10]}|{attn}|{r.stage}|{r.direction}"   # attention/strategy → transitions re-fire
        if key in seen:
            continue
        seen.add(key)
        out.append(ProactiveAlert(symbol=sym, kind=kind, urgency=urg, headline=headline,
                                  message=pr.speak, as_of=r.as_of, key=key, attention=attn))
    return out


def monitor_positions(positions, as_of=None, interval: str = "5m", already_alerted: set | None = None):
    """One pass over open positions → alerts when a position needs attention (against you / story
    changing / exhausting). `positions` = {symbol: Position}. Trader/strategies can be attached per
    symbol later; v1 reads the brain + the position direction."""
    from intelligence.cognition_service import read
    from intelligence.recognize import recognize, ATTENTION_POSITION_RISK, ATTENTION_EXHAUSTION
    from agents.partner import partner_from_read, PartnerContext
    try:
        from intelligence.taught_strategies import load_strategies
        strategies = load_strategies()
    except Exception:
        strategies = []
    seen = already_alerted if already_alerted is not None else set()
    out = []
    for sym, pos in positions.items():
        try:
            r = read(sym, interval=interval, as_of=as_of)
        except Exception:
            continue
        if not r.ok:
            continue
        pr = partner_from_read(r, PartnerContext(sym, position=pos, strategies=strategies))
        act = pr.action or ""
        # RECOGNITION OVERLAY: is the move you're RIDING exhausting? (cheap tracker read — the partner's
        # highest-value tap: "the move you're in is rolling over"). direction match = your side is fading.
        try:
            rec = recognize(sym, interval, as_of)
            riding_exhaustion = rec.attention == ATTENTION_EXHAUSTION and rec.direction == getattr(pos, "direction", "")
        except Exception:
            riding_exhaustion = False
        urgent = (act.startswith(("AGAINST YOU", "MANAGE")) or r.state == "EXHAUSTED"
                  or r.stage == "STORY_CHANGED" or riding_exhaustion)
        if not urgent:
            continue
        key = f"{sym}|{r.as_of[:10]}|{act[:12]}|{'EXH' if riding_exhaustion else ''}"
        if key in seen:
            continue
        seen.add(key)
        out.append(ProactiveAlert(
            symbol=sym, kind="position",
            urgency="high" if (act.startswith("AGAINST YOU") or riding_exhaustion) else "medium",
            headline=act.split(":")[0] or ("your move is exhausting" if riding_exhaustion else r.state),
            message=pr.speak, as_of=r.as_of, key=key, attention=ATTENTION_POSITION_RISK))
    return out


def live_open_positions() -> dict:
    """The user's OPEN broker positions → {underlying: Position(direction)} for `monitor_positions`.
    The eye reads UNDERLYINGS, so each option contract is mapped to its underlying + the effective
    directional bet (long CE = up, long PE = down). Empty if no broker/positions. Best-effort."""
    try:
        from ingestion.trade_sync import get_trade_sync_service
        from intelligence.trade_review import underlying_of, effective_underlying_direction
        from agents.partner import Position
    except Exception:
        return {}
    svc = get_trade_sync_service()
    client = getattr(svc, "_groww", None) if svc else None
    if client is None:
        return {}
    try:
        rows = client.get_positions() or []
    except Exception:
        return {}
    out: dict = {}
    for p in rows:
        sym = (p.get("symbol") or "").strip()
        qty = p.get("quantity") or 0
        if not sym or not qty:
            continue
        side = p.get("side") or ("long" if qty > 0 else "short")
        eff = effective_underlying_direction(sym, side)   # "up" / "down"
        if eff:
            out[underlying_of(sym)] = Position(direction=eff)
    return out


def render(alerts) -> str:
    if not alerts:
        return "(nothing worth surfacing)"
    L = []
    for a in alerts:
        L.append(f"🔔 [{a.urgency.upper()}] {a.symbol} — {a.kind}: {a.headline}")
        L.append("\n".join("    " + ln for ln in a.message.splitlines()))
    return "\n".join(L)
