"""
intelligence.trade_review — the EYE-grounded trade review (the Learning-Journal core; PRODUCT layer).

Re-wires the trade→learn loop onto the FROZEN-CORE eye (project_brain/09): instead of replaying raw
indicators (the pre-brain `behavioral/trade_replay`), it asks `cognition_service.read(symbol, as_of=…)`
— "what did the EYE see the moment you clicked?" — and turns that into a coach-language journal entry:
- ENTRY: was it WITH or AGAINST the eye's read, what stance the eye had, what its exit-trigger was.
- EXIT: did you get out while the move was still alive, as it rolled over, or after it turned.

Point-in-time by construction (the eye narrates only up to `as_of`). Cost ≈ 2 eye reads per trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


def _dir(d: str) -> str:
    d = (d or "").lower().strip()
    if d in ("long", "buy", "up", "b", "bull"):
        return "up"
    if d in ("short", "sell", "down", "s", "bear"):
        return "down"
    return ""


@dataclass
class TradeReview:
    symbol: str
    direction: str
    entry_time: str
    exit_time: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    move_atr: float = float("nan")        # realised move in ATR (eye's atr at entry), signed by direction
    ok: bool = True
    note: str = ""
    entry_read: dict = field(default_factory=dict)
    exit_read: dict = field(default_factory=dict)
    entry_verdict: str = ""
    exit_verdict: str = ""
    speak: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def review_trade(symbol: str, direction: str, entry_time, exit_time=None,
                 entry_price=None, exit_price=None, interval: str = "5m") -> TradeReview:
    """Review one trade through the eye. `entry_time`/`exit_time` = your click times."""
    from intelligence.cognition_service import read
    td = _dir(direction)
    tr = TradeReview(symbol=symbol.upper(), direction=td or direction,
                     entry_time=str(entry_time), exit_time=str(exit_time or ""),
                     entry_price=float(entry_price or 0), exit_price=float(exit_price or 0))

    try:
        er = read(symbol, interval=interval, as_of=entry_time)
    except Exception as e:
        tr.ok = False; tr.note = f"entry read failed: {e}"; return tr
    if not er.ok:
        tr.ok = False; tr.note = er.note or "no read at entry"; return tr
    tr.entry_read = er.to_dict()

    # ── ENTRY verdict: with/against the eye + its stance ──
    if td and er.direction:
        align = "WITH the eye's read" if er.direction == td else "AGAINST the eye's read"
    elif td:
        align = "into BALANCE — no clear move yet (anticipatory entry)"
    else:
        align = ""
    state = f"{er.state}{'/' + er.substate if er.substate else ''}"
    tr.entry_verdict = (f"{align} · eye stance: {er.bias or 'n/a'} ({state})").strip(" ·")

    # ── EXIT read + timing verdict ──
    if exit_time:
        try:
            xr = read(symbol, interval=interval, as_of=exit_time)
            if xr.ok:
                tr.exit_read = xr.to_dict()
                story = (xr.move_story or "").lower()
                st = (xr.state or "").upper()
                reversed_ = bool(xr.direction and td and xr.direction != td)
                done = ("EXHAUST" in st) or any(w in story for w in ("rolling over", "stalling", "rolled over"))
                alive = ("ALIVE" in st) or ("advancing" in story)
                if reversed_:
                    tr.exit_verdict = f"the move had TURNED ({xr.direction}) by your exit — you held past the turn"
                elif done:
                    tr.exit_verdict = "exited as the move was rolling over / exhausted — well-timed"
                elif alive:
                    tr.exit_verdict = "the move was STILL ALIVE at your exit — you may have left runway on the table"
                else:
                    tr.exit_verdict = f"at exit the eye read {xr.state}{'/' + xr.substate if xr.substate else ''}"
        except Exception:
            pass

    # ── magnitude in ATR (using the eye's atr at entry) ──
    if entry_price and exit_price and er.atr and er.atr == er.atr and er.atr > 0:
        sign = 1 if td == "up" else (-1 if td == "down" else 1)
        tr.move_atr = round(sign * (float(exit_price) - float(entry_price)) / er.atr, 2)

    tr.speak = _speak(tr, er)
    return tr


def _speak(tr: TradeReview, er) -> str:
    lines = [f"[{tr.symbol} · {tr.direction.upper()} · entry {tr.entry_time}]"]
    head = f"WHEN YOU ENTERED, the eye saw: {er.state}{'/' + er.substate if er.substate else ''}"
    if er.mechanism:
        head += f" — {er.mechanism}"
    lines.append(head)
    lines.append(f"  -> {tr.entry_verdict}")
    if er.invalidation:
        lines.append(f"  the eye's exit-trigger then: {er.invalidation}")
    if tr.exit_verdict:
        lines.append(f"AT YOUR EXIT: {tr.exit_verdict}")
    if tr.move_atr == tr.move_atr:
        lines.append(f"  trade moved {tr.move_atr:+} ATR")
    return "\n".join(lines)


def review_trade_model(trade, interval: str = "5m") -> TradeReview:
    """Convenience: review a stored `Trade` ORM row (symbol/direction/entry_time/exit_time/prices)."""
    direction = getattr(getattr(trade, "direction", None), "value", None) or str(getattr(trade, "direction", ""))
    return review_trade(
        symbol=trade.symbol, direction=direction,
        entry_time=trade.entry_time, exit_time=getattr(trade, "exit_time", None),
        entry_price=getattr(trade, "entry_price", None), exit_price=getattr(trade, "exit_price", None),
        interval=interval,
    )


def underlying_of(symbol: str) -> str:
    """Option contract → its underlying for the eye (which reads underlyings, not contracts):
    INFY26JUN1020CE → INFY · SENSEX2661174300PE → SENSEX. Equity symbols pass through."""
    import re
    m = re.match(r"^([A-Z&\-]+?)\d", (symbol or "").upper())
    return m.group(1) if m else (symbol or "").upper()


def _opt_type(symbol: str) -> str:
    s = (symbol or "").upper()
    return "CE" if s.endswith("CE") else ("PE" if s.endswith("PE") else "")


def effective_underlying_direction(symbol: str, trade_direction: str) -> str:
    """The bet on the UNDERLYING: long call = bullish (up); long put = bearish (down)."""
    ot, td = _opt_type(symbol), _dir(trade_direction)
    if ot == "CE":
        return "up" if td == "up" else "down"
    if ot == "PE":
        return "down" if td == "up" else "up"
    return td


def memory_recall(tags, direction: str) -> dict | None:
    """How did setups with this recognition signature resolve HISTORICALLY (the harvested library)? — the
    grounded edge a generic P&L analyzer can never produce. Coarsens the signature until the sample is usable."""
    tags = [t for t in (tags or [])]
    if not tags or not direction:
        return None
    from intelligence.strategy_backtest import Strategy, library_stats
    for sig in (tags, tags[:4], tags[:3], tags[:2]):
        st = library_stats(Strategy("(setup)", direction, set(sig), set())).to_dict()
        if st.get("n", 0) >= 8:
            return {"tags": sig, **st}
    st = library_stats(Strategy("(setup)", direction, set(tags[:2]), set())).to_dict()
    return {"tags": tags[:2], **st} if st.get("n") else None


def review_for_coplan(trades, max_eye: int = 2, interval: str = "15m") -> str:
    """A co-planner context block: the FACTS of each recent trade (always) + the EYE's read of the
    UNDERLYING at entry for the most recent few (where the lake has data) — for grounded mistake-finding."""
    if not trades:
        return ""
    lines = ["## Your Recent Trades — PTIS eye-grounded review (the WHY behind each, from what the MARKET",
             "## was actually doing at your entry/exit — this is what a generic P&L analyzer cannot see):"]
    for i, t in enumerate(trades):
        sym = getattr(t, "symbol", "?")
        d = getattr(getattr(t, "direction", None), "value", None) or str(getattr(t, "direction", ""))
        pnl = getattr(t, "pnl", None)
        ep, xp = getattr(t, "entry_price", None), getattr(t, "exit_price", None)
        et, xt = getattr(t, "entry_time", None), getattr(t, "exit_time", None)
        und = underlying_of(sym)
        opt = _opt_type(sym)
        eff = effective_underlying_direction(sym, d) or d
        tag = f" [{opt} on {und} = a bet {eff.upper()} on {und}]" if opt else ""
        lines.append(f"- {sym}{tag} · entry {ep} @ {et} → exit {xp} @ {xt} · P&L {pnl}")
        # Cross-day close recovered from a broker POSITION has no real entry timestamp — reading the eye at
        # the estimated (today) entry would fabricate a WRONG "at entry" story. Report P&L honestly, skip it.
        if getattr(t, "source", "") == "groww_position_sync":
            lines.append("    (cross-day close — entry time isn't in the broker record, so the eye can't "
                         "read your entry moment; the P&L above is Groww's authoritative realised figure)")
            continue
        # FULL eye review of the UNDERLYING (entry + exit + timing) — the deep WHY, for the most recent few
        if i < max_eye:
            try:
                tr = review_trade(und, eff, et, xt, interval=interval)
                if tr.ok:
                    er, xr = tr.entry_read or {}, tr.exit_read or {}
                    lines.append(f"    AT ENTRY: {tr.entry_verdict}")
                    if er.get("mechanism"):
                        lines.append(f"      what {und} was doing: {er['mechanism']}")
                    if er.get("invalidation"):
                        lines.append(f"      the move's exit-trigger then: {er['invalidation']}")
                    if tr.exit_verdict:
                        lines.append(f"    AT EXIT: {tr.exit_verdict}")
                    ep_u, xp_u = er.get("price"), xr.get("price")
                    if isinstance(ep_u, (int, float)) and isinstance(xp_u, (int, float)) and ep_u == ep_u and xp_u == xp_u and ep_u:
                        mv = (xp_u - ep_u) / ep_u * 100 * (1 if eff == "up" else -1)
                        lines.append(f"      {und} moved {mv:+.1f}% in your bet's direction while you held it")
                    # VALIDATED-MEMORY recall — the grounded edge a generic analyzer cannot produce
                    mem = memory_recall(er.get("tags"), eff)
                    if mem and mem.get("n"):
                        fav = "FAVORED" if mem["ran_rate"] > mem["base_ran_rate"] else "were AGAINST"
                        lines.append(f"      MEMORY: setups like this ({', '.join(mem['tags'][:5])}) historically "
                                     f"resolved {round(100 * mem['ran_rate'])}% vs {round(100 * mem['base_ran_rate'])}% "
                                     f"base (n={mem['n']}) — the odds {fav} your {eff} bet [11-symbol library, provisional]")
                    elif (er.get("state") or "").upper() == "BALANCE" and not er.get("tags"):
                        lines.append("      MEMORY: no recognized money-move at entry (BALANCE, no signature) — "
                                     "you traded noise; there's no historical edge for a setup that isn't there")
                else:
                    lines.append(f"    (eye can't read {und} at {et} yet — {tr.note}; refresh the lake to unlock this)")
            except Exception:
                pass
    return "\n".join(lines)
