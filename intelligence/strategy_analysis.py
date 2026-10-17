"""
intelligence.strategy_analysis — the MONEY side of a taught strategy (the Analysis thread).

Teaching refines RECOGNITION (the eye: what the setup is). Analysis measures MONEY (AVT: does operating it
make money). They are deliberately separate concerns and separate conversations.

This module turns a taught signature into a REAL P&L ledger by replaying the matching sessions through
`avt_trader.simulate_session` — structural stop (the move's birth level) + asymmetric exit (hold while the
move is alive, exit on story-change / close), NOT a capped +1.5R. That honest exit is the whole reason the
edge shows up where capped-winner tests hid it. Because each replay runs the eye (~11s/session), a backtest
runs as a BACKGROUND job with progress; the result is stored on the strategy's performance ledger.
"""
from __future__ import annotations

import threading
import time
import logging

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict] = {}      # strategy name -> live job status
_LOCK = threading.Lock()


def aggregate_ledger(trades) -> dict:
    """Real-money summary from virtual trades' R-multiples: expectancy, profit factor, drawdown, etc.
    Accepts VirtualTrade objects OR their dict form (stored trades), so it works on persisted backtests."""
    rs = [(t["R"] if isinstance(t, dict) else t.R) for t in trades]
    rs = [r for r in rs if r == r]                    # drop NaN
    n = len(rs)
    if not n:
        return {"n": 0, "wins": 0, "win_rate": 0.0, "total_R": 0.0, "avg_R": 0.0, "avg_win_R": 0.0,
                "avg_loss_R": 0.0, "expectancy_R": 0.0, "profit_factor": None, "max_dd_R": 0.0,
                "best_R": 0.0, "worst_R": 0.0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    total = sum(rs)
    eq = peak = mdd = 0.0
    for r in rs:                                       # equity curve in R → max drawdown
        eq += r
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    pf = round(gross_win / gross_loss, 2) if gross_loss else (None if not wins else None)
    return {"n": n, "wins": len(wins), "win_rate": round(len(wins) / n, 3),
            "total_R": round(total, 2), "avg_R": round(total / n, 2),
            "avg_win_R": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss_R": round(sum(losses) / len(losses), 2) if losses else 0.0,
            "expectancy_R": round(total / n, 2), "profit_factor": pf,
            "max_dd_R": round(mdd, 2), "best_R": round(max(rs), 2), "worst_R": round(min(rs), 2)}


def _spread(items: list, k: int) -> list:
    """A representative sample of k items spread across the list (not just the first k)."""
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def matching_sessions(sig, max_sessions: int | None = None) -> list[tuple[str, str]]:
    """The (symbol, date) sessions where this signature has a recognised episode — the only sessions where
    the strategy would ever trade. Capped to a representative spread when max_sessions is set."""
    from intelligence.episodic import EpisodeStore
    eps = EpisodeStore().load()
    sess = sorted(set((e.symbol, str(e.date)[:10]) for e in eps
                      if getattr(e, "symbol", None) and getattr(e, "date", None) and sig.matches(e)))
    return _spread(sess, max_sessions) if max_sessions else sess


def _run(name: str, sig, interval: str, sessions: list[tuple[str, str]]) -> None:
    from intelligence.strategy_backtest import _EpShim
    from intelligence.avt_trader import simulate_session
    job = _JOBS[name]
    flt = lambda direction, tags: sig.matches(_EpShim(direction, tags))
    trades = []
    for i, (sym, d) in enumerate(sessions):
        if job.get("cancel"):
            break
        try:
            res = simulate_session(sym, interval, d, strategy_filter=flt)
            trades += res["trades"]
        except Exception:
            logger.debug("backtest session failed: %s %s", sym, d, exc_info=True)
        job["done"] = i + 1
        # STREAM a provisional ledger after every session so the UI shows a number building live
        # (a real read appears in ~15s instead of after the whole run), and the user can leave & return.
        partial = aggregate_ledger(trades)
        partial.update({"sessions_run": i + 1, "sessions_total": len(sessions),
                        "interval": interval, "partial": True})
        job["result"] = partial
    led = aggregate_ledger(trades)
    led.update({"sessions_run": job["done"], "sessions_total": len(sessions), "partial": False,
                "interval": interval, "ran_at": time.strftime("%Y-%m-%d %H:%M"),
                "trades": [t.to_dict() for t in trades]})   # persist trades → attribution/discoveries
    job["result"] = {k: v for k, v in led.items() if k != "trades"}   # keep live status light
    job["running"] = False
    try:
        from intelligence.taught_strategies import save_performance
        save_performance(name, led)
    except Exception:
        logger.warning("could not persist performance for %s", name, exc_info=True)


def start_backtest(name: str, direction: str, require=None, exclude=None,
                   interval: str = "5m", max_sessions: int = 60) -> dict:
    """Kick off a background P&L backtest for a strategy. Returns the live status immediately."""
    from intelligence.strategy_backtest import Strategy
    sig = Strategy(name or "(x)", direction, set(require or []), set(exclude or []))
    sessions = matching_sessions(sig, max_sessions)
    with _LOCK:
        cur = _JOBS.get(name)
        if cur and cur.get("running"):
            return _status(name)
        _JOBS[name] = {"running": True, "total": len(sessions), "done": 0,
                       "result": None, "started": time.time()}
    if not sessions:
        _JOBS[name].update({"running": False, "result": aggregate_ledger([])})
        return _status(name)
    threading.Thread(target=_run, args=(name, sig, interval, sessions), daemon=True).start()
    return _status(name)


def _status(name: str) -> dict:
    j = _JOBS.get(name)
    if j:
        elapsed = time.time() - j.get("started", time.time())
        done = j.get("done", 0)
        eta = round((j["total"] - done) * (elapsed / done)) if done and j["running"] else None
        return {"running": j["running"], "total": j["total"], "done": done,
                "result": j["result"], "eta_sec": eta}
    from intelligence.taught_strategies import load_session
    perf = (load_session(name) or {}).get("performance") if name else None
    return {"running": False, "total": 0, "done": 0, "result": perf, "eta_sec": None}


def backtest_status(name: str) -> dict:
    return _status(name)


# ── Attribution & discoveries: turn the trades into GROUNDED findings (the research-partner layer) ──
# Every finding is COMPUTED from the real trades (deterministic). The LLM may later PHRASE them, but never
# sources a fact — a fabricated insight is worse than a number. Each finding carries its evidence (n).

_DIMS = {
    "time of day": ["AT_OPEN", "PHASE_OPEN", "PHASE_MORNING", "PHASE_MIDDAY", "PHASE_AFTERNOON", "PHASE_LATE"],
    "room": ["ROOM_CLEAR", "ROOM_SOME", "ROOM_TIGHT", "ROOM_WALL"],
    "effort": ["STRONG_EFFORT", "WEAK_EFFORT"],
    "structure": ["STRUCT_HH_HL", "STRUCT_LH_LL", "STRUCT_RANGE"],
    "trend": ["WITH_TREND", "COUNTER_TREND", "NO_TREND"],
    "regime": ["REGIME_BULL", "REGIME_BEAR"],
    "vwap": ["ABOVE_VWAP", "BELOW_VWAP"],
}
_NICE = {"AT_OPEN": "at the open", "PHASE_OPEN": "the opening", "PHASE_MORNING": "the morning",
         "PHASE_MIDDAY": "midday", "PHASE_AFTERNOON": "the afternoon", "PHASE_LATE": "late session",
         "ROOM_CLEAR": "clear room", "ROOM_SOME": "some room", "ROOM_TIGHT": "tight room", "ROOM_WALL": "into a wall",
         "STRONG_EFFORT": "strong effort", "WEAK_EFFORT": "weak effort", "STRUCT_RANGE": "a tight base",
         "STRUCT_HH_HL": "an up-structure", "STRUCT_LH_LL": "a down-structure", "WITH_TREND": "with the trend",
         "COUNTER_TREND": "counter-trend", "REGIME_BULL": "a bull regime", "REGIME_BEAR": "a bear regime",
         "ABOVE_VWAP": "above VWAP", "BELOW_VWAP": "below VWAP"}


def _confidence(n: int) -> str:
    return "High" if n >= 25 else "Medium" if n >= 10 else "Low"


def _grp(rows: list, total_R: float, total_win_R: float) -> dict:
    rs = [t["R"] for t in rows if isinstance(t.get("R"), (int, float)) and t["R"] == t["R"]]
    n = len(rs)
    tot = sum(rs)
    win_R = sum(r for r in rs if r > 0)
    wins = sum(1 for r in rs if r > 0)
    return {"n": n, "total_R": round(tot, 2), "share": round(tot / total_R, 3) if total_R else 0.0,
            "profit_share": round(win_R / total_win_R, 3) if total_win_R else 0.0,
            "win_rate": round(wins / n, 3) if n else 0.0, "avg_R": round(tot / n, 2) if n else 0.0}


def attribute(trades: list) -> dict:
    """Group the real trades by each dimension → R-contribution / win-rate / n per bucket. Pure facts."""
    valid = [t for t in trades if isinstance(t.get("R"), (int, float)) and t["R"] == t["R"]]
    total_R = sum(t["R"] for t in valid) or 1e-9
    total_win_R = sum(t["R"] for t in valid if t["R"] > 0) or 1e-9
    dims = {}
    for dim, dim_tags in _DIMS.items():
        groups = {}
        for t in valid:
            tags = set(t.get("entry_tags") or [])
            hit = next((g for g in dim_tags if g in tags), None)
            if hit:
                groups.setdefault(hit, []).append(t)
        if groups:
            dims[dim] = {b: _grp(rows, total_R, total_win_R) for b, rows in groups.items()}
    by_sym, by_exit = {}, {}
    for t in valid:
        by_sym.setdefault(t.get("symbol", "?"), []).append(t)
        by_exit.setdefault(t.get("exit_reason", "?"), []).append(t)
    dims["symbol"] = {s: _grp(rows, total_R, total_win_R) for s, rows in by_sym.items()}
    dims["exit"] = {e: _grp(rows, total_R, total_win_R) for e, rows in by_exit.items()}
    return {"total_R": round(total_R, 2), "n": len(valid), "dims": dims}


def discoveries(trades: list, top: int = 4, constant_tags=None) -> list[dict]:
    """Rank the strongest TRUE patterns into research observations — concentration (one bucket holds most
    R), contrast (one bucket far out-earns another), and where the losers cluster. Confidence = sample n.
    `constant_tags` = the strategy's own require/exclude tags: SKIPPED, because they are present in every
    trade by definition and would trivially 'explain 100%' — the insight is in the FREE dimensions."""
    constant = set(constant_tags or [])
    attr = attribute(trades)
    total_R, found = attr["total_R"], []
    for dim, buckets in attr["dims"].items():
        if dim in ("symbol", "exit"):
            continue
        meaningful = {b: g for b, g in buckets.items() if g["n"] >= 6 and b not in constant}
        if not meaningful:
            continue
        b, g = max(meaningful.items(), key=lambda kv: kv[1]["profit_share"])
        if len(buckets) == 1 and g["n"] == attr["n"]:
            # one bucket covering EVERY trade → not concentration, a blind-spot (untested elsewhere)
            found.append({"kind": "coverage", "dimension": dim, "bucket": b,
                          "headline": f"Every trade happened in {_NICE.get(b, b)} — untested elsewhere",
                          "detail": f"all {g['n']} trades · no data outside {_NICE.get(b, b)}",
                          "confidence": _confidence(g["n"]), "evidence_n": g["n"], "effect": 0.35})
        elif g["profit_share"] >= 0.45:
            # concentration — a single bucket holds a large share of the PROFIT (bounded, intuitive)
            found.append({"kind": "concentration", "dimension": dim, "bucket": b,
                          "headline": f"{_NICE.get(b, b)} generated {round(100 * g['profit_share'])}% of all profit",
                          "detail": f"{g['n']} trades · {g['avg_R']}R avg · {round(100 * g['win_rate'])}% win · {g['total_R']}R",
                          "confidence": _confidence(g["n"]), "evidence_n": g["n"], "effect": g["profit_share"]})
        # contrast — best vs worst avg_R within the dimension
        if len(meaningful) >= 2:
            best = max(meaningful.items(), key=lambda kv: kv[1]["avg_R"])
            worst = min(meaningful.items(), key=lambda kv: kv[1]["avg_R"])
            bw, ww = best[1]["avg_R"], worst[1]["avg_R"]
            if bw > 0 and ww < bw:
                ratio = (bw / ww) if ww > 0 else None
                if ratio is None or ratio >= 1.8:
                    rtxt = f"{ratio:.1f}×" if ratio else f"{bw}R vs {ww}R"
                    found.append({"kind": "contrast", "dimension": dim, "bucket": best[0], "vs": worst[0],
                                  "headline": f"{_NICE.get(best[0], best[0])} out-earns {_NICE.get(worst[0], worst[0])} by {rtxt}",
                                  "detail": f"{_NICE.get(best[0], best[0])}: {bw}R avg (n={best[1]['n']}) · {_NICE.get(worst[0], worst[0])}: {ww}R avg (n={worst[1]['n']})",
                                  "confidence": _confidence(min(best[1]["n"], worst[1]["n"])),
                                  "evidence_n": min(best[1]["n"], worst[1]["n"]),
                                  "effect": (ratio or 2.0) / 4})
    # where the losers cluster (drives "why is win-rate X?")
    losers = [t for t in trades if isinstance(t.get("R"), (int, float)) and t["R"] < 0]
    if losers:
        loss_R = sum(t["R"] for t in losers)
        clusters = []
        for dim, dim_tags in _DIMS.items():
            for b in dim_tags:
                if b in constant:
                    continue
                rows = [t for t in losers if b in set(t.get("entry_tags") or [])]
                if len(rows) >= 4:
                    s = sum(t["R"] for t in rows)
                    clusters.append((b, len(rows), round(s, 2), s / loss_R if loss_R else 0))
        clusters.sort(key=lambda c: c[3], reverse=True)
        if clusters:
            top2 = clusters[:2]
            found.append({"kind": "loser_cluster", "dimension": "losses",
                          "headline": f"Losses concentrate in {', '.join(_NICE.get(b, b) for b, *_ in top2)}",
                          "detail": " · ".join(f"{_NICE.get(b, b)}: {r}R over {nn} trades" for b, nn, r, _ in top2),
                          "confidence": _confidence(sum(c[1] for c in top2)),
                          "evidence_n": sum(c[1] for c in top2), "effect": abs(top2[0][3]),
                          "clusters": [{"bucket": b, "n": nn, "total_R": r} for b, nn, r, _ in top2]})
    # attach a SUGGESTED refinement (an applicable signature change) to each actionable finding
    for f in found:
        if f["kind"] == "concentration" and f.get("bucket"):
            f["refinement"] = {"op": "require", "tags": [f["bucket"]],
                               "label": f"Require {_NICE.get(f['bucket'], f['bucket'])}"}
        elif f["kind"] == "contrast" and f.get("vs"):
            f["refinement"] = {"op": "exclude", "tags": [f["vs"]],
                               "label": f"Exclude {_NICE.get(f['vs'], f['vs'])}"}
        elif f["kind"] == "loser_cluster":
            tags = [c["bucket"] for c in f.get("clusters", [])]
            f["refinement"] = {"op": "exclude", "tags": tags,
                               "label": f"Exclude {', '.join(_NICE.get(t, t) for t in tags)}"}
        # coverage = a blind-spot, not actionable (you can't add untested-regime trades)
    found.sort(key=lambda f: (f["confidence"] != "Low", f.get("effect", 0)), reverse=True)
    return found[:top]


def apply_refinement(name: str, op: str, tags: list, reason: str = "") -> dict:
    """Apply a finding as a versioned signature change (require/exclude a tag). Bumps the version, logs it
    in the hypothesis history as PROVISIONAL — a re-run then validates whether it actually holds (so a
    small-sample finding can't quietly overfit the strategy)."""
    from intelligence.taught_strategies import load_session, save_session, add_hypothesis, ALL_TAGS
    s = load_session(name)
    if not s:
        raise ValueError("strategy not found")
    tags = [t for t in (tags or []) if t in ALL_TAGS]
    if not tags:
        raise ValueError("no valid tags to apply")
    req, exc = set(s.get("require") or []), set(s.get("exclude") or [])
    if op == "require":
        req |= set(tags); exc -= set(tags)
    elif op == "exclude":
        exc |= set(tags); req -= set(tags)
    else:
        raise ValueError("op must be require or exclude")
    bp = s.get("performance") or {}
    baseline = {"total_R": bp.get("total_R"), "expectancy_R": bp.get("expectancy_R"), "n": bp.get("n")}
    save_session(name, direction=s.get("direction", "up"), require=sorted(req), exclude=sorted(exc),
                 description=s.get("description", ""))
    add_hypothesis(name, f"{op} {', '.join(tags)} — {reason}".strip(),
                   result=f"applied (provisional) · baseline {baseline['total_R']}R · re-run to validate",
                   op=op, tags=tags, baseline=baseline, status="pending")
    return {"require": sorted(req), "exclude": sorted(exc), "direction": s.get("direction", "up")}


def evaluate_refinement(name: str) -> dict | None:
    """After a refinement's re-run, compare the new ledger to the stored baseline and render an HONEST
    verdict — judged on EXPECTANCY (per-trade quality), because a 'require' refinement is more selective
    (fewer trades, lower total R, but each trade better), while an 'exclude' lifts total R too. Resolves
    the pending hypothesis. Returns None if nothing is pending or the new run isn't in yet."""
    from intelligence.taught_strategies import load_session, save_hypotheses
    s = load_session(name) or {}
    hyps = s.get("hypotheses", []) or []
    pending = next((h for h in reversed(hyps) if h.get("status") == "pending"), None)
    if not pending:
        return None
    cur = s.get("performance") or {}
    base = pending.get("baseline") or {}
    new_exp, base_exp = cur.get("expectancy_R"), base.get("expectancy_R")
    new_R, base_R, new_n, base_n = cur.get("total_R"), base.get("total_R"), cur.get("n"), base.get("n")
    if new_exp is None or base_exp is None or new_n is None:
        return None
    d_exp = round(new_exp - base_exp, 2)
    held = d_exp > 0
    thin = new_n < 10
    verdict = {"held": held, "thin": thin, "delta_exp": d_exp, "delta_R": round((new_R or 0) - (base_R or 0), 2),
               "new": {"total_R": new_R, "expectancy_R": new_exp, "n": new_n},
               "baseline": {"total_R": base_R, "expectancy_R": base_exp, "n": base_n},
               "label": pending.get("text", "")}
    pending["status"] = "held" if held else "didnt_hold"
    pending["result"] = (f"{'held ✓' if held else 'did not hold ✗'}: expectancy {base_exp}→{new_exp}R "
                         f"({d_exp:+}), total {base_R}→{new_R}R, n {base_n}→{new_n}"
                         + (" · thin sample" if thin else ""))
    save_hypotheses(name, hyps)
    return verdict


def discover_strategy(name: str) -> dict:
    """Grounded discoveries from a strategy's stored backtest trades — the research-partner opening."""
    from intelligence.taught_strategies import load_session
    sess = load_session(name) or {}
    perf = sess.get("performance") or {}
    trades = perf.get("trades") or []
    if not trades:
        return {"ready": False, "n": 0, "discoveries": [], "attribution": {}}
    constant = list((sess.get("require") or [])) + list((sess.get("exclude") or []))
    discs = discoveries(trades, constant_tags=constant)
    # PROJECT each refinement's money effect via a what-if on the existing trades (instant, propose-only;
    # the real re-run on Apply is what validates it) — so the user sees the best money bet up front.
    base_exp = aggregate_ledger(trades)["expectancy_R"]
    for d in discs:
        ref = d.get("refinement")
        if not ref:
            continue
        tset = set(ref["tags"])
        if ref["op"] == "exclude":
            kept = [t for t in trades if not (tset & set(t.get("entry_tags") or []))]
        else:  # require
            kept = [t for t in trades if tset.issubset(set(t.get("entry_tags") or []))]
        ref["proj_exp_delta"] = round(aggregate_ledger(kept)["expectancy_R"] - base_exp, 2) if kept else 0.0
        ref["n_after"] = len(kept)
    # surface the single best money refinement first
    discs.sort(key=lambda d: (d.get("refinement") or {}).get("proj_exp_delta", -99), reverse=True)
    return {"ready": True, "n": len(trades), "total_R": perf.get("total_R"),
            "discoveries": discs, "attribution": attribute(trades)}


def _sample_trade(t: dict) -> dict:
    tags = [x for x in (t.get("entry_tags") or []) if x in _NICE]
    return {"symbol": t.get("symbol"), "date": t.get("date"), "time": t.get("entry_ts"),
            "R": round(t.get("R", 0), 2), "outcome": "RAN" if t.get("R", 0) > 0 else "FAILED",
            "exit": t.get("exit_reason", ""), "context": ", ".join(_NICE.get(x, x) for x in tags[:3])}


def top_samples(trades: list, k: int = 3) -> dict:
    """A few real winning + losing trades with their context — evidence the user can see."""
    valid = [t for t in trades if isinstance(t.get("R"), (int, float)) and t["R"] == t["R"]]
    ws = sorted((t for t in valid if t["R"] > 0), key=lambda t: -t["R"])[:k]
    ls = sorted((t for t in valid if t["R"] <= 0), key=lambda t: t["R"])[:k]
    return {"winners": [_sample_trade(t) for t in ws], "losers": [_sample_trade(t) for t in ls]}


def _to_min(hhmm: str):
    try:
        h, m = str(hhmm).split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def parse_whatif(question: str) -> dict | None:
    """LLM PARSES a 'skip/keep these trades' hypothesis into a structured filter (the data then applies it,
    deterministically). Returns None if the question isn't a what-if."""
    from intelligence.teaching_service import _get_llm, _extract_json
    sysp = ("Decide if the trader is asking a WHAT-IF about skipping or keeping a subset of trades. Output "
            "STRICT JSON: {\"is_whatif\":bool,\"action\":\"exclude|only\",\"kind\":\"time|tag|symbol\","
            "\"start\":\"HH:MM\"|null,\"end\":\"HH:MM\"|null,\"tag\":str|null,\"symbol\":str|null}. "
            "Time window like '11 to 1pm' → 24h start/end (market 09:15-15:30, so 1pm=13:00). "
            "action=exclude for \"don't trade / skip / without / avoid\"; only for \"only / just\".")
    try:
        import asyncio
        txt = asyncio.run(_get_llm().generate(prompt="", system_prompt=sysp, user_message=question,
                          temperature=0.0, max_tokens=120, caller="whatif_parse"))
        d = _extract_json(txt) or {}
        if d.get("is_whatif") and d.get("kind"):
            return d
    except Exception:
        pass
    return None


def apply_whatif(trades: list, spec: dict) -> dict:
    """Apply a what-if filter to the REAL trades and recompute — deterministic, grounded."""
    valid = [t for t in trades if isinstance(t.get("R"), (int, float)) and t["R"] == t["R"]]
    kind, action = spec.get("kind"), spec.get("action", "exclude")
    s, e = _to_min(spec.get("start") or "00:00"), _to_min(spec.get("end") or "23:59")
    tag, sym = (spec.get("tag") or "").upper(), (spec.get("symbol") or "").upper()

    def hit(t):
        if kind == "time":
            m = _to_min(t.get("entry_ts") or "")
            return m is not None and s is not None and e is not None and s <= m <= e
        if kind == "tag":
            return tag in set(t.get("entry_tags") or [])
        if kind == "symbol":
            return (t.get("symbol", "").upper() == sym)
        return False

    inset = [t for t in valid if hit(t)]
    kept, removed = ([t for t in valid if not hit(t)], inset) if action == "exclude" else (inset, [t for t in valid if not hit(t)])
    full, keptl, reml = aggregate_ledger(valid), aggregate_ledger(kept), aggregate_ledger(removed)
    return {"full": full, "kept": keptl, "removed": reml, "removed_n": len(removed),
            "delta_R": round(keptl["total_R"] - full["total_R"], 2),
            "removed_samples": top_samples(removed, 2), "kept_samples": top_samples(kept, 2)}


def _fmt_samples(s: dict) -> str:
    rows = [f"{t['symbol']} {t['date']} {t['time']} {t['R']:+}R {t['outcome']} ({t['context']})"
            for t in (s.get("winners", []) + s.get("losers", []))]
    return "; ".join(rows) if rows else "none"


def answer_analysis(question: str, name: str = "", direction: str = "up", require=None, exclude=None) -> str:
    """Answer a money/hypothesis question grounded in the strategy's REAL backtest ledger (if run) and its
    recognition stats — honest about what each is."""
    perf = None
    trades = []
    if name:
        from intelligence.taught_strategies import load_session
        perf = (load_session(name) or {}).get("performance")
        trades = (perf or {}).get("trades") or []

    # WHAT-IF / SAMPLES intent — detected even when no trades are stored, so the reply is honest + actionable
    ql = question.lower()
    wants_whatif = any(w in ql for w in ("what if", "what-if", "dont trade", "don't trade", "do not trade",
                                         "without", "skip", "exclude ", "only trade", "just trade", "avoid", "remove"))
    wants_samples = any(w in ql for w in ("example", "sample", "show me", "which trades", "winning trade",
                                          "losing trade", "best trade", "worst trade"))
    if wants_whatif or wants_samples:
        if not trades:
            return ("I can test that for real — but I need the individual trades from a backtest, and this "
                    "strategy's last run was saved before I started keeping them. Click “Re-run” at the top, "
                    "then ask again and I'll filter the actual trades and show the exact effect with winning "
                    "and losing examples.")
        spec = parse_whatif(question) if wants_whatif else None
        if spec:
            wf = apply_whatif(trades, spec)
            what = ({"time": f"trades between {spec.get('start')} and {spec.get('end')}",
                     "tag": f"trades tagged {spec.get('tag')}", "symbol": f"{spec.get('symbol')} trades"}
                    .get(spec.get("kind"), "those trades"))
            verb = "skipping" if spec.get("action") == "exclude" else "trading only"
            facts = (f"Full result: {wf['full']['n']} trades, {wf['full']['total_R']}R total, "
                     f"{round(100 * wf['full']['win_rate'])}% win.\n"
                     f"What-if — {verb} {what}: removes {wf['removed_n']} trades (they made {wf['removed']['total_R']}R). "
                     f"Remaining {wf['kept']['n']} trades total {wf['kept']['total_R']}R "
                     f"(change {wf['delta_R']:+}R; expectancy {wf['kept']['expectancy_R']}R/trade vs {wf['full']['expectancy_R']}).\n"
                     f"Examples REMOVED: {_fmt_samples(wf['removed_samples'])}.\n"
                     f"Examples KEPT: {_fmt_samples(wf['kept_samples'])}.")
            sysp = ("You answer a trader's WHAT-IF about their strategy, grounded ONLY in the facts given (real "
                    "trades, R-multiples = risk-units vs the structural stop; not brokerage P&L). Say clearly "
                    "whether the change HELPS or HURTS and by how much (the R delta), then cite a couple of the "
                    "example trades (winners and losers) to explain WHY. Never invent numbers. 3-5 sentences.")
            from intelligence.teaching_service import _get_llm
            try:
                import asyncio
                out = asyncio.run(_get_llm().generate(prompt="", system_prompt=sysp,
                                  user_message=f"FACTS:\n{facts}\n\nQUESTION: {question}",
                                  temperature=0.2, max_tokens=320, caller="analysis_whatif"))
                if out and out.strip():
                    return out.strip()
            except Exception:
                pass
            return facts

        # "show me example / sample winning & losing trades"
        if wants_samples:
            s = top_samples(trades, 3)
            return ("Winning trades:\n  " + "\n  ".join(_fmt_samples({"winners": s["winners"]}).split("; ")) +
                    "\nLosing trades:\n  " + "\n  ".join(_fmt_samples({"losers": s["losers"]}).split("; ")))

    rec = {}
    try:
        from intelligence.strategy_backtest import Strategy, library_stats
        rec = library_stats(Strategy("(x)", direction, set(require or []), set(exclude or []))).to_dict()
    except Exception:
        pass
    if perf and perf.get("n"):
        money = (f"P&L backtest (disciplined operating result — structural stop + asymmetric exit, no capped "
                 f"target): {perf['n']} trades over {perf.get('sessions_run','?')} sessions, "
                 f"{round(100 * perf['win_rate'])}% win rate, total {perf['total_R']}R, "
                 f"expectancy {perf['expectancy_R']}R/trade, profit factor {perf.get('profit_factor')}, "
                 f"avg win {perf['avg_win_R']}R / avg loss {perf['avg_loss_R']}R, max drawdown {perf['max_dd_R']}R.")
    elif perf is not None:
        money = (f"A P&L backtest ran over {perf.get('sessions_run','?')} sessions but found 0 qualifying "
                 f"trades in that sample — entries didn't confirm, or the sample was too small. Run a fuller backtest.")
    else:
        money = "No P&L backtest has been run yet — run one to get the money result."
    recognition = (f"Recognition: catches a real move {round(100 * (rec.get('ran_rate') or 0))}% of the time "
                   f"vs {round(100 * (rec.get('base_ran_rate') or 0))}% base (n={rec.get('n','?')})." if rec.get("n") else "")
    facts = money + "\n" + recognition
    from intelligence.teaching_service import _get_llm
    sysp = ("You answer a trader's questions about a strategy's MONEY performance, plainly and HONESTLY. "
            "The P&L is in R-multiples (1R = the risk to the structural stop); total_R is cumulative risk-units "
            "made/lost; expectancy is per-trade. It is a SIMULATED, disciplined operating result (the stop is "
            "the move's invalidation level, the exit holds while the move is alive then exits on story-change "
            "or close) — NOT a guarantee and NOT actual brokerage P&L. Recognition (how often the move runs) "
            "is a SEPARATE axis from money. Use ONLY the facts given; never invent numbers. If no backtest has "
            "run, say so and suggest running one. 2-4 short sentences.")
    try:
        import asyncio
        out = asyncio.run(_get_llm().generate(prompt="", system_prompt=sysp,
                          user_message=f"FACTS:\n{facts}\n\nQUESTION: {question}",
                          temperature=0.2, max_tokens=240, caller="analysis_ask"))
        if out and out.strip():
            return out.strip()
    except Exception:
        pass
    return facts if perf else ("No P&L backtest has been run yet for this strategy — run one and I'll explain "
                               "the money result (R-multiples, win rate, expectancy, drawdown).")
