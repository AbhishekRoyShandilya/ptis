"""
intelligence.episodic — the LIBRARY tier: remember EVERY recognised birth + how it RESOLVED, and
recall the past situations the current one RHYMES with.

This is the cross-session memory the scoped working-memory reserved as LIBRARY (working_memory.py).
It is the substrate the COACH/AVT learn from — "I have seen this kind of situation; here is how
situations like it have gone" — grounded recall, NOT a forecast.

PHILOSOPHY (inherited from narrative.py, kept honest):
  · An episode is recalled by its RECOGNITION SIGNATURE — a small set of categorical tags that ARE
    what the eye recognised at the birth moment (direction · with/counter-trend · clear-air/into-
    structure · intraday structure · 1h regime · gap · session phase). Tags, not a feature vector or
    a cosine distance: you can SEE why two episodes are analogues (the shared plot).
  · Recall returns DATED ANALOGUES + an honest outcome tally (n ran / modest / failed, median run),
    explicitly labelled "analogy, not a forecast". Recognition-not-prediction: the tally is grounding
    for the reasoner's judgment, never the verdict itself.
  · Market-sourced: an episode's OUTCOME is the move's realised peak extension-from-birth (the same
    monotonic ext the eye already tracks), classified RAN / MODEST / FAILED. No labels invented.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── the recognition tag vocabulary (readable; this is the "plot grammar") ────────────────────────
# Direction is kept SEPARATE (recall only compares same-direction births — an up-birth and a
# down-birth are never analogues). The rest form the shared plot we score overlap on.

def _phase_tag(hhmm: str) -> str:
    """Session phase from an HH:MM string (IST 5m session 09:15–15:30)."""
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except Exception:
        return ""
    t = h * 60 + m
    if t <= 9 * 60 + 45:
        return "PHASE_OPEN"
    if t <= 11 * 60 + 30:
        return "PHASE_MORNING"
    if t <= 13 * 60:
        return "PHASE_MIDDAY"
    if t <= 14 * 60 + 45:
        return "PHASE_AFTERNOON"
    return "PHASE_LATE"


def _room_tag(room_atr: float) -> str:
    """GRADED room (distance to the nearest OPPOSING structural level, in ATR) — replaces the coarse
    CLEAR_AIR/INTO_STRUCTURE binary (which fired on ~85% of candidates and so validated weak). The
    gradation is what lets the eye tell 'stalled at a wall 0.3 ATR overhead' (decisive) from 'a level
    1.8 ATR away it will blow through'. <0.5 = WALL (into immediate resistance), <1.5 = TIGHT, <3 = SOME,
    else CLEAR (open air)."""
    if room_atr < 0.5:
        return "ROOM_WALL"
    if room_atr < 1.5:
        return "ROOM_TIGHT"
    if room_atr < 3.0:
        return "ROOM_SOME"
    return "ROOM_CLEAR"


def signature_from_birth(direction: str, trend: str, room_atr: float, cur_struct: str,
                         cur_htf: str, open_fact: str, at_open: bool, vwap_side: str,
                         hhmm: str, effort: str = "") -> list[str]:
    """The recognition signature (the plot tags) at a birth, from facts the eye ALREADY has. Direction
    is NOT included here (handled separately by recall); everything else is categorical + readable.
    `room_atr` = distance to nearest opposing structural level (graded); `effort` = the break's
    participation/conviction ('strong'/'weak' from volume or range-expansion)."""
    tags: list[str] = []
    t = (trend or "").upper()
    tags.append("COUNTER_TREND" if "COUNTER" in t else "WITH_TREND" if "WITH" in t else "NO_TREND")
    tags.append(_room_tag(room_atr))
    if cur_struct in ("HH_HL", "LH_LL", "RANGE"):
        tags.append(f"STRUCT_{cur_struct}")
    if cur_htf == "bull":
        tags.append("REGIME_BULL")
    elif cur_htf == "bear":
        tags.append("REGIME_BEAR")
    of = (open_fact or "").upper()
    if "GAPPED UP" in of:
        tags.append("GAP_UP")
    elif "GAPPED DOWN" in of:
        tags.append("GAP_DOWN")
    elif "OPENED FLAT" in of:
        tags.append("FLAT_OPEN")
    if at_open:
        tags.append("AT_OPEN")
    if vwap_side in ("above", "below"):
        tags.append(f"{vwap_side.upper()}_VWAP")
    if effort in ("strong", "weak"):
        tags.append(f"{effort.upper()}_EFFORT")
    ph = _phase_tag(hhmm)
    if ph:
        tags.append(ph)
    return tags


# ── outcome classification ───────────────────────────────────────────────────────────────────────
RAN, MODEST, FAILED = "RAN", "MODEST", "FAILED"


def classify_outcome(peak_ext: float) -> str:
    """Realised result of a birth by its peak extension-from-birth (ATR). A move that never got
    >3 ATR of follow-through FAILED (the user's weak-follow-through tell); ≥5 ATR RAN; between = MODEST."""
    if not (peak_ext == peak_ext):           # NaN
        return FAILED
    if peak_ext >= 5.0:
        return RAN
    if peak_ext >= 3.0:
        return MODEST
    return FAILED


def episodes_from_session(symbol: str, date, steps) -> list["Episode"]:
    """Turn a narrated session (state_narrator.narrate_session output) into outcome-labelled episodes:
    each birth step (those carrying a recognition signature in `.tags`) is paired with the PEAK
    extension-from-birth its move reached before it died — the same monotonic ext the eye tracks. The
    move's life runs from the birth until the next birth or until the move releases (move_ext → NaN)."""
    births = [j for j, s in enumerate(steps) if s.tags]
    out: list[Episode] = []
    for idx, j in enumerate(births):
        end = births[idx + 1] if idx + 1 < len(births) else len(steps)
        peak = float("nan")
        for k in range(j, end):
            me = steps[k].move_ext
            if me == me:                                  # finite (move still active)
                peak = me if peak != peak else max(peak, me)
        s = steps[j]
        parts = (s.birth or "").split()
        direction = parts[1] if len(parts) > 1 and parts[1] in ("up", "down") else ""
        if not direction:
            continue
        out.append(Episode(symbol=symbol, date=str(date), ts=f"{s.ts:%H:%M}", direction=direction,
                           tags=list(s.tags), peak_ext=peak, outcome=classify_outcome(peak)))
    return out


@dataclass
class Episode:
    symbol: str
    date: str                 # YYYY-MM-DD
    ts: str                   # HH:MM (birth time)
    direction: str            # up / down
    tags: list[str] = field(default_factory=list)
    peak_ext: float = float("nan")
    outcome: str = FAILED

    def key(self) -> str:
        return f"{self.symbol}|{self.date}|{self.ts}|{self.direction}"


# ── the store (append-only jsonl; cheap, inspectable, git-diffable) ─────────────────────────────────
class EpisodeStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path("data/lake/_episodes/episodes.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Episode]:
        if not self.path.exists():
            return []
        out: list[Episode] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Episode(**json.loads(line)))
        return out

    def save(self, episodes: list[Episode]) -> None:
        """Replace the store with `episodes` (deduped by key — re-harvesting a session overwrites)."""
        dedup: dict[str, Episode] = {e.key(): e for e in episodes}
        with self.path.open("w", encoding="utf-8") as fh:
            for e in dedup.values():
                fh.write(json.dumps(asdict(e)) + "\n")

    def merge(self, new: list[Episode]) -> list[Episode]:
        """Add `new` to whatever is on disk (new wins on key collision) and persist; returns all."""
        dedup: dict[str, Episode] = {e.key(): e for e in self.load()}
        for e in new:
            dedup[e.key()] = e
        allall = list(dedup.values())
        self.save(allall)
        return allall


# ── recall (shared-plot overlap; same direction; dated analogues + honest tally) ────────────────────
@dataclass
class Analogue:
    episode: Episode
    shared: list[str]
    score: float


@dataclass
class RecallResult:
    direction: str
    tags: list[str]
    analogues: list[Analogue]      # top-k by score (closest first)
    n: int = 0                     # all candidates sharing >= min_shared tags (the loose pool)
    n_close: int = 0               # candidates that genuinely RHYME (score >= close_min) — the tally base
    n_ran: int = 0                 # the tally below is over the CLOSE subset (honest: weak overlaps excluded)
    n_modest: int = 0
    n_failed: int = 0
    median_run: float = float("nan")

    def any(self) -> bool:
        return self.n_close > 0


def recall(library: list[Episode], direction: str, tags: list[str], k: int = 5,
           min_shared: int = 2, close_min: float = 0.4, exclude_key: str | None = None) -> RecallResult:
    """Find the same-direction past episodes whose plot RHYMES with `tags`. Score = (shared)/(union) —
    Jaccard, so a small precise overlap isn't drowned by tag count. The outcome TALLY is computed over
    the CLOSE subset (score >= close_min) only — NOT every loose ≥min_shared candidate — so a handful of
    weak overlaps can't manufacture a misleading 'mostly ran' headline against the close analogues."""
    want = set(tags)
    cand: list[Analogue] = []
    for e in library:
        if e.direction != direction or e.key() == exclude_key:
            continue
        shared = want & set(e.tags)
        if len(shared) < min_shared:
            continue
        union = want | set(e.tags)
        score = len(shared) / len(union) if union else 0.0
        cand.append(Analogue(episode=e, shared=sorted(shared), score=round(score, 3)))
    cand.sort(key=lambda a: (a.score, a.episode.peak_ext if a.episode.peak_ext == a.episode.peak_ext else 0), reverse=True)
    close = [a.episode for a in cand if a.score >= close_min]
    runs = sorted(e.peak_ext for e in close if e.outcome == RAN and e.peak_ext == e.peak_ext)
    med = runs[len(runs) // 2] if runs else float("nan")
    return RecallResult(
        direction=direction, tags=tags, analogues=cand[:k], n=len(cand), n_close=len(close),
        n_ran=sum(e.outcome == RAN for e in close),
        n_modest=sum(e.outcome == MODEST for e in close),
        n_failed=sum(e.outcome == FAILED for e in close),
        median_run=round(med, 1) if med == med else float("nan"),
    )


def render_recall(rr: RecallResult) -> str:
    """One LIBRARY-scope fact line (+ a couple of dated analogues). Honest: analogy, not a forecast; the
    tally is over the CLOSE analogues only."""
    if not rr.any():
        return ""
    plot = " · ".join([rr.direction.upper()] + [t for t in rr.tags if not t.startswith("PHASE")][:4])
    med = f", median run +{rr.median_run} ATR" if rr.median_run == rr.median_run else ""
    head = (f"{rr.n_close} close analogues to this ({plot}): "
            f"{rr.n_ran} ran{med}, {rr.n_modest} modest, {rr.n_failed} failed (<3 ATR) "
            f"[analogy, not a forecast; n={rr.n_close}]")
    lines = [head]
    for a in rr.analogues[:3]:
        e = a.episode
        ext = f"+{e.peak_ext:.1f}" if e.peak_ext == e.peak_ext else "?"
        lines.append(f"   ~ {e.symbol} {e.date} {e.ts} → {e.outcome} ({ext} ATR) [shares: {', '.join(a.shared)}]")
    return "\n".join(lines)
