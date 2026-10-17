# Engineering notes: how a trading system lies to you

Most of the effort in PTIS went into finding the places where the data or the backtest was quietly wrong. These
are the cases that changed the design. Strategy rules and live parameters are private; this page is about the
engineering.

---

## 1. Look-ahead in a multi-timeframe backtest (July 2026)

**What happened.** An early intraday strategy confirmed a signal across 5-minute, 15-minute and 1-hour bars. Its
backtest looked excellent and survived a sealed out-of-sample check.

**The bug.** Higher-timeframe bars were labelled by their *start* time. A 1-hour bar labelled 10:15 cannot be known
until 11:15, but the backtest let the strategy act at 10:15. Every multi-timeframe trade entered before its
confirming bar existed: a median of 40 minutes early, in 30 out of 30 sampled trades.

**The proof.** I re-ran both versions with one change: when you are allowed to act. The edge went from
+1.29R per trade to −0.14R. Before costs it went from +1.83R to +0.12R, so this was not a cost problem: the move
*was* the bar, and by the time the bar closed the move was over. A second, independently built causal version
agreed (−0.26R). The strategy was killed, along with everything selected from its results.

**What changed.**
- Every timestamp now carries the moment it is *knowable*, not just the moment it describes.
- I stopped trusting a result because it passed an out-of-sample check. Out-of-sample does not catch look-ahead;
  only a causality audit does.

## 2. Look-ahead hiding in post-processing (September 2026)

**What happened.** Features and labels had been audited twice. The leak was in a step nobody had audited: a score
normalised as a percentile rank over the *whole day*, including later time slots, then used for morning decisions.

**The gate.** `truncation_test`: rebuild the decision pipeline on data cut off at the decision time and compare it
with the full-data decision. Thirty days were tested at three cut times each.
- The causal score produced 0 mismatches out of 90.
- **Positive control:** the old whole-day score produced 16 mismatches out of 87. A test that cannot fail proves
  nothing, so every gate gets a deliberately broken input that it must catch.

**Rule since then:** any `groupby(date)` rank, quantile, z-score or mean over intraday rows is a red flag until
proven causal. The truncation test runs after any change to scoring, selection or sizing.

## 3. Survivorship bias in the universe

**What happened.** The convenient price history covered stocks that are in the derivatives segment *today*, and
those are mostly yesterday's winners. Every stock-selection test run on it was biased by hindsight.

**The fix.** An all-NSE daily panel built from the exchange's own bhavcopy files: 2012 to 2026, about 3,760
symbols including delisted ones, about 6.5 million rows. Universe membership is point-in-time (what was eligible
*then*). Data traps found on the way:
- Old-format files report a previous close that is often not split-adjusted, which fakes a 50% crash. Splits are
  inferred only when an ISIN change or a standard ratio confirms them.
- NSE trades on some weekends (Budget Saturdays, Muhurat sessions, disaster-recovery drills). A missing weekend
  session fakes a corporate action on the next day.
- The exchange throttles bursts, so failed ranges are retried, never silently skipped.

## 4. Prices that were wrong in three different ways

- **Stitched intraday history:** the 5-minute store was unadjusted, with ~90% level jumps on split dates.
- **Partly adjusted daily history:** splits were adjusted but demergers were not.
- **Coverage holes:** about 16% of stock-days in 2022–26 had no intraday data at all. A return computed across a
  hole looked like a +200% day, and some stocks had no data for years. The backtest was quietly trading fewer
  stocks than it claimed.

The repair layer lists every level break. Each break is verified against an independent source and back-adjusted
only if it is an artefact. Rows right after a hole are dropped, and returns are never computed across a gap. One
public price source turned out to be wrong by 1.5× for part of a stock's history, so a single source is only ever
used to *confirm* a break, never as the truth.

## 5. Two feeds, two truths, and a read that rewrote history (September 2026)

**What happened.** A reconciliation alarm showed that the candle store kept whichever broker feed wrote last. The
two feeds agreed on price (73 of 74 bars identical) but not on volume (10 of 74). One feed reported the
closing-auction window with zero volume. Earlier "clean" reconciliation days had been clean only because both
sides were reading the same wrong source.

**The worse finding.** A plain *read* of a past session could re-fetch it and overwrite it: the file's mtime
changed on a read. A manual repair was silently undone minutes later.

**What changed.** `marketdata/session_seal.py`. Finished sessions are frozen, fetched rows for sealed days are
dropped before any write, and a deliberate repair must re-seal, which records a new fingerprint. This makes the bad
state impossible instead of detecting it afterwards, and it ended a whole class of "the same replay gives a
different answer today" bugs.

## 6. A bug the checks caught on their first run

Pre-market invariant checks are blocking, and each one names the bug it prevents. On their first run they caught
one stock holding 369 bars while every other stock held 375. A repair script built partition paths from the raw
symbol (`M&M`), but the lake sanitises symbols (`M_M`). So the existence check failed, the write was skipped, and
the script still printed "APPLIED". No other test saw it. The script had been written hours earlier, in an AI
pair-programming session: generated code gets the same gates as anything else.

## 7. Prices you cannot actually trade at

Small and mid-cap stocks can lock at their daily circuit limit.
- A buy queued at a locked upper circuit fills only when sellers arrive. That is adverse selection.
- A sell at a locked lower circuit may not fill for days.

A backtest that assumes every close is tradeable overstates both entries and exits. Locked days are modelled as
unfillable at that price, and the live layer has an explicit policy for them.

## 8. Frozen records and line endings

Research verdicts are frozen with SHA-256 manifests, so a result cannot be edited after the fact. On a fresh
checkout the hashes failed: git had normalised line endings. `.gitattributes` now pins each frozen file to the
line endings it was hashed with, and the fix was verified in a clean worktree.

---

## The standard that came out of it

You cannot remove every bug before going live; no firm does. The achievable guarantee is that **no defect reaches
money undetected**. That guarantee is built in five layers:
1. Make bad states impossible (sealed sessions).
2. Verify every assumption before use (blocking invariant checks).
3. Detect problems at runtime, judging heartbeats by their output rather than by liveness.
4. Make failure safe: dry-run by default, a dated live confirmation, a kill switch, and stale data refuses to
   trade.
5. Prove it afterwards: reconciliation, an immutable audit trail and deterministic replay.
