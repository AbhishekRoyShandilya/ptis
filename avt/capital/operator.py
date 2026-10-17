"""
avt.capital.operator — the CAPITAL OPERATOR seam (v0).

The spine is: Strategy → **INTENT** → Capital Operator (take? wait? skip? which instrument? how big?) → Execution
Engine. This class is that operator node. In v0 it is PHYSICALLY in the loop but ADVISORY: it always resolves to
`take` at the intent's `preferred_instrument` — which is exactly the validated reflex baseline — so wiring it in
does NOT move the frozen golden cascade numbers (391 / 68.8% / +1.16R). It consults the SimWorld
(`decide_candidate`) for a point-in-time outcome distribution as TELEMETRY only; that estimate is recorded on the
decision but never gates yet.

Why advisory first: capital errors MULTIPLY (docs/avt_adaptive_operator.md §1). We put the seam in place on the
already-proven path with zero behaviour change, THEN — once the SimWorld is calibrated and back-measured — switch
`gating` on so `wait`/`skip` and instrument override actually fire. The LLM Reasoner stays deferred (built last).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Tuple

from avt.intent import Instrument, Intent
from avt.capital.simworld import OutcomeDistribution, decide_candidate
from avt.execution.options import MoneyPolicy, money_lots


@dataclass
class CapitalDecision:
    action: str                              # take | wait | skip
    instrument: Instrument
    size_mult: float = 1.0                   # operator's size scalar (v0: 1.0 = reflex)
    verdict: str = "use_reflex"              # 'estimate' | 'use_reflex' (from the SimWorld)
    rationale: List[str] = field(default_factory=list)
    distribution: Optional[OutcomeDistribution] = None


@dataclass
class AccountState:
    """The operator's live view of the account — the ONE place equity / peak / drawdown live, shared across
    instruments so money discipline (compounding cap, drawdown scaling) is unified."""
    starting_capital: float
    equity: Optional[float] = None
    peak_equity: Optional[float] = None

    def __post_init__(self):
        if self.equity is None:
            self.equity = self.starting_capital
        if self.peak_equity is None:
            self.peak_equity = self.equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def set_equity(self, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)


class CapitalOperator:
    """The capital brain. TWO roles:
      1) decide_open() — the take/instrument seam (v0 advisory-reflex; unchanged, keeps the futures golden intact).
      2) ACCOUNT-AWARE SIZING (size_option / drawdown_mult) — the ONE disciplined money brain: a MoneyPolicy base
         (capped-compound + a user-permission gate to scale past the cap) scaled by a unified drawdown ladder and
         bounded by whole lots / max_lots / affordability. The live options runner sizes through this so discipline
         lives in one place; futures unification (size_futures) is a later, golden-gated step.

    `bench_provider()` (optional) returns closed trades for the SimWorld telemetry. `account` + `money` enable the
    sizing role; `scale_permission` (a deliberate USER action) lets size compound past the capped_compound cap;
    `dd_scaling` toggles the drawdown ladder."""

    def __init__(self, bench_provider: Optional[Callable[[], Any]] = None, gating: bool = False,
                 money: Optional[MoneyPolicy] = None, account: Optional[AccountState] = None,
                 scale_permission: bool = False, dd_scaling: bool = False):
        self._bench_provider = bench_provider
        self._gating = gating
        self.money = money or MoneyPolicy()
        self.account = account
        self.scale_permission = scale_permission
        self.dd_scaling = dd_scaling

    # ── account-aware sizing (the money brain) ──
    def drawdown_mult(self, account: Optional[AccountState] = None) -> float:
        """Unified drawdown ladder (opt-in via `dd_scaling`; OFF by default). Reduce risk as the account bleeds;
        halt at 20%. `account` overrides the operator's own (per-sleeve drawdown, B.1). Kept off during the live
        TESTING phase — on small virtual capital a 10% drawdown is normal in a losing streak, and we want to
        OBSERVE the strategy's natural trades/capital first, then decide drawdown handling from data rather than
        impose a fixed ladder blind."""
        acct = account if account is not None else self.account
        if not self.dd_scaling or acct is None:
            return 1.0
        dd = acct.drawdown_pct
        if dd >= 0.20:
            return 0.0
        if dd >= 0.15:
            return 0.25
        if dd >= 0.10:
            return 0.50
        if dd >= 0.05:
            return 0.75
        return 1.0

    def size_option(self, buy_fill: float, lot: int,
                    account: Optional[AccountState] = None) -> Tuple[int, List[str]]:
        """Whole option lots under the disciplined policy + account state. `account` overrides the operator's own
        account when sizing off a specific capital SLEEVE (B.1 — e.g. the ₹50k options pool); defaults to the
        operator's account. Returns (lots, rationale)."""
        acct = account if account is not None else self.account
        if acct is None:
            return 0, ["no account wired → size 0"]
        dd_mult = self.drawdown_mult(acct)
        if dd_mult <= 0:
            return 0, [f"drawdown {acct.drawdown_pct:.0%} ≥ 20% halt → size 0"]
        eff = self.money
        if self.scale_permission and eff.mode == "capped_compound":
            eff = replace(eff, grow_to_x=1e9)   # user permission: uncap the compounding base (max_lots still bounds)
        lots = money_lots(eff, capital=acct.equity,
                          starting_capital=acct.starting_capital, buy_fill=buy_fill, lot=lot)
        lots = int(lots * dd_mult)
        why = [f"{eff.mode}{' +permission' if self.scale_permission else ''} eq₹{acct.equity:,.0f} "
               f"dd {acct.drawdown_pct:.0%}×{dd_mult:.2f} → {max(0, lots)} lots (≤{eff.max_lots})"]
        return max(0, lots), why

    def mark_equity(self, equity: float) -> None:
        """Sync the account after a close (equity is the runner's running capital)."""
        if self.account is not None:
            self.account.set_equity(equity)

    def decide_open(self, intent: Intent, as_of: Any) -> CapitalDecision:
        """Decide whether/how to act on an Intent at `as_of`. v0: always take at the preferred instrument; the
        SimWorld estimate is attached for telemetry but does not gate (unless `gating` is explicitly enabled)."""
        dist: Optional[OutcomeDistribution] = None
        rationale: List[str] = ["v0 reflex: take at preferred instrument"]
        if self._bench_provider is not None:
            try:
                bench = self._bench_provider()
                if bench is not None and len(bench):
                    dist = decide_candidate(bench, intent.symbol, intent.direction, as_of).get("take")
                    if dist is not None:
                        rationale = list(dist.evidence)
            except Exception:
                # telemetry must never break the trade path
                dist = None

        action = "take"
        instrument = intent.preferred_instrument
        if self._gating and dist is not None and dist.verdict == "estimate":
            # gating path (OFF by default until the SimWorld is calibrated): only skip on a confidently-negative read
            if dist.mean_r <= 0 and dist.prob_profit < 0.5:
                action = "skip"
                rationale = list(dist.evidence) + [f"gated skip: meanR {dist.mean_r:+.2f}, P(profit) {dist.prob_profit:.0%}"]

        return CapitalDecision(
            action=action, instrument=instrument, size_mult=1.0,
            verdict=(dist.verdict if dist is not None else "use_reflex"),
            rationale=rationale, distribution=dist,
        )
