# avt/conversation.py
"""
AVT Conversation Layer — Natural Language Interface
=====================================================

Makes AVT feel like a trading partner you can talk to:

  "What are you working on?"
  → "I'm testing an entry filter for NIFTY breakouts (23 trades so far).
     No results yet — need 50 trades minimum."

  "How did I do today?"
  → "You followed your plan on 4/5 trades. The RELIANCE exit was early.
     Discipline score: 8.1."

  "Why should I trust this improvement?"
  → "I traded with this rule for 67 virtual trades over 4 weeks.
     Win rate went from 42% to 56%. Here are 3 specific examples."

Two modes:
  1. TEMPLATE — fast, structured responses (no LLM needed)
  2. LLM — richer, contextual responses using Claude

RULES:
  - Never show unproven hypotheses as suggestions
  - Behavioral feedback is direct and immediate
  - Strategy improvements require AVT proof (50+ trades, 3+ weeks)
  - Always back claims with specific numbers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

class Intent:
    STATUS = "status"             # "What are you working on?"
    DAILY_REVIEW = "daily_review" # "How did I do today?"
    EXPLAIN = "explain"           # "Why this improvement?" / "Explain X"
    MARKET = "market"             # "What's the market doing?"
    PERFORMANCE = "performance"   # "How am I performing?" / "AVT vs me"
    SUGGESTION = "suggestion"     # "What should I do differently?"
    ADOPT = "adopt"               # "Adopt this improvement"
    GENERAL = "general"


_INTENT_KEYWORDS = {
    Intent.STATUS: [
        "what are you working on", "what are you testing", "what are you doing",
        "status", "avt status", "current work", "what's new", "what are you proving",
    ],
    Intent.DAILY_REVIEW: [
        "how did i do", "today's review", "daily review", "session review",
        "how was my day", "my performance today", "today's trades",
    ],
    Intent.EXPLAIN: [
        "why", "explain", "how does", "what does", "tell me about",
        "more detail", "evidence", "proof", "why should i trust",
    ],
    Intent.MARKET: [
        "market", "regime", "nifty", "volatility", "what's happening",
        "market condition", "today's market",
    ],
    Intent.PERFORMANCE: [
        "performance", "avt vs me", "competition", "how am i doing",
        "my stats", "win rate", "expectancy", "compare",
    ],
    Intent.SUGGESTION: [
        "what should i", "suggest", "recommendation", "advice",
        "what can i improve", "help me", "tips",
    ],
    Intent.ADOPT: [
        "adopt", "accept", "use this", "apply this", "yes do it",
    ],
}


def classify_intent(message: str) -> str:
    """Simple keyword-based intent classification."""
    lower = message.lower().strip()
    best_intent = Intent.GENERAL
    best_score = 0

    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


# ══════════════════════════════════════════════════════════════════
#  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════

class ConversationContext:
    """Gathers context from all AVT subsystems for response generation."""

    def __init__(
        self,
        avt_engine: Any = None,
        self_improver: Any = None,
        competition: Any = None,
        observer: Any = None,
        triage: Any = None,
    ):
        self._avt = avt_engine
        self._improver = self_improver
        self._competition = competition
        self._observer = observer
        self._triage = triage

    def get_status_context(self) -> Dict[str, Any]:
        """Context for 'What are you working on?'"""
        ctx: Dict[str, Any] = {
            "avt_running": False,
            "strategies": [],
            "active_proofs": {},
            "proven_improvements": {},
            "pending_messages": 0,
        }

        if self._avt:
            ctx["avt_running"] = getattr(self._avt, "status", "") == "running"
            ctx["strategies"] = list(getattr(self._avt, "strategies", {}).keys())

        if self._improver:
            # What AVT is currently proving (per strategy)
            status = self._improver.get_all_status()
            for sid, s in status.items():
                if s.get("proving"):
                    ctx["active_proofs"][sid] = s["proving"]
                if s.get("proven_count", 0) > 0:
                    ctx["proven_improvements"][sid] = {
                        "count": s["proven_count"],
                        "total_impact_r": s["proven_total_impact_r"],
                    }

        if self._triage:
            ctx["pending_messages"] = self._triage.get_unread_count()

        return ctx

    def get_daily_context(self) -> Dict[str, Any]:
        """Context for 'How did I do today?'"""
        ctx: Dict[str, Any] = {}

        if self._observer:
            ctx["today_summary"] = self._observer.get_today_summary()

        if self._triage:
            ctx["pending_messages"] = self._triage.get_pending_messages(20)

        return ctx

    def get_performance_context(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Context for performance/competition questions."""
        ctx: Dict[str, Any] = {}

        if self._competition and strategy_id:
            try:
                report = self._competition.generate_comparison(
                    strategy_id=strategy_id, period_days=30,
                )
                ctx["competition"] = {
                    "user_stats": report.user_stats,
                    "avt_stats": report.avt_stats,
                    "gap": report.performance_gap,
                    "narrative": report.overall_narrative,
                }
            except Exception:
                pass

        if self._improver and strategy_id:
            proven = self._improver.get_proven_improvements(strategy_id)
            ctx["proven_improvements"] = [
                {
                    "title": p.title,
                    "delta_r": p.proof_delta_r,
                    "trades": p.proof_trades_count,
                }
                for p in proven
            ]

        return ctx

    def get_improvement_detail(self, strategy_id: str) -> Dict[str, Any]:
        """Context for explaining a specific proven improvement."""
        ctx: Dict[str, Any] = {}

        if self._improver:
            proven = self._improver.get_proven_improvements(strategy_id)
            if proven:
                top = proven[0]
                ctx["improvement"] = {
                    "title": top.title,
                    "description": top.description,
                    "type": top.improvement_type,
                    "delta_r": top.proof_delta_r,
                    "trades_tested": top.proof_trades_count,
                    "perf_before": top.proof_performance_before,
                    "perf_after": top.proof_performance_after,
                    "examples": top.proof_example_trades[:3],
                }

            active = self._improver.get_active_proof(strategy_id)
            if active:
                ctx["active_proof"] = {
                    "title": active.title,
                    "trades_so_far": active.proof_trades_count,
                    "trades_needed": active.proof_trades_needed,
                }

        return ctx


# ══════════════════════════════════════════════════════════════════
#  TEMPLATE RESPONSES (no LLM needed)
# ══════════════════════════════════════════════════════════════════

class TemplateResponder:
    """Generates structured responses without LLM."""

    def __init__(self, context: ConversationContext):
        self._ctx = context

    def respond_status(self) -> str:
        ctx = self._ctx.get_status_context()
        parts = []

        if not ctx["avt_running"]:
            return "I'm currently stopped. Start me to begin autonomous trading and learning."

        # What's being proven
        proofs = ctx.get("active_proofs", {})
        if proofs:
            parts.append(f"Currently proving {len(proofs)} improvement{'s' if len(proofs) > 1 else ''}:")
            for sid, proof in proofs.items():
                parts.append(
                    f"  - {proof['title']} for {sid} "
                    f"({proof['trades']}/{proof['needed']} trades)"
                )
        else:
            parts.append("No active improvements being proven right now.")
            parts.append("I'll discover hypotheses from my own trades and start testing them.")

        # What's been proven
        proven = ctx.get("proven_improvements", {})
        if proven:
            total_impact = sum(p["total_impact_r"] for p in proven.values())
            total_count = sum(p["count"] for p in proven.values())
            parts.append(
                f"\n{total_count} proven improvement{'s' if total_count > 1 else ''} "
                f"ready for you (total impact: +{total_impact:.3f}R/trade)."
            )
            parts.append("Ask 'What should I improve?' to see details.")

        pending = ctx.get("pending_messages", 0)
        if pending > 0:
            parts.append(f"\n{pending} unread message{'s' if pending > 1 else ''}.")

        return "\n".join(parts)

    def respond_daily_review(self) -> str:
        ctx = self._ctx.get_daily_context()
        summary = ctx.get("today_summary", {})
        parts = []

        obs = summary.get("observations", [])
        if not obs:
            return "No trading activity observed today."

        by_sev = summary.get("by_severity", {})
        total = summary.get("total_observations", 0)

        parts.append(f"Today's summary ({total} observations):")

        # Positive first
        for o in obs:
            if o.get("severity") == "positive":
                parts.append(f"  + {o['title']}")

        # Then warnings/critical
        for o in obs:
            if o.get("severity") == "critical":
                parts.append(f"  !! {o['title']}: {o['description']}")
            elif o.get("severity") == "warning":
                parts.append(f"  ! {o['title']}")

        if by_sev.get("critical", 0) > 0:
            parts.append(f"\n{by_sev['critical']} critical issue{'s' if by_sev['critical'] > 1 else ''}.")

        return "\n".join(parts)

    def respond_performance(self, strategy_id: Optional[str] = None) -> str:
        ctx = self._ctx.get_performance_context(strategy_id)
        comp = ctx.get("competition")

        if not comp:
            return "Need a strategy_id to compare performance. Try: 'How is my NIFTY strategy doing?'"

        user = comp.get("user_stats", {})
        avt = comp.get("avt_stats", {})
        gap = comp.get("gap", {})

        return (
            f"You vs AVT (last 30 days):\n"
            f"  Your win rate: {user.get('win_rate', 0):.1%} | AVT: {avt.get('win_rate', 0):.1%}\n"
            f"  Your expectancy: {user.get('expectancy_r', 0):.3f}R | AVT: {avt.get('expectancy_r', 0):.3f}R\n"
            f"  Performance gap: {gap.get('expectancy_gap_r', 0):+.3f}R/trade\n\n"
            f"{comp.get('narrative', '')}"
        )

    def respond_suggestion(self) -> str:
        ctx = self._ctx.get_status_context()

        proven = ctx.get("proven_improvements", {})
        if proven:
            # Find the strategy with most impact
            best_sid = max(proven, key=lambda s: proven[s]["total_impact_r"])
            best = proven[best_sid]
            return (
                f"I have {best['count']} proven improvement{'s' if best['count'] > 1 else ''} "
                f"for {best_sid}.\n"
                f"Total expected impact: +{best['total_impact_r']:.3f}R/trade.\n"
                f"These are backed by real virtual trade evidence — not just statistics.\n"
                f"Say 'tell me about the improvement for {best_sid}' for details, "
                f"or 'adopt' to apply."
            )

        proofs = ctx.get("active_proofs", {})
        if proofs:
            sid, proof = next(iter(proofs.items()))
            return (
                f"I'm currently testing an improvement: {proof['title']}\n"
                f"Progress: {proof['trades']}/{proof['needed']} trades.\n"
                f"I'll let you know once I have enough evidence."
            )

        return (
            "No proven improvements yet. I'm analyzing my own trades to find patterns.\n"
            "Once I find something, I'll test it for 50+ trades before suggesting it to you."
        )

    def respond_explain(self, strategy_id: Optional[str] = None) -> str:
        if not strategy_id:
            return "Which strategy? Try: 'Explain the improvement for [strategy name]'"

        ctx = self._ctx.get_improvement_detail(strategy_id)
        imp = ctx.get("improvement")

        if not imp:
            active = ctx.get("active_proof")
            if active:
                return (
                    f"I'm currently testing: {active['title']}\n"
                    f"Progress: {active['trades_so_far']}/{active['trades_needed']} trades.\n"
                    f"I can't explain results yet — still gathering evidence."
                )
            return f"No improvements found for {strategy_id} yet."

        before = imp.get("perf_before", {})
        after = imp.get("perf_after", {})
        examples = imp.get("examples", [])

        parts = [
            f"Improvement: {imp['title']}",
            f"Type: {imp['type']}",
            f"",
            f"Before: {before.get('expectancy_r', 0):.3f}R expectancy, {before.get('win_rate', 0):.1%} win rate",
            f"After:  {after.get('expectancy_r', 0):.3f}R expectancy, {after.get('win_rate', 0):.1%} win rate",
            f"Delta:  +{imp['delta_r']:.3f}R/trade over {imp['trades_tested']} virtual trades",
        ]

        if examples:
            parts.append(f"\nExample trades where this improvement helped:")
            for ex in examples[:3]:
                symbol = ex.get("symbol", "?")
                r = ex.get("r_multiple", 0)
                parts.append(f"  - {symbol}: {r:+.2f}R")

        parts.append(f"\nThis is evidence from my own trading, not theory.")

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
#  LLM RESPONDER
# ══════════════════════════════════════════════════════════════════

class LLMResponder:
    """Uses LLM for richer responses. Falls back to templates."""

    def __init__(
        self,
        context: ConversationContext,
        template: TemplateResponder,
        llm_client: Any = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._ctx = context
        self._template = template
        self._llm = llm_client
        self._model = model

    async def respond(self, intent: str, message: str, **kwargs) -> str:
        if not self._llm:
            return self._template_response(intent, **kwargs)

        try:
            return await self._llm_response(intent, message, **kwargs)
        except Exception as e:
            logger.debug("LLM error, falling back to template: %s", e)
            return self._template_response(intent, **kwargs)

    def _template_response(self, intent: str, **kwargs) -> str:
        dispatch = {
            Intent.STATUS: self._template.respond_status,
            Intent.DAILY_REVIEW: self._template.respond_daily_review,
            Intent.PERFORMANCE: lambda: self._template.respond_performance(kwargs.get("strategy_id")),
            Intent.SUGGESTION: self._template.respond_suggestion,
            Intent.EXPLAIN: lambda: self._template.respond_explain(kwargs.get("strategy_id")),
        }
        handler = dispatch.get(intent)
        if handler:
            return handler()
        return (
            "I'm not sure what you're asking. Try:\n"
            "  - 'What are you working on?'\n"
            "  - 'How did I do today?'\n"
            "  - 'What should I improve?'\n"
            "  - 'AVT vs me' (with a strategy name)"
        )

    async def _llm_response(self, intent: str, message: str, **kwargs) -> str:
        # Build context
        if intent == Intent.STATUS:
            data_ctx = self._ctx.get_status_context()
        elif intent == Intent.DAILY_REVIEW:
            data_ctx = self._ctx.get_daily_context()
        elif intent == Intent.PERFORMANCE:
            data_ctx = self._ctx.get_performance_context(kwargs.get("strategy_id"))
        elif intent == Intent.EXPLAIN:
            data_ctx = self._ctx.get_improvement_detail(kwargs.get("strategy_id", ""))
        else:
            data_ctx = self._ctx.get_status_context()

        from agents.prompts import AVT_PERSONA as system_prompt

        import json
        user_prompt = (
            f"User asked: {message}\n\n"
            f"Intent: {intent}\n\n"
            f"Current context:\n{json.dumps(data_ctx, default=str, indent=2)}\n\n"
            f"Respond naturally as AVT."
        )

        from agents.llm_client import LLMClient, LLMMessage as _LLMMsg

        # Route through LLMClient for token tracking
        if isinstance(self._llm, LLMClient):
            resp = await self._llm.chat(
                messages=[
                    _LLMMsg(role="system", content=system_prompt),
                    _LLMMsg(role="user", content=user_prompt),
                ],
                max_tokens=500,
                temperature=0.5,
                caller="avt:conversation",
            )
            return resp.content
        else:
            # Raw Anthropic client fallback
            response = await self._llm.messages.create(
                model=self._model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text


# ══════════════════════════════════════════════════════════════════
#  AVT CONVERSATION ENGINE
# ══════════════════════════════════════════════════════════════════

class AVTConversation:
    """
    Main conversation interface.

    Usage:
        conversation = AVTConversation(avt_engine, self_improver, ...)
        result = await conversation.chat("What are you working on?")
    """

    def __init__(
        self,
        avt_engine: Any = None,
        self_improver: Any = None,
        competition: Any = None,
        observer: Any = None,
        triage: Any = None,
        llm_client: Any = None,
    ):
        self._context = ConversationContext(
            avt_engine=avt_engine,
            self_improver=self_improver,
            competition=competition,
            observer=observer,
            triage=triage,
        )
        self._template = TemplateResponder(self._context)
        self._llm_responder = LLMResponder(
            context=self._context,
            template=self._template,
            llm_client=llm_client,
        )
        self._triage = triage
        self._self_improver = self_improver

        self._history: List[Dict[str, str]] = []
        self._max_history = 50

    async def chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Process a user message and return AVT's response."""
        intent = classify_intent(message)

        if intent == Intent.ADOPT:
            response = await self._handle_adopt(message)
        else:
            response = await self._llm_responder.respond(intent, message, **kwargs)

        # Record
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "avt", "content": response})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2):]

        # Pending messages
        pending = []
        if self._triage:
            pending = self._triage.get_pending_messages(5)

        return {
            "response": response,
            "intent": intent,
            "pending_messages": pending,
        }

    async def _handle_adopt(self, message: str) -> str:
        if not self._self_improver:
            return "AVT self-improver is not running."

        # Find proven improvements across all strategies
        all_status = self._self_improver.get_all_status()
        proven_strategies = [
            sid for sid, s in all_status.items()
            if s.get("proven_count", 0) > 0
        ]

        if not proven_strategies:
            return "No proven improvements to adopt right now."

        # Show what's available
        parts = ["Proven improvements ready for adoption:"]
        for sid in proven_strategies:
            proven = self._self_improver.get_proven_improvements(sid)
            for p in proven:
                parts.append(
                    f"  - [{sid}] {p.title} (+{p.proof_delta_r:.3f}R, "
                    f"{p.proof_trades_count} trades)"
                )

        parts.append(
            "\nTo adopt a specific one, use the Improvements tab in the AVT dashboard."
        )
        return "\n".join(parts)

    def get_history(self, limit: int = 20) -> List[Dict[str, str]]:
        return self._history[-(limit * 2):]

    def get_proactive_message(self) -> Optional[str]:
        """Check if AVT has something urgent to tell the user."""
        if not self._triage:
            return None

        messages = self._triage.get_pending_messages(1)
        if not messages:
            return None

        top = messages[0]
        # Only proactively push critical or proven improvement messages
        if top.get("priority", 0) >= 8:
            return top.get("message")

        return None
