# rag/semantic_converter.py
"""
Semantic Converter — Dual Storage Pattern.

Every conversion produces BOTH:
  text       → for embedding in vector DB (semantic search)
  structured → dict for Postgres (computation, stats, filtering)

Architecture Rules:
  - Convert features → semantic text → THEN embed (never raw numbers)
  - Granular vocabulary per feature type
  - Combination rules for context-aware phrases
  - Quality score per conversion (0-1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rag.layered_store import MemoryLayer

logger = logging.getLogger(__name__)


@dataclass
class DualOutput:
    """Every conversion produces both text and structured data."""
    text: str               # for embedding (semantic search)
    structured: dict        # for Postgres (computation, filtering)
    quality_score: float    # 0-1, how complete/reliable the conversion is
    layer: MemoryLayer      # which memory layer this belongs to
    entry_id: str = ""      # source ID (trade_id, profile_id, etc.)
    title: str = ""         # human-readable title for the chunk
    source: str = "system"  # "user" | "avt" | "system"
    tags: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  VOCABULARY — granular semantic phrases per feature type
# ══════════════════════════════════════════════════════════════════

def _volume_phrase(ratio: float) -> str:
    if ratio >= 3.0:
        return f"massive volume surge ({ratio:.1f}x average)"
    if ratio >= 1.8:
        return f"significant volume ({ratio:.1f}x average)"
    if ratio >= 1.2:
        return f"moderate volume ({ratio:.1f}x average)"
    if ratio >= 0.5:
        return f"below-average volume ({ratio:.1f}x)"
    return f"volume dry-up ({ratio:.1f}x)"


def _momentum_phrase(atr_move: float) -> str:
    if atr_move >= 2.5:
        return f"explosive momentum (+{atr_move:.1f} ATR)"
    if atr_move >= 1.5:
        return f"strong momentum (+{atr_move:.1f} ATR)"
    if atr_move >= 0.8:
        return f"moderate momentum (+{atr_move:.1f} ATR)"
    if atr_move >= 0.3:
        return f"weak momentum (+{atr_move:.1f} ATR)"
    return "stalled momentum"


def _compression_phrase(cv: float) -> str:
    if cv < 0.15:
        return f"ultra-tight compression (CV={cv:.2f})"
    if cv < 0.30:
        return f"clean compression (CV={cv:.2f})"
    if cv < 0.50:
        return f"moderate compression (CV={cv:.2f})"
    if cv < 0.70:
        return f"loose range (CV={cv:.2f})"
    return "no compression"


def _wick_phrase(ratio: float) -> str:
    if ratio < 0.3:
        return f"clean candle (wick ratio {ratio:.2f})"
    if ratio < 0.5:
        return f"moderate wicks (ratio {ratio:.2f})"
    if ratio < 0.7:
        return f"indecisive price action (wick ratio {ratio:.2f})"
    return f"ugly rejection candle (wick ratio {ratio:.2f})"


def _regime_phrase(regime: str) -> str:
    phrases = {
        "trending": "strong trending market with directional momentum",
        "ranging": "range-bound, sideways market with no clear direction",
        "high_volatility": "high-volatility environment with large moves",
        "low_volatility": "low-volatility, compressed market environment",
        "breakout": "breakout regime with expanding ranges",
        "mean_reverting": "mean-reverting market favoring fade strategies",
        "unknown": "uncertain market regime",
    }
    return phrases.get(regime.lower(), f"{regime} market regime")


def _conviction_phrase(score: float) -> str:
    if score >= 9:
        return f"very high conviction ({score:.0f}/10)"
    if score >= 7:
        return f"high conviction ({score:.0f}/10)"
    if score >= 4:
        return f"moderate conviction ({score:.0f}/10)"
    return f"low conviction ({score:.0f}/10)"


def _rr_phrase(r_multiple: float) -> str:
    if r_multiple >= 3.0:
        return f"excellent R-multiple (+{r_multiple:.1f}R)"
    if r_multiple >= 2.0:
        return f"strong R-multiple (+{r_multiple:.1f}R)"
    if r_multiple >= 1.0:
        return f"acceptable R-multiple (+{r_multiple:.1f}R)"
    if r_multiple >= 0:
        return f"marginal outcome (+{r_multiple:.1f}R)"
    if r_multiple >= -0.5:
        return f"small loss ({r_multiple:.1f}R)"
    return f"full stop-loss hit ({r_multiple:.1f}R)"


def _outcome_phrase(pnl: float, r_multiple: Optional[float]) -> str:
    if r_multiple is not None:
        tag = "winner" if r_multiple > 0 else "loser"
        return f"trade {tag}, {_rr_phrase(r_multiple)}"
    tag = "profit" if pnl > 0 else "loss"
    return f"trade closed at {tag} (P&L {pnl:+.2f})"


# ══════════════════════════════════════════════════════════════════
#  COMBINATION RULES — context-aware multi-feature phrases
# ══════════════════════════════════════════════════════════════════

def _combination_phrases(features: Dict[str, float]) -> List[str]:
    """Generate context-aware phrases from feature combinations."""
    phrases = []
    vol = features.get("volume_surge", 1.0)
    compression = features.get("range_compression", 1.0)
    momentum = features.get("momentum_score", 0.0)
    wick = features.get("wick_ratio", 0.5)

    if vol > 2.0 and compression < 0.5:
        phrases.append("coiled spring with volume confirmation")
    if wick > 0.5 and momentum < 0.3:
        phrases.append("indecisive price action, momentum stalling")
    if vol > 1.5 and momentum > 1.5:
        phrases.append("strong conviction move with volume backing")
    if compression < 0.3 and momentum > 1.0:
        phrases.append("compressed range breakout with directional thrust")
    if vol < 0.7 and momentum > 1.5:
        phrases.append("light-volume momentum, sustainability questionable")

    return phrases


# ══════════════════════════════════════════════════════════════════
#  CONVERTER CLASS
# ══════════════════════════════════════════════════════════════════

class SemanticConverter:
    """
    Converts PTIS domain objects into DualOutput (text + structured).

    Each method handles one object type. All are safe — failures
    are logged but never crash the pipeline.
    """

    # ── Trade → TRADE layer ──────────────────────────────────────

    def trade_to_dual(
        self,
        trade,
        attribution=None,
        journal=None,
        dq=None,
    ) -> DualOutput:
        """Convert a Trade (SQLAlchemy model) to dual output."""
        try:
            parts = []
            quality = 0.0
            fields_present = 0
            fields_total = 8

            # Core trade info
            symbol = getattr(trade, "symbol", "???")
            direction = str(getattr(trade, "direction", "")).replace("TradeDirection.", "")
            strategy = getattr(trade, "declared_strategy_id", None) or ""
            regime = str(getattr(trade, "market_regime", "unknown")).replace("MarketRegime.", "")
            pnl = getattr(trade, "pnl", 0.0) or 0.0
            r_mult = getattr(trade, "r_multiple", None)
            entry_price = getattr(trade, "entry_price", 0.0)
            exit_price = getattr(trade, "exit_price", None)
            conviction = getattr(trade, "conviction_score", None)

            parts.append(f"{symbol} {direction} trade")
            fields_present += 1

            if strategy:
                parts.append(f"using {strategy} strategy")
                fields_present += 1

            parts.append(f"in {_regime_phrase(regime)}")
            fields_present += 1

            if entry_price:
                price_str = f"entry at {entry_price:.2f}"
                if exit_price:
                    price_str += f", exit at {exit_price:.2f}"
                parts.append(price_str)
                fields_present += 1

            parts.append(_outcome_phrase(pnl, r_mult))
            fields_present += 1

            if conviction:
                parts.append(_conviction_phrase(conviction))
                fields_present += 1

            # Attribution context
            if attribution:
                disc = getattr(attribution, "discipline_score", None)
                if disc is not None:
                    parts.append(f"discipline score {disc:.0f}/100")
                    fields_present += 1
                resolved = getattr(attribution, "resolved_strategy_id", None)
                if resolved and resolved != strategy:
                    parts.append(f"auto-matched to {resolved}")

            # Journal context
            if journal:
                lesson = getattr(journal, "lesson", None)
                if lesson:
                    parts.append(f"Lesson: {lesson[:200]}")
                    fields_present += 1
                plan = getattr(journal, "plan", None)
                if plan:
                    parts.append(f"Plan: {plan[:150]}")

            # Decision quality
            if dq:
                grade = getattr(dq, "grade", None)
                composite = getattr(dq, "composite_score", None)
                if grade:
                    parts.append(f"decision quality: {grade} ({composite:.0f}/100)")
                    fields_present += 1

            quality = fields_present / fields_total
            text = ". ".join(parts) + "."

            structured = {
                "symbol": symbol,
                "direction": direction,
                "strategy": strategy,
                "regime": regime,
                "pnl": pnl,
                "r_multiple": r_mult,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "conviction": conviction,
                "discipline_score": getattr(attribution, "discipline_score", None) if attribution else None,
                "grade": getattr(dq, "grade", None) if dq else None,
                "had_journal": journal is not None,
                "followed_plan": getattr(journal, "followed_plan", None) if journal else None,
            }

            trade_id = str(getattr(trade, "id", ""))
            source_val = getattr(trade, "source", "manual") or "manual"
            # Map broker sources to "user"
            src = "user" if source_val != "avt" else "avt"

            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.TRADE,
                entry_id=trade_id,
                title=f"{symbol} {direction} {_outcome_phrase(pnl, r_mult)[:30]}",
                source=src,
                tags=[symbol, strategy, regime] if strategy else [symbol, regime],
            )
        except Exception as e:
            logger.warning("trade_to_dual failed: %s", e)
            return DualOutput(
                text=f"Trade conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.TRADE,
            )

    # ── AVT Trade → TRADE layer (source="avt") ──────────────────

    def avt_trade_to_dual(self, virtual_trade: dict) -> DualOutput:
        """Convert AVT virtual trade dict to dual output."""
        try:
            symbol = virtual_trade.get("symbol", "???")
            direction = virtual_trade.get("direction", "long")
            strategy = virtual_trade.get("strategy_id", "")
            regime = virtual_trade.get("regime", "unknown")
            pnl = virtual_trade.get("pnl", 0.0)
            r_mult = virtual_trade.get("r_multiple", None)
            version = virtual_trade.get("avt_version", "")

            parts = [
                f"AVT virtual trade: {symbol} {direction}",
                f"strategy {strategy}" if strategy else "",
                f"in {_regime_phrase(regime)}",
                _outcome_phrase(pnl, r_mult),
                f"AVT version {version}" if version else "",
            ]
            text = ". ".join(p for p in parts if p) + "."

            structured = {
                "symbol": symbol,
                "direction": direction,
                "strategy": strategy,
                "regime": regime,
                "pnl": pnl,
                "r_multiple": r_mult,
                "avt_version": version,
                "source": "avt",
            }

            quality = 0.6 if r_mult is not None else 0.4
            trade_id = virtual_trade.get("id", virtual_trade.get("trade_id", ""))

            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.TRADE,
                entry_id=str(trade_id),
                title=f"AVT {symbol} {direction}",
                source="avt",
                tags=[symbol, strategy, "avt"],
            )
        except Exception as e:
            logger.warning("avt_trade_to_dual failed: %s", e)
            return DualOutput(
                text=f"AVT trade conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.TRADE,
                source="avt",
            )

    # ── StrategyProfile → STRATEGY layer ─────────────────────────

    def strategy_profile_to_dual(self, profile) -> DualOutput:
        """Convert a StrategyProfile (SQLAlchemy model) to dual output."""
        try:
            name = getattr(profile, "name", "")
            code = getattr(profile, "short_code", "")
            narrative = getattr(profile, "narrative", "") or ""
            status = getattr(profile, "status", "draft")

            entry_criteria = getattr(profile, "entry_criteria", []) or []
            exit_rules = getattr(profile, "exit_rules", {}) or {}
            risk_params = getattr(profile, "risk_parameters", {}) or {}
            regime_prefs = getattr(profile, "regime_preferences", {}) or {}
            confidence = getattr(profile, "confidence_scores", {}) or {}
            examples = getattr(profile, "examples_count", 0) or 0

            parts = [f"Strategy: {name} ({code})"]

            if narrative:
                parts.append(narrative[:500])

            # Entry criteria
            if entry_criteria:
                criteria_texts = []
                for c in entry_criteria[:8]:
                    desc = c.get("description", c.get("criterion", ""))
                    conf = c.get("confidence", 0)
                    criteria_texts.append(f"  - {desc} (confidence {conf:.0%})")
                parts.append("Entry criteria:\n" + "\n".join(criteria_texts))

            # Exit rules
            if exit_rules:
                stop_method = exit_rules.get("stop_method", "")
                if stop_method:
                    parts.append(f"Stop method: {stop_method}")
                typical_atr = exit_rules.get("typical_stop_atr", 0)
                if typical_atr:
                    parts.append(f"Typical stop: {typical_atr:.1f} ATR")

            # Risk
            if risk_params:
                rr = risk_params.get("typical_rr", 0)
                risk_pct = risk_params.get("typical_risk_pct", 0)
                if rr:
                    parts.append(f"Typical R:R = {rr:.1f}")
                if risk_pct:
                    parts.append(f"Risk per trade: {risk_pct:.1%}")

            # Regime preferences
            if regime_prefs:
                preferred = regime_prefs.get("preferred", [])
                avoid = regime_prefs.get("avoid", [])
                if preferred:
                    regimes = [r if isinstance(r, str) else r.get("regime", "") for r in preferred]
                    parts.append(f"Preferred regimes: {', '.join(regimes)}")
                if avoid:
                    regimes = [r if isinstance(r, str) else r.get("regime", "") for r in avoid]
                    parts.append(f"Avoid regimes: {', '.join(regimes)}")

            # Confidence
            overall_conf = confidence.get("overall", 0)
            if overall_conf:
                parts.append(f"Overall confidence: {overall_conf:.0%}")

            parts.append(f"Status: {status}, {examples} examples")

            text = ". ".join(parts) + "."
            quality = min(1.0, 0.3 + 0.1 * min(len(entry_criteria), 5) + (0.2 if narrative else 0))

            structured = {
                "name": name,
                "short_code": code,
                "status": status,
                "entry_criteria_count": len(entry_criteria),
                "regime_preferences": regime_prefs,
                "risk_parameters": risk_params,
                "confidence_scores": confidence,
                "examples_count": examples,
            }

            profile_id = str(getattr(profile, "id", ""))
            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.STRATEGY,
                entry_id=profile_id,
                title=f"Strategy: {name} ({code})",
                source="user",
                tags=[code, name, status],
            )
        except Exception as e:
            logger.warning("strategy_profile_to_dual failed: %s", e)
            return DualOutput(
                text=f"Strategy profile conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.STRATEGY,
            )

    # ── PatternFingerprint → PATTERN layer ───────────────────────

    def fingerprint_to_dual(self, fp) -> DualOutput:
        """Convert a PatternFingerprint (dataclass) to dual output."""
        try:
            labels = getattr(fp, "labels", [])
            vector = getattr(fp, "vector", [])
            symbol = getattr(fp, "symbol", "")
            timeframe = getattr(fp, "timeframe", "")
            direction = getattr(fp, "direction", 1)

            features = {}
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            for label, val in zip(labels, vector):
                features[label] = val

            # Build semantic text from features
            parts = [f"Pattern fingerprint for {symbol} ({timeframe})"]
            dir_str = "bullish" if direction == 1 else "bearish"
            parts.append(f"{dir_str} pattern")

            vol = features.get("volume_surge", 1.0)
            parts.append(_volume_phrase(vol))
            parts.append(_momentum_phrase(features.get("momentum_score", 0.0)))

            rc = features.get("range_compression", 1.0)
            if rc < 0.7:
                parts.append(_compression_phrase(rc))

            wick = features.get("upper_wick_ratio", 0.5)
            parts.append(_wick_phrase(wick))

            rsi = features.get("rsi_at_trigger", 0.5) * 100
            parts.append(f"RSI at trigger: {rsi:.0f}")

            # Combination phrases
            combos = _combination_phrases(features)
            parts.extend(combos)

            text = ". ".join(parts) + "."
            quality = 0.7 if len(vector) == 12 else 0.4

            structured = {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "features": features,
            }

            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.PATTERN,
                entry_id=f"fp_{symbol}_{timeframe}",
                title=f"Pattern: {symbol} {dir_str}",
                source="system",
                tags=[symbol, timeframe, dir_str],
            )
        except Exception as e:
            logger.warning("fingerprint_to_dual failed: %s", e)
            return DualOutput(
                text=f"Fingerprint conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.PATTERN,
            )

    # ── Micro-features dict → PATTERN layer ──────────────────────

    def micro_features_to_dual(
        self, features: Dict[str, float], label: str = ""
    ) -> DualOutput:
        """Convert a raw features dict into semantic text."""
        try:
            parts = [f"Micro-features{f' for {label}' if label else ''}"]
            for key, val in features.items():
                if "volume" in key.lower():
                    parts.append(f"{key}: {_volume_phrase(val)}")
                elif "momentum" in key.lower():
                    parts.append(f"{key}: {_momentum_phrase(val)}")
                elif "compression" in key.lower() or "cv" in key.lower():
                    parts.append(f"{key}: {_compression_phrase(val)}")
                elif "wick" in key.lower():
                    parts.append(f"{key}: {_wick_phrase(val)}")
                else:
                    parts.append(f"{key}: {val:.3f}")

            combos = _combination_phrases(features)
            parts.extend(combos)

            text = ". ".join(parts) + "."
            return DualOutput(
                text=text,
                structured={"label": label, "features": features},
                quality_score=0.5,
                layer=MemoryLayer.PATTERN,
                entry_id=f"mf_{label}",
                title=f"Micro-features: {label}",
                source="system",
                tags=[label] if label else [],
            )
        except Exception as e:
            logger.warning("micro_features_to_dual failed: %s", e)
            return DualOutput(
                text=f"Micro-features conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.PATTERN,
            )

    # ── DecisionQualityRecord → INSIGHT layer ────────────────────

    def decision_quality_to_dual(self, dq) -> DualOutput:
        """Convert a DecisionQualityRecord to dual output."""
        try:
            grade = getattr(dq, "grade", "?")
            composite = getattr(dq, "composite_score", 0)
            is_quality = getattr(dq, "is_quality_trade", True)
            is_violation = getattr(dq, "is_process_violation", False)
            coaching = getattr(dq, "coaching_notes", "") or ""
            dimensions = getattr(dq, "dimension_details", {}) or {}
            insights = getattr(dq, "insights", []) or []

            parts = [f"Decision quality assessment: {grade} ({composite:.0f}/100)"]

            if is_violation:
                parts.append("PROCESS VIOLATION detected")
            elif is_quality:
                parts.append("quality trade execution")

            # Dimension breakdown
            if isinstance(dimensions, dict):
                for dim_name, dim_data in dimensions.items():
                    if isinstance(dim_data, dict):
                        score = dim_data.get("score", 0)
                        parts.append(f"{dim_name}: {score:.0f}")
                    else:
                        parts.append(f"{dim_name}: {dim_data}")

            if coaching:
                parts.append(f"Coaching: {coaching[:200]}")

            if insights:
                for ins in insights[:3]:
                    if isinstance(ins, dict):
                        parts.append(ins.get("text", ins.get("insight", str(ins)))[:150])
                    else:
                        parts.append(str(ins)[:150])

            text = ". ".join(parts) + "."

            structured = {
                "grade": grade,
                "composite_score": composite,
                "is_quality_trade": is_quality,
                "is_process_violation": is_violation,
                "dimensions": dimensions,
            }

            eval_id = str(getattr(dq, "evaluation_id", ""))
            return DualOutput(
                text=text,
                structured=structured,
                quality_score=0.7 if grade else 0.3,
                layer=MemoryLayer.INSIGHT,
                entry_id=eval_id,
                title=f"DQ: {grade} ({composite:.0f}/100)",
                source="system",
                tags=["decision_quality", grade],
            )
        except Exception as e:
            logger.warning("decision_quality_to_dual failed: %s", e)
            return DualOutput(
                text=f"DQ conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.INSIGHT,
            )

    # ── Journal → INSIGHT layer ──────────────────────────────────

    def journal_to_dual(self, journal, trade=None) -> DualOutput:
        """Convert a JournalEntry to dual output."""
        try:
            plan = getattr(journal, "plan", "") or ""
            lesson = getattr(journal, "lesson", "") or ""
            what_well = getattr(journal, "what_went_well", "") or ""
            what_wrong = getattr(journal, "what_went_wrong", "") or ""
            followed = getattr(journal, "followed_plan", None)
            would_again = getattr(journal, "would_take_again", None)
            confidence = getattr(journal, "pre_trade_confidence", None)

            symbol = ""
            if trade:
                symbol = getattr(trade, "symbol", "")

            parts = []
            if symbol:
                parts.append(f"Journal for {symbol} trade")

            if plan:
                parts.append(f"Plan: {plan[:200]}")
            if lesson:
                parts.append(f"Lesson learned: {lesson[:200]}")
            if what_well:
                parts.append(f"What went well: {what_well[:150]}")
            if what_wrong:
                parts.append(f"What went wrong: {what_wrong[:150]}")
            if followed is not None:
                parts.append(f"{'Followed' if followed else 'Did NOT follow'} the plan")
            if would_again is not None:
                parts.append(f"{'Would' if would_again else 'Would NOT'} take this trade again")
            if confidence:
                parts.append(f"Pre-trade confidence: {confidence}/10")

            if not parts:
                parts.append("Empty journal entry")

            text = ". ".join(parts) + "."
            quality = min(1.0, 0.2 * sum([
                bool(plan), bool(lesson), bool(what_well),
                bool(what_wrong), followed is not None,
            ]))

            structured = {
                "symbol": symbol,
                "has_plan": bool(plan),
                "has_lesson": bool(lesson),
                "followed_plan": followed,
                "would_take_again": would_again,
                "pre_trade_confidence": confidence,
            }

            journal_id = str(getattr(journal, "id", ""))
            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.INSIGHT,
                entry_id=journal_id,
                title=f"Journal: {symbol}" if symbol else "Journal entry",
                source="user",
                tags=["journal", symbol] if symbol else ["journal"],
            )
        except Exception as e:
            logger.warning("journal_to_dual failed: %s", e)
            return DualOutput(
                text=f"Journal conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.INSIGHT,
            )

    # ── PatternTemplate → STRATEGY layer (pattern library seed) ──

    def pattern_template_to_dual(self, template) -> DualOutput:
        """
        Convert a PatternTemplate (bootstrap/pattern_library.py) into STRATEGY layer.
        Seeds curated domain knowledge for cold-start.
        """
        try:
            pid = getattr(template, "pattern_id", "")
            name = getattr(template, "name", "")
            category = getattr(template, "category", "")
            description = getattr(template, "description", "")
            aliases = getattr(template, "aliases", [])

            hard_rules = getattr(template, "hard_rules", [])
            soft_scorers = getattr(template, "soft_scorers", [])
            variants = getattr(template, "variants", [])
            failure_modes = getattr(template, "failure_modes", [])
            regime_perf = getattr(template, "regime_performance", {})
            best_regimes = getattr(template, "best_regimes", [])
            worst_regimes = getattr(template, "worst_regimes", [])
            edge_source = getattr(template, "edge_source", "")
            edge_fragility = getattr(template, "edge_fragility", "")
            typical_wr = getattr(template, "typical_win_rate", 0)
            typical_pf = getattr(template, "typical_profit_factor", 0)

            parts = [f"Pattern: {name} ({category})"]
            if aliases:
                parts.append(f"Also known as: {', '.join(aliases)}")
            if description:
                parts.append(description[:400])

            # Hard rules
            if hard_rules:
                rule_texts = [r.get("name", str(r)) for r in hard_rules[:5]]
                parts.append(f"Hard rules: {'; '.join(rule_texts)}")

            # Soft scorers
            if soft_scorers:
                scorer_texts = [s.get("name", str(s)) for s in soft_scorers[:5]]
                parts.append(f"Quality factors: {'; '.join(scorer_texts)}")

            # Variants
            for v in variants[:3]:
                v_name = getattr(v, "name", v) if not isinstance(v, str) else v
                v_desc = getattr(v, "description", "") if not isinstance(v, str) else ""
                parts.append(f"Variant: {v_name}. {v_desc}"[:100])

            # Failure modes
            for fm in failure_modes[:3]:
                fm_name = getattr(fm, "name", fm) if not isinstance(fm, str) else fm
                fm_desc = getattr(fm, "description", "") if not isinstance(fm, str) else ""
                parts.append(f"Failure mode: {fm_name}. {fm_desc}"[:100])

            # Regime
            if best_regimes:
                parts.append(f"Best in: {', '.join(best_regimes)}")
            if worst_regimes:
                parts.append(f"Worst in: {', '.join(worst_regimes)}")

            # Edge
            if edge_source:
                parts.append(f"Edge source: {edge_source}")
            if edge_fragility:
                parts.append(f"Edge fragility: {edge_fragility}")

            # Stats
            if typical_wr:
                parts.append(f"Typical win rate: {typical_wr:.0%}")
            if typical_pf:
                parts.append(f"Typical profit factor: {typical_pf:.1f}")

            text = ". ".join(parts) + "."
            quality = 0.9  # curated knowledge is high quality

            structured = {
                "pattern_id": pid,
                "name": name,
                "category": category,
                "aliases": aliases,
                "hard_rules_count": len(hard_rules),
                "soft_scorers_count": len(soft_scorers),
                "variants_count": len(variants),
                "failure_modes_count": len(failure_modes),
                "best_regimes": best_regimes,
                "worst_regimes": worst_regimes,
                "typical_win_rate": typical_wr,
                "typical_profit_factor": typical_pf,
                "edge_source": edge_source,
                "edge_fragility": edge_fragility,
                "regime_performance": regime_perf,
            }

            return DualOutput(
                text=text,
                structured=structured,
                quality_score=quality,
                layer=MemoryLayer.STRATEGY,
                entry_id=f"pattern_lib_{pid}",
                title=f"Pattern Library: {name}",
                source="system",
                tags=[pid, name, category, "pattern_library"],
            )
        except Exception as e:
            logger.warning("pattern_template_to_dual failed: %s", e)
            return DualOutput(
                text=f"Pattern template conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.STRATEGY,
            )

    # ── Coaching insight (generic dict) → INSIGHT layer ──────────

    def coaching_insight_to_dual(self, insight: dict) -> DualOutput:
        """Convert a coaching insight dict to dual output."""
        try:
            insight_type = insight.get("type", "general")
            text_content = insight.get("text", insight.get("content", ""))
            severity = insight.get("severity", "info")
            strategy = insight.get("strategy", "")
            symbol = insight.get("symbol", "")

            parts = [f"Coaching insight ({insight_type})"]
            if severity != "info":
                parts.append(f"severity: {severity}")
            if strategy:
                parts.append(f"regarding strategy {strategy}")
            if symbol:
                parts.append(f"for {symbol}")
            if text_content:
                parts.append(text_content[:400])

            text = ". ".join(parts) + "."

            structured = {
                "type": insight_type,
                "severity": severity,
                "strategy": strategy,
                "symbol": symbol,
            }

            insight_id = insight.get("id", insight.get("insight_id", ""))
            return DualOutput(
                text=text,
                structured=structured,
                quality_score=0.5,
                layer=MemoryLayer.INSIGHT,
                entry_id=str(insight_id),
                title=f"Coaching: {insight_type}",
                source="system",
                tags=[insight_type, strategy, symbol],
            )
        except Exception as e:
            logger.warning("coaching_insight_to_dual failed: %s", e)
            return DualOutput(
                text=f"Coaching insight conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.INSIGHT,
            )

    # ── Behavioral event dict → INSIGHT layer ────────────────────

    def behavioral_event_to_dual(self, event_data: dict) -> DualOutput:
        """Convert a behavioral event to dual output."""
        try:
            event_type = event_data.get("event_type", event_data.get("type", "unknown"))
            symbol = event_data.get("symbol", "")
            severity = event_data.get("severity", "medium")
            description = event_data.get("description", "")
            trade_id = event_data.get("trade_id", "")

            parts = [f"Behavioral event: {event_type}"]
            if symbol:
                parts.append(f"on {symbol}")
            parts.append(f"severity: {severity}")
            if description:
                parts.append(description[:300])

            text = ". ".join(parts) + "."

            structured = {
                "event_type": event_type,
                "symbol": symbol,
                "severity": severity,
                "trade_id": str(trade_id),
            }

            return DualOutput(
                text=text,
                structured=structured,
                quality_score=0.6,
                layer=MemoryLayer.INSIGHT,
                entry_id=f"beh_{trade_id}_{event_type}",
                title=f"Behavior: {event_type}",
                source="system",
                tags=["behavioral", event_type, symbol],
            )
        except Exception as e:
            logger.warning("behavioral_event_to_dual failed: %s", e)
            return DualOutput(
                text=f"Behavioral event conversion failed: {e}",
                structured={"error": str(e)},
                quality_score=0.0,
                layer=MemoryLayer.INSIGHT,
            )
