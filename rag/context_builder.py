# rag/context_builder.py
"""
Context Builder — Compresses retrieved chunks BEFORE LLM injection.

Key for cost: 15 similar trades → "15 trades: 67% WR, +1.2R avg" (NOT individual trades).
Includes consistency check (GAP 1) and sample size transparency.

Architecture Rules:
  - Consistency check: split last N trades into 2 halves, compare WR.
    Δ<10% = HIGH, 10-25% = MED, >25% = LOW. Downgrades trust, never blocks.
  - Sample sizes always included so LLM and user see data backing.
  - Compression: merge similar chunks, aggregate stats from structured data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.knowledge_types import MultiLayerContext, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class StructuredContext:
    """Compressed context ready for LLM prompt injection."""
    query: str
    strategy_context: str = ""      # compressed strategy rules
    pattern_context: str = ""       # compressed pattern info
    trade_context: str = ""         # compressed trade stats (not 50 trades, just stats)
    insight_context: str = ""       # compressed coaching insights
    confidence_score: float = 0.0
    sample_sizes: dict = field(default_factory=dict)  # per-layer sample counts
    consistency_level: str = "unknown"  # "high" | "medium" | "low" | "unknown"
    unknown_pattern_warning: bool = False
    _source_chunks: list = field(default_factory=list)

    def format_for_prompt(self, max_tokens: int = 2000) -> str:
        """Format into a string suitable for LLM system prompt injection."""
        sections = []

        if self.strategy_context:
            sections.append(f"=== STRATEGY KNOWLEDGE ===\n{self.strategy_context}")
        if self.trade_context:
            sections.append(f"=== TRADE HISTORY ===\n{self.trade_context}")
        if self.pattern_context:
            sections.append(f"=== PATTERN KNOWLEDGE ===\n{self.pattern_context}")
        if self.insight_context:
            sections.append(f"=== BEHAVIORAL INSIGHTS ===\n{self.insight_context}")

        if not sections:
            return "No relevant personal knowledge found."

        # Add metadata footer
        meta_parts = []
        for layer, count in self.sample_sizes.items():
            if not layer.startswith("_") and count > 0:
                meta_parts.append(f"{layer}: {count}")
        if meta_parts:
            sections.append(f"[Data backing: {', '.join(meta_parts)}]")
        if self.consistency_level != "unknown":
            sections.append(f"[Data consistency: {self.consistency_level}]")
        if self.unknown_pattern_warning:
            sections.append("[Warning: Low pattern familiarity — limited historical matches]")

        full = "\n\n".join(sections)

        # Rough token limit enforcement
        est_tokens = len(full) // 4
        if est_tokens > max_tokens:
            char_limit = max_tokens * 4
            full = full[:char_limit] + "\n[...truncated for token budget]"

        return full

    def get_grounding_sources(self) -> List[Dict[str, Any]]:
        """Return source metadata for citation/grounding."""
        return [
            {
                "chunk_id": rc.chunk_id,
                "score": rc.score,
                "layer": rc.chunk.metadata.get("_memory_layer", ""),
                "title": rc.title,
            }
            for rc in self._source_chunks
        ]


class ContextBuilder:
    """
    Compresses retrieved chunks into structured context.

    Merges similar chunks, aggregates trade stats, and applies
    consistency checks before LLM consumption.
    """

    def __init__(self, db_factory=None):
        self._db_factory = db_factory

    def build(
        self,
        retrieval: MultiLayerContext,
        max_strategy_tokens: int = 500,
        max_trade_tokens: int = 500,
        max_pattern_tokens: int = 300,
        max_insight_tokens: int = 300,
    ) -> StructuredContext:
        """Build compressed context from multi-layer retrieval."""

        strategy_chunks = retrieval.chunks_by_layer.get("strategy_memory", [])
        trade_chunks = retrieval.chunks_by_layer.get("trade_memory", [])
        pattern_chunks = retrieval.chunks_by_layer.get("pattern_memory", [])
        insight_chunks = retrieval.chunks_by_layer.get("insight_memory", [])

        # Compress each layer
        strategy_text = self._compress_strategy(strategy_chunks, max_strategy_tokens)
        trade_text = self._compress_trades(trade_chunks, max_trade_tokens)
        pattern_text = self._compress_patterns(pattern_chunks, max_pattern_tokens)
        insight_text = self._compress_insights(insight_chunks, max_insight_tokens)

        # Consistency check on trade data (GAP 1)
        consistency = self._compute_consistency(trade_chunks)

        # Unknown pattern check (GAP 2)
        unknown_warning = retrieval.sample_sizes.get("_unknown_pattern", 0) > 0

        return StructuredContext(
            query=retrieval.query,
            strategy_context=strategy_text,
            pattern_context=pattern_text,
            trade_context=trade_text,
            insight_context=insight_text,
            confidence_score=retrieval.confidence_score,
            sample_sizes=retrieval.sample_sizes,
            consistency_level=consistency,
            unknown_pattern_warning=unknown_warning,
            _source_chunks=retrieval.all_chunks,
        )

    # ── Layer compressors ─────────────────────────────────────────

    def _compress_strategy(self, chunks: List[RetrievedChunk], max_tokens: int) -> str:
        """Compress strategy chunks — keep highest-scoring, merge overlapping."""
        if not chunks:
            return ""
        # Sort by score, take top 3
        sorted_chunks = sorted(chunks, key=lambda c: -c.score)[:3]
        parts = []
        for rc in sorted_chunks:
            text = rc.text[:max_tokens * 2]  # rough char limit
            parts.append(text)
        full = "\n---\n".join(parts)
        # Enforce token budget
        return full[: max_tokens * 4]

    def _compress_trades(self, chunks: List[RetrievedChunk], max_tokens: int) -> str:
        """
        Compress trade chunks into aggregate stats.
        Instead of N individual trades, produce: "N trades: X% WR, avg R, best regime".
        """
        if not chunks:
            return ""

        # Try to extract structured data from ingestion log
        stats = self._aggregate_trade_stats(chunks)
        if stats:
            return stats

        # Fallback: summarize from text
        n = len(chunks)
        top_3 = sorted(chunks, key=lambda c: -c.score)[:3]
        parts = [f"{n} relevant trades found:"]
        for rc in top_3:
            parts.append(f"  - {rc.title} (relevance: {rc.score:.2f})")
            # Include first 100 chars of text
            parts.append(f"    {rc.text[:100]}")
        return "\n".join(parts)[: max_tokens * 4]

    def _compress_patterns(self, chunks: List[RetrievedChunk], max_tokens: int) -> str:
        """Compress pattern chunks."""
        if not chunks:
            return ""
        sorted_chunks = sorted(chunks, key=lambda c: -c.score)[:3]
        parts = []
        for rc in sorted_chunks:
            parts.append(f"[{rc.score:.2f}] {rc.text[:200]}")
        return "\n".join(parts)[: max_tokens * 4]

    def _compress_insights(self, chunks: List[RetrievedChunk], max_tokens: int) -> str:
        """Compress insight chunks — keep highest-confidence version only."""
        if not chunks:
            return ""
        # Deduplicate by title (keep highest score)
        seen_titles = {}
        for rc in chunks:
            title = rc.title
            if title not in seen_titles or rc.score > seen_titles[title].score:
                seen_titles[title] = rc
        unique = sorted(seen_titles.values(), key=lambda c: -c.score)[:4]
        parts = [rc.text[:200] for rc in unique]
        return "\n".join(parts)[: max_tokens * 4]

    # ── Trade stats aggregation (DUAL pattern payoff) ─────────────

    def _aggregate_trade_stats(self, chunks: List[RetrievedChunk]) -> str:
        """
        Look up structured_data from ingestion log for computation.
        This is the "retrieval uses embeddings, computation uses structured data" rule.
        """
        try:
            if not self._db_factory:
                return ""

            db = self._db_factory()
            if hasattr(db, "__next__"):
                db = next(db)

            try:
                from rag.ingestion_store import IngestionRecord

                # Get source IDs from chunks
                entry_ids = [c.chunk.entry_id for c in chunks if c.chunk.entry_id]
                if not entry_ids:
                    return ""

                records = (
                    db.query(IngestionRecord)
                    .filter(
                        IngestionRecord.source_id.in_(entry_ids),
                        IngestionRecord.layer == "trade_memory",
                    )
                    .all()
                )

                if not records:
                    return ""

                # Aggregate stats from structured_data
                wins = 0
                losses = 0
                total_r = 0.0
                r_count = 0
                regimes = {}
                strategies = {}

                for rec in records:
                    data = rec.structured_data or {}
                    r_mult = data.get("r_multiple")
                    if r_mult is not None:
                        if r_mult > 0:
                            wins += 1
                        else:
                            losses += 1
                        total_r += r_mult
                        r_count += 1

                    regime = data.get("regime", "unknown")
                    regimes[regime] = regimes.get(regime, 0) + 1

                    strategy = data.get("strategy", "")
                    if strategy:
                        strategies[strategy] = strategies.get(strategy, 0) + 1

                total = wins + losses
                if total == 0:
                    return ""

                wr = wins / total * 100
                avg_r = total_r / r_count if r_count else 0
                top_regime = max(regimes.items(), key=lambda x: x[1])[0] if regimes else "?"
                top_strategy = max(strategies.items(), key=lambda x: x[1])[0] if strategies else "?"

                parts = [
                    f"{total} similar trades: {wr:.0f}% win rate, avg R = {avg_r:+.2f}",
                    f"Most common regime: {top_regime}",
                ]
                if top_strategy:
                    parts.append(f"Most common strategy: {top_strategy}")
                parts.append(f"(Based on {total} trades)")

                return "\n".join(parts)
            finally:
                db.close()
        except Exception as e:
            logger.debug("Trade stats aggregation failed: %s", e)
            return ""

    # ── Consistency check (GAP 1) ─────────────────────────────────

    def _compute_consistency(self, trade_chunks: List[RetrievedChunk]) -> str:
        """
        Split-half consistency check on trade data.
        Split last N trades into 2 halves, compare win rates.
        Δ<10% = HIGH, 10-25% = MED, >25% = LOW.
        """
        if len(trade_chunks) < 6:
            return "unknown"  # not enough data

        try:
            # Try structured data first
            r_multiples = []
            for rc in trade_chunks:
                # Check metadata for r_multiple
                meta = rc.chunk.metadata
                r_str = meta.get("r_multiple", None)
                if r_str is not None:
                    try:
                        r_multiples.append(float(r_str))
                    except (ValueError, TypeError):
                        pass

            # If not enough from metadata, try ingestion log
            if len(r_multiples) < 6 and self._db_factory:
                r_multiples = self._get_r_multiples_from_db(trade_chunks)

            if len(r_multiples) < 6:
                return "unknown"

            # Split into two halves
            mid = len(r_multiples) // 2
            first_half = r_multiples[:mid]
            second_half = r_multiples[mid:]

            wr1 = sum(1 for r in first_half if r > 0) / len(first_half) if first_half else 0
            wr2 = sum(1 for r in second_half if r > 0) / len(second_half) if second_half else 0

            delta = abs(wr1 - wr2)

            if delta < 0.10:
                return "high"
            elif delta < 0.25:
                return "medium"
            else:
                return "low"
        except Exception as e:
            logger.debug("Consistency check failed: %s", e)
            return "unknown"

    def _get_r_multiples_from_db(self, chunks: List[RetrievedChunk]) -> List[float]:
        """Fetch r_multiples from ingestion log structured_data."""
        try:
            db = self._db_factory()
            if hasattr(db, "__next__"):
                db = next(db)
            try:
                from rag.ingestion_store import IngestionRecord
                entry_ids = [c.chunk.entry_id for c in chunks if c.chunk.entry_id]
                records = (
                    db.query(IngestionRecord)
                    .filter(
                        IngestionRecord.source_id.in_(entry_ids),
                        IngestionRecord.layer == "trade_memory",
                    )
                    .all()
                )
                r_mults = []
                for rec in records:
                    data = rec.structured_data or {}
                    r = data.get("r_multiple")
                    if r is not None:
                        r_mults.append(float(r))
                return r_mults
            finally:
                db.close()
        except Exception:
            return []
