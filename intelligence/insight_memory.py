"""
PTIS Insight Memory System
========================

Persistent storage and retrieval of coaching insights and behavioral patterns.

Features:
- Recurring mistake detection and tracking
- Improvement insight persistence
- Behavioral trend analysis
- Coaching recommendation memory
- Insight relevance scoring
- Cross-session learning

Provides intelligence for:
- Personalized coaching
- Alert prioritization
- Progress tracking
- Intervention timing

Design
------
- Event-driven insight capture
- Semantic similarity matching
- Temporal relevance decay
- Confidence-based retrieval
- Cross-trader pattern sharing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
from collections import defaultdict

from core.events import Event, EventType, get_event_bus
from core.models import Trade, DecisionQualityRecord, BehavioralEvent

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Types of coaching insights."""
    RECURRING_MISTAKE = "recurring_mistake"
    IMPROVEMENT_INSIGHT = "improvement_insight"
    BEHAVIORAL_TREND = "behavioral_trend"
    PROCESS_VIOLATION = "process_violation"
    STRENGTH_IDENTIFIED = "strength_identified"
    INTERVENTION_RECOMMENDATION = "intervention_recommendation"


class InsightSeverity(str, Enum):
    """Severity levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InsightStatus(str, Enum):
    """Status of insights in memory."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ARCHIVED = "archived"
    REINFORCED = "reinforced"


@dataclass
class CoachingInsight:
    """Persistent coaching insight with metadata."""
    
    insight_id: str
    trader_id: str
    insight_type: InsightType
    severity: InsightSeverity
    status: InsightStatus
    
    # Content
    title: str
    description: str
    category: str  # risk_management, discipline, emotional, strategy
    tags: List[str] = field(default_factory=list)
    
    # Evidence and context
    source_events: List[str] = field(default_factory=list)  # Event IDs that generated this
    supporting_trades: List[str] = field(default_factory=list)  # Trade IDs
    confidence_score: float = 0.0  # 0-1
    relevance_score: float = 1.0  # 0-1, decays over time
    
    # Temporal data
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reinforced: datetime = field(default_factory=datetime.utcnow)
    reinforcement_count: int = 0
    last_triggered: Optional[datetime] = None
    
    # Intervention data
    coaching_actions: List[str] = field(default_factory=list)
    intervention_success: Optional[float] = None  # 0-1 if measured
    
    # Resolution tracking
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "insight_id": self.insight_id,
            "trader_id": self.trader_id,
            "insight_type": self.insight_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "source_events": self.source_events,
            "supporting_trades": self.supporting_trades,
            "confidence_score": round(self.confidence_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "reinforcement_count": self.reinforcement_count,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "coaching_actions": self.coaching_actions,
            "intervention_success": self.intervention_success,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class InsightMemoryConfig:
    """Configuration for insight memory system."""
    
    # Storage thresholds
    max_insights_per_trader: int = 1000
    relevance_decay_days: int = 30
    reinforcement_threshold: float = 0.7
    
    # Matching thresholds
    similarity_threshold: float = 0.8
    category_matching: bool = True
    tag_matching: bool = True
    
    # Prioritization weights
    severity_weight: float = 0.4
    recency_weight: float = 0.3
    frequency_weight: float = 0.3
    
    # Auto-categorization
    auto_categorize: bool = True
    auto_tag: bool = True


class InsightMatcher:
    """Matches new insights against existing memory."""
    
    def __init__(self, config: InsightMemoryConfig):
        self.cfg = config
    
    def find_similar_insights(
        self, 
        new_insight: CoachingInsight, 
        existing_insights: List[CoachingInsight]
    ) -> List[Tuple[CoachingInsight, float]]:
        """Find similar existing insights."""
        similar = []
        
        for existing in existing_insights:
            similarity = self._calculate_similarity(new_insight, existing)
            if similarity >= self.cfg.similarity_threshold:
                similar.append((existing, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def _calculate_similarity(
        self, 
        insight1: CoachingInsight, 
        insight2: CoachingInsight
    ) -> float:
        """Calculate similarity between two insights."""
        score = 0.0
        
        # Category matching
        if self.cfg.category_matching and insight1.category == insight2.category:
            score += 0.4
        
        # Tag overlap
        if self.cfg.tag_matching:
            tags1 = set(insight1.tags)
            tags2 = set(insight2.tags)
            if tags1 and tags2:
                overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
                score += overlap * 0.3
        
        # Text similarity (simplified)
        title_sim = self._text_similarity(insight1.title, insight2.title)
        desc_sim = self._text_similarity(insight1.description, insight2.description)
        score += (title_sim + desc_sim) * 0.15
        
        # Severity matching
        if insight1.severity == insight2.severity:
            score += 0.15
        
        return min(score, 1.0)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class InsightMemory:
    """
    Main insight memory system.
    
    Stores, retrieves, and manages coaching insights over time.
    Provides persistent learning and adaptive recommendations.
    """
    
    def __init__(self, config: Optional[InsightMemoryConfig] = None):
        self.cfg = config or InsightMemoryConfig()
        self._event_bus = get_event_bus()
        self.matcher = InsightMatcher(self.cfg)
        
        # Insight storage
        self.insights: Dict[str, CoachingInsight] = {}
        self.trader_insights: Dict[str, List[str]] = defaultdict(list)
        
        # Subscribe to events
        self._setup_event_listeners()
        
        logger.info("InsightMemory initialized")
    
    def _setup_event_listeners(self) -> None:
        """Subscribe to events for insight generation."""
        self._event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        self._event_bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_behavioral_event)
        self._event_bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_decision_quality)
    
    async def store_insight(self, insight: CoachingInsight) -> str:
        """
        Store a new coaching insight.
        
        Parameters
        ----------
        insight : CoachingInsight
            Insight to store
        
        Returns
        -------
        str
            Insight ID
        """
        # Check for similar existing insights
        existing = self.get_trader_insights(insight.trader_id)
        similar = self.matcher.find_similar_insights(insight, existing)
        
        if similar:
            # Reinforce existing insight instead of creating duplicate
            existing_insight, similarity = similar[0]
            await self.reinforce_insight(existing_insight.insight_id, insight)
            logger.debug(f"Reinforced existing insight {existing_insight.insight_id}")
            return existing_insight.insight_id
        
        # Store new insight
        self.insights[insight.insight_id] = insight
        self.trader_insights[insight.trader_id].append(insight.insight_id)
        
        # Cleanup old insights if needed
        await self._cleanup_old_insights(insight.trader_id)
        
        logger.info(f"Stored new insight {insight.insight_id}: {insight.title}")
        
        # Publish insight stored event
        self._event_bus.publish(Event(
            event_type=EventType.BEHAVIORAL_EVENT,
            source="insight_memory",
            data={
                "event_type": "insight_stored",
                "insight_id": insight.insight_id,
                "trader_id": insight.trader_id,
                "insight_type": insight.insight_type.value,
                "severity": insight.severity.value,
                "title": insight.title,
            }
        ))
        
        return insight.insight_id
    
    async def reinforce_insight(
        self, 
        insight_id: str, 
        reinforcing_insight: CoachingInsight
    ) -> None:
        """Reinforce an existing insight with new evidence."""
        if insight_id not in self.insights:
            return
        
        insight = self.insights[insight_id]
        
        # Update reinforcement data
        insight.last_reinforced = datetime.utcnow()
        insight.reinforcement_count += 1
        
        # Merge supporting evidence
        insight.source_events.extend(reinforcing_insight.source_events)
        insight.supporting_trades.extend(reinforcing_insight.supporting_trades)
        
        # Update confidence based on reinforcement
        insight.confidence_score = min(1.0, insight.confidence_score + 0.1)
        insight.relevance_score = 1.0  # Reset relevance on reinforcement
        
        # Merge coaching actions if new ones provided
        for action in reinforcing_insight.coaching_actions:
            if action not in insight.coaching_actions:
                insight.coaching_actions.append(action)
        
        logger.debug(f"Reinforced insight {insight_id}")
    
    def get_relevant_insights(
        self, 
        trader_id: str, 
        context: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[CoachingInsight]:
        """
        Get relevant insights for a trader.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        context : Optional[Dict[str, Any]]
            Current context for filtering
        limit : int
            Maximum number of insights to return
        
        Returns
        -------
        List[CoachingInsight]
            Relevant insights, sorted by priority
        """
        trader_insight_ids = self.trader_insights.get(trader_id, [])
        insights = [self.insights[iid] for iid in trader_insight_ids if iid in self.insights]
        
        # Filter by status
        active_insights = [
            insight for insight in insights 
            if insight.status in [InsightStatus.ACTIVE, InsightStatus.REINFORCED]
        ]
        
        # Apply context filtering if provided
        if context:
            active_insights = self._filter_by_context(active_insights, context)
        
        # Update relevance scores
        self._update_relevance_scores(active_insights)
        
        # Sort by priority
        prioritized = self._prioritize_insights(active_insights)
        
        return prioritized[:limit]
    
    def get_insight_by_id(self, insight_id: str) -> Optional[CoachingInsight]:
        """Get a specific insight by ID."""
        return self.insights.get(insight_id)
    
    def get_trader_insights(self, trader_id: str) -> List[CoachingInsight]:
        """Get all insights for a trader."""
        trader_insight_ids = self.trader_insights.get(trader_id, [])
        return [self.insights[iid] for iid in trader_insight_ids if iid in self.insights]
    
    async def resolve_insight(
        self, 
        insight_id: str, 
        resolution_notes: str
    ) -> None:
        """Mark an insight as resolved."""
        if insight_id not in self.insights:
            return
        
        insight = self.insights[insight_id]
        insight.status = InsightStatus.RESOLVED
        insight.resolution_notes = resolution_notes
        insight.resolved_at = datetime.utcnow()
        
        logger.info(f"Resolved insight {insight_id}: {insight.title}")
        
        # Publish resolution event
        self._event_bus.publish(Event(
            event_type=EventType.BEHAVIORAL_EVENT,
            source="insight_memory",
            data={
                "event_type": "insight_resolved",
                "insight_id": insight_id,
                "trader_id": insight.trader_id,
                "resolution_notes": resolution_notes,
            }
        ))
    
    def get_insight_summary(self, trader_id: str) -> Dict[str, Any]:
        """Get summary of insights for a trader."""
        insights = self.get_trader_insights(trader_id)
        
        if not insights:
            return {
                "total_insights": 0,
                "active_insights": 0,
                "resolved_insights": 0,
                "by_category": {},
                "by_severity": {},
                "top_priorities": []
            }
        
        # Group by category and severity
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        
        active_count = 0
        resolved_count = 0
        
        for insight in insights:
            by_category[insight.category] += 1
            by_severity[insight.severity.value] += 1
            
            if insight.status == InsightStatus.ACTIVE:
                active_count += 1
            elif insight.status == InsightStatus.RESOLVED:
                resolved_count += 1
        
        # Get top priority insights
        active_insights = [
            insight for insight in insights 
            if insight.status in [InsightStatus.ACTIVE, InsightStatus.REINFORCED]
        ]
        prioritized = self._prioritize_insights(active_insights)
        top_priorities = prioritized[:5]
        
        return {
            "total_insights": len(insights),
            "active_insights": active_count,
            "resolved_insights": resolved_count,
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "top_priorities": [
                {
                    "insight_id": insight.insight_id,
                    "title": insight.title,
                    "severity": insight.severity.value,
                    "priority_score": self._calculate_priority_score(insight)
                }
                for insight in top_priorities
            ]
        }
    
    def _filter_by_context(
        self, 
        insights: List[CoachingInsight], 
        context: Dict[str, Any]
    ) -> List[CoachingInsight]:
        """Filter insights based on current context."""
        filtered = insights
        
        # Filter by category if specified
        if "categories" in context:
            allowed_categories = set(context["categories"])
            filtered = [
                insight for insight in filtered 
                if insight.category in allowed_categories
            ]
        
        # Filter by severity if specified
        if "min_severity" in context:
            min_severity = context["min_severity"]
            severity_order = ["low", "medium", "high", "critical"]
            min_index = severity_order.index(min_severity)
            
            filtered = [
                insight for insight in filtered 
                if severity_order.index(insight.severity.value) >= min_index
            ]
        
        return filtered
    
    def _update_relevance_scores(self, insights: List[CoachingInsight]) -> None:
        """Update relevance scores based on recency and reinforcement."""
        now = datetime.utcnow()
        
        for insight in insights:
            # Time decay
            days_since_created = (now - insight.created_at).days
            time_factor = max(0.1, 1.0 - (days_since_created / self.cfg.relevance_decay_days))
            
            # Reinforcement boost
            reinforcement_factor = 1.0 + (insight.reinforcement_count * 0.1)
            
            # Combined relevance score
            insight.relevance_score = time_factor * reinforcement_factor * 0.5 + insight.confidence_score * 0.5
    
    def _prioritize_insights(self, insights: List[CoachingInsight]) -> List[CoachingInsight]:
        """Prioritize insights based on multiple factors."""
        def priority_score(insight):
            return self._calculate_priority_score(insight)
        
        return sorted(insights, key=priority_score, reverse=True)
    
    def _calculate_priority_score(self, insight: CoachingInsight) -> float:
        """Calculate overall priority score for an insight."""
        severity_weight = {
            InsightSeverity.CRITICAL: 4.0,
            InsightSeverity.HIGH: 3.0,
            InsightSeverity.MEDIUM: 2.0,
            InsightSeverity.LOW: 1.0,
        }
        
        severity_score = severity_weight.get(insight.severity, 1.0)
        recency_score = insight.relevance_score
        frequency_score = min(1.0, insight.reinforcement_count / 10.0)
        
        return (
            severity_score * self.cfg.severity_weight +
            recency_score * self.cfg.recency_weight +
            frequency_score * self.cfg.frequency_weight
        )
    
    async def _cleanup_old_insights(self, trader_id: str) -> None:
        """Remove old insights to maintain storage limits."""
        trader_insight_ids = self.trader_insights.get(trader_id, [])
        
        if len(trader_insight_ids) <= self.cfg.max_insights_per_trader:
            return
        
        # Sort by relevance and keep the best ones
        insights = [self.insights[iid] for iid in trader_insight_ids if iid in self.insights]
        insights.sort(key=lambda i: i.relevance_score, reverse=True)
        
        # Keep only the top insights
        keep_count = self.cfg.max_insights_per_trader
        kept_insights = insights[:keep_count]
        removed_insights = insights[keep_count:]
        
        # Update storage
        self.trader_insights[trader_id] = [insight.insight_id for insight in kept_insights]
        
        for insight in removed_insights:
            if insight.insight_id in self.insights:
                del self.insights[insight.insight_id]
        
        if removed_insights:
            logger.debug(f"Cleaned up {len(removed_insights)} old insights for {trader_id}")
    
    # Event handlers
    def _on_trade_closed(self, event: Event) -> None:
        """Generate insights from trade close events."""
        if event.trade_id and event.data:
            self._generate_trade_insights(event)
    
    def _on_behavioral_event(self, event: Event) -> None:
        """Generate insights from behavioral events."""
        event_type = event.data.get("event_type", "")
        
        if event_type == "decision_quality_evaluated":
            self._generate_dq_insights(event)
        elif event_type in ["override_detected", "behavioral_flag"]:
            self._generate_behavioral_insights(event)
    
    def _on_decision_quality(self, event: Event) -> None:
        """Generate insights from decision quality evaluations."""
        event_type = event.data.get("event_type", "")
        if event_type == "decision_quality_evaluated":
            self._generate_dq_insights(event)
    
    def _generate_trade_insights(self, event: Event) -> None:
        """Generate insights from trade data."""
        # This would analyze trade patterns and generate insights
        # For now, just log the event
        logger.debug(f"Generating trade insights for {event.trade_id}")
    
    def _generate_dq_insights(self, event: Event) -> None:
        """Generate insights from decision quality evaluations."""
        data = event.data
        
        # Generate insights for low decision quality
        if data.get("composite_score", 100) < 50:
            insight = CoachingInsight(
                insight_id=f"dq_low_{event.trade_id}",
                trader_id=data.get("trader_id") or str(event.session_id or "unknown"),
                insight_type=InsightType.PROCESS_VIOLATION,
                severity=InsightSeverity.HIGH if data["composite_score"] < 40 else InsightSeverity.MEDIUM,
                status=InsightStatus.ACTIVE,
                title="Low Decision Quality Detected",
                description=f"Decision quality score of {data['composite_score']:.1f} indicates process issues",
                category="discipline",
                tags=["decision_quality", "process_violation"],
                source_events=[data.get("evaluation_id", "")],
                supporting_trades=[str(event.trade_id)] if event.trade_id else [],
                confidence_score=0.8,
                coaching_actions=[
                    "Review trade execution process",
                    "Focus on weakest dimensions",
                    "Consider reducing position size"
                ]
            )
            
            # Store insight asynchronously
            import asyncio
            asyncio.create_task(self.store_insight(insight))
    
    def _generate_behavioral_insights(self, event: Event) -> None:
        """Generate insights from behavioral events."""
        # This would analyze behavioral patterns and generate insights
        logger.debug(f"Generating behavioral insights for event: {event.data.get('event_type', '')}")


# Global instance
_insight_memory: Optional[InsightMemory] = None

def get_insight_memory() -> InsightMemory:
    """Get global insight memory instance."""
    global _insight_memory
    if _insight_memory is None:
        _insight_memory = InsightMemory()
    return _insight_memory


# Backward-compatible alias (was in insight_service.py)
get_insight_service = get_insight_memory
