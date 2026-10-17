"""
PTIS Pattern Learning Service
===========================

Service layer for behavioral pattern learning and management.

Provides a clean API for:
- Pattern detection and learning
- Pattern retrieval and caching
- Event-driven pattern maintenance
- Integration with coaching agents and alerts

Design
------
- Service layer pattern with clean interfaces
- Async database operations
- Event-driven updates
- Memory caching for performance
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as DBSession

from core.events import Event, EventType, get_event_bus
from core.models import Trade, BehavioralEvent, DecisionQualityRecord
from behavioral.pattern_learning import (
    BehavioralPatternLearner, get_pattern_learner, PatternType, PatternSeverity
)

logger = logging.getLogger(__name__)


class PatternLearningService:
    """
    Service for managing behavioral pattern learning.
    
    Provides high-level interface for pattern detection,
    learning, and maintenance. Handles caching, event
    subscriptions, and database operations.
    """
    
    def __init__(self, cache_ttl_hours: int = 2):
        self._learner = get_pattern_learner()
        self._event_bus = get_event_bus()
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._pattern_cache: Dict[str, tuple[List, datetime]] = {}
        
        # Subscribe to relevant events for cache invalidation
        self._setup_cache_invalidation()
        
        logger.info("PatternLearningService initialized")
    
    def _setup_cache_invalidation(self) -> None:
        """Setup event listeners for cache invalidation."""
        # Invalidate cache when new trades close
        self._event_bus.subscribe(EventType.TRADE_CLOSED, self._invalidate_trader_cache)
        
        # Invalidate on behavioral events
        self._event_bus.subscribe(EventType.BEHAVIORAL_EVENT, self._on_behavioral_event)
    
    async def learn_patterns(
        self, 
        trader_id: str, 
        db: AsyncSession,
        force_refresh: bool = False
    ) -> List[Any]:
        """
        Learn behavioral patterns for a trader.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        force_refresh : bool
            Force pattern refresh even if cached
        
        Returns
        -------
        List[Any]
            Detected behavioral patterns
        """
        # Check cache first
        if not force_refresh:
            cached_patterns, cached_at = self._pattern_cache.get(trader_id, ([], None))
            if cached_at and (datetime.utcnow() - cached_at) < self._cache_ttl:
                logger.debug(f"Returning cached patterns for {trader_id}")
                return cached_patterns
        
        # Fetch data for pattern learning
        trades = await self._fetch_trades(db, trader_id)
        behavioral_events = await self._fetch_behavioral_events(db, trader_id)
        dq_records = await self._fetch_decision_quality_records(db, trader_id)
        
        # Learn patterns
        patterns = await self._learner.learn_patterns(
            trader_id, trades, behavioral_events, dq_records
        )
        
        # Cache result
        self._pattern_cache[trader_id] = (patterns, datetime.utcnow())
        
        return patterns
    
    async def get_active_patterns(
        self, 
        trader_id: str, 
        db: AsyncSession
    ) -> List[Any]:
        """
        Get currently active patterns for a trader.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        
        Returns
        -------
        List[Any]
            Active behavioral patterns
        """
        return self._learner.get_active_patterns(trader_id)
    
    async def get_pattern_summary(
        self, 
        trader_id: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get summary of patterns for a trader.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        
        Returns
        -------
        Dict[str, Any]
            Pattern summary statistics
        """
        return self._learner.get_pattern_summary(trader_id)
    
    async def get_coaching_context(
        self, 
        trader_id: str, 
        db: AsyncSession
    ) -> str:
        """
        Generate coaching context string from patterns.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        
        Returns
        -------
        str
            Coaching context narrative
        """
        patterns = await self.get_active_patterns(trader_id, db)
        
        if not patterns:
            return "No significant behavioral patterns detected."
        
        context_parts = [
            f"Behavioral Pattern Analysis ({len(patterns)} active patterns):",
            f""
        ]
        
        # Group patterns by severity
        critical_patterns = [p for p in patterns if p.severity == PatternSeverity.CRITICAL]
        high_patterns = [p for p in patterns if p.severity == PatternSeverity.HIGH]
        
        if critical_patterns:
            context_parts.append("🚨 CRITICAL PATTERNS:")
            for pattern in critical_patterns[:3]:
                context_parts.append(f"• {pattern.title}: {pattern.description}")
            context_parts.append("")
        
        if high_patterns:
            context_parts.append("⚠️ HIGH PRIORITY PATTERNS:")
            for pattern in high_patterns[:3]:
                context_parts.append(f"• {pattern.title}: {pattern.description}")
            context_parts.append("")
        
        # Top intervention strategies
        all_interventions = []
        for pattern in patterns:
            all_interventions.extend(pattern.intervention_strategies)
        
        if all_interventions:
            from collections import Counter
            intervention_counts = Counter(all_interventions)
            top_interventions = intervention_counts.most_common(3)
            
            context_parts.append("🎯 RECOMMENDED INTERVENTIONS:")
            for intervention, count in top_interventions:
                context_parts.append(f"• {intervention} (triggered {count} times)")
        
        return "\n".join(context_parts)
    
    async def check_pattern_alerts(
        self, 
        trader_id: str, 
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """
        Check if trader meets conditions for pattern-based alerts.
        
        Parameters
        ----------
        trader_id : str
            Trader identifier
        db : AsyncSession
            Database session
        
        Returns
        -------
        List[Dict[str, Any]]
            List of pattern-based alerts
        """
        patterns = await self.get_active_patterns(trader_id, db)
        alerts = []
        
        # Critical pattern alerts
        critical_patterns = [p for p in patterns if p.severity == PatternSeverity.CRITICAL]
        for pattern in critical_patterns:
            alerts.append({
                "type": "critical_pattern",
                "severity": "critical",
                "message": f"Critical behavioral pattern: {pattern.title}",
                "suggested_action": "Immediate intervention recommended",
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "interventions": pattern.intervention_strategies
            })
        
        # High-frequency pattern alerts
        high_frequency_patterns = [
            p for p in patterns 
            if p.frequency > 0.5 and p.severity in [PatternSeverity.HIGH, PatternSeverity.MEDIUM]
        ]
        for pattern in high_frequency_patterns:
            alerts.append({
                "type": "high_frequency_pattern",
                "severity": "high",
                "message": f"Frequent pattern: {pattern.title} ({pattern.frequency:.0%} occurrence)",
                "suggested_action": "Pattern-specific intervention needed",
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "frequency": pattern.frequency
            })
        
        # High-impact pattern alerts
        high_impact_patterns = [
            p for p in patterns 
            if p.impact_score > 0.7 and p.confidence.value in ["strong", "very_strong"]
        ]
        for pattern in high_impact_patterns:
            alerts.append({
                "type": "high_impact_pattern",
                "severity": "medium",
                "message": f"High-impact pattern: {pattern.title}",
                "suggested_action": "Focus coaching on this pattern",
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "impact_score": pattern.impact_score
            })
        
        return alerts
    
    def invalidate_cache(self, trader_id: str) -> None:
        """Invalidate cached patterns for a trader."""
        if trader_id in self._pattern_cache:
            del self._pattern_cache[trader_id]
            logger.debug(f"Invalidated pattern cache for {trader_id}")
    
    def clear_all_caches(self) -> None:
        """Clear all pattern caches."""
        self._pattern_cache.clear()
        logger.info("Cleared all pattern caches")
    
    # Event handlers
    def _invalidate_trader_cache(self, event: Event) -> None:
        """Invalidate cache when trade closes."""
        # Extract trader_id from trade (would need database lookup in production)
        # For now, clear all caches when any trade closes
        self.clear_all_caches()
    
    def _on_behavioral_event(self, event: Event) -> None:
        """Handle behavioral events for cache invalidation."""
        event_type = event.data.get("event_type", "")
        
        # Invalidate cache on significant events
        if event_type in ["pattern_detected", "behavioral_flag"]:
            trader_id = event.data.get("trader_id")
            if trader_id:
                self.invalidate_cache(trader_id)
    
    # Data fetchers (simplified for demo)
    async def _fetch_trades(self, db: AsyncSession, trader_id: str) -> List[Trade]:
        """Fetch trades for pattern learning."""
        # Implementation would query database
        return []
    
    async def _fetch_behavioral_events(
        self, 
        db: AsyncSession, 
        trader_id: str
    ) -> List[BehavioralEvent]:
        """Fetch behavioral events."""
        # Implementation would query database  
        return []
    
    async def _fetch_decision_quality_records(
        self, 
        db: AsyncSession, 
        trader_id: str
    ) -> List[DecisionQualityRecord]:
        """Fetch decision quality records."""
        # Implementation would query database
        return []


# Global service instance
_pattern_service: Optional[PatternLearningService] = None

def get_pattern_service() -> PatternLearningService:
    """Get global pattern learning service instance."""
    global _pattern_service
    if _pattern_service is None:
        _pattern_service = PatternLearningService()
    return _pattern_service


# Convenience functions for common operations
async def get_behavioral_patterns(trader_id: str, db: AsyncSession) -> List[Any]:
    """Convenience function to get behavioral patterns."""
    service = get_pattern_service()
    return await service.get_active_patterns(trader_id, db)


async def get_pattern_coaching_context(trader_id: str, db: AsyncSession) -> str:
    """Convenience function to get pattern coaching context."""
    service = get_pattern_service()
    return await service.get_coaching_context(trader_id, db)


async def check_pattern_alerts(trader_id: str, db: AsyncSession) -> List[Dict[str, Any]]:
    """Convenience function to check pattern alerts."""
    service = get_pattern_service()
    return await service.check_pattern_alerts(trader_id, db)
