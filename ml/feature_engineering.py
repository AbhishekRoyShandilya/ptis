# feature_engineering.py
# ml/feature_engineering.py
"""
Feature extraction from trade data for ML models.
"""

from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
from core.models import Trade


class TradeFeatureExtractor:
    """Extract features from trade data for ML consumption."""

    @staticmethod
    def extract_trade_features(trade: Trade) -> Dict:
        """Extract features from a single trade."""
        features = {
            "hour_of_day": trade.entry_time.hour if trade.entry_time else 0,
            "day_of_week": trade.entry_time.weekday() if trade.entry_time else 0,
            "is_long": 1 if trade.direction.value == "long" else 0,
            "position_value": trade.position_value or 0,
            "initial_risk": trade.initial_risk or 0,
            "hold_minutes": trade.hold_duration_minutes or 0,
        }

        # Tags-based features
        if trade.tags:
            features["checklist_score"] = trade.tags.get("checklist_score", 50)
            features["discipline_score"] = trade.tags.get("discipline_score", 50)
            features["has_overrides"] = 1 if trade.tags.get("overrides") else 0
            features["override_count"] = len(trade.tags.get("overrides", []))
        else:
            features["checklist_score"] = 50
            features["discipline_score"] = 50
            features["has_overrides"] = 0
            features["override_count"] = 0

        # Journal features
        if trade.journal_entry:
            j = trade.journal_entry
            features["was_planned"] = 1 if j.was_planned else 0
            features["confidence"] = j.pre_trade_confidence or 5
            features["sleep_quality"] = j.sleep_quality or 5
            features["energy_level"] = j.energy_level or 5
            features["stress_level"] = j.stress_level or 5

            # Emotion encoding
            risky_emotions = {"revenge", "fomo", "frustrated", "euphoric", "anxious"}
            if j.pre_trade_emotion:
                features["is_risky_emotion"] = 1 if j.pre_trade_emotion.value in risky_emotions else 0
            else:
                features["is_risky_emotion"] = 0
        else:
            features["was_planned"] = 0
            features["confidence"] = 5
            features["sleep_quality"] = 5
            features["energy_level"] = 5
            features["stress_level"] = 5
            features["is_risky_emotion"] = 0

        return features

    @staticmethod
    def extract_batch(trades: List[Trade]) -> pd.DataFrame:
        """Extract features from a batch of trades."""
        extractor = TradeFeatureExtractor()
        rows = [extractor.extract_trade_features(t) for t in trades]
        return pd.DataFrame(rows)