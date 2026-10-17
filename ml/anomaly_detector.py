# ml/anomaly_detector.py
"""
Trading anomaly detection using Isolation Forest.
Detects unusual trades that deviate from normal patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.enums import TradeStatus
from core.models import Trade
from ml.feature_engineering import TradeFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single trade."""
    trade_id: str
    is_anomaly: bool
    anomaly_score: float       # -1 (most anomalous) to 1 (most normal)
    anomaly_reasons: List[str] = field(default_factory=list)
    feature_deviations: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Summary of anomaly detection across multiple trades."""
    total_trades: int
    anomalies_found: int
    anomaly_rate: float
    results: List[AnomalyResult] = field(default_factory=list)
    top_anomalous_features: List[str] = field(default_factory=list)


class TradingAnomalyDetector:
    """
    Detects anomalous trades using Isolation Forest or statistical
    fallback when sklearn is not available.

    Anomalies include:
    - Unusual position sizes relative to history
    - Trades at atypical times
    - Emotional state outliers
    - Abnormal hold durations
    - Unusual override patterns
    """

    def __init__(
        self,
        db: Session,
        contamination: float = 0.1,
        min_training_samples: int = 30,
    ):
        self.db = db
        self._contamination = contamination
        self._min_samples = min_training_samples
        self._model = None
        self._feature_means: Dict[str, float] = {}
        self._feature_stds: Dict[str, float] = {}
        self._is_trained = False

    def train(self, lookback_days: int = 90) -> bool:
        """
        Train the anomaly detection model on recent trades.
        Returns True if training succeeded.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        trades = self.db.query(Trade).filter(
            and_(
                Trade.status == TradeStatus.CLOSED,
                Trade.entry_time >= cutoff,
            )
        ).all()

        if len(trades) < self._min_samples:
            logger.warning(
                "Insufficient trades for anomaly training: %d < %d",
                len(trades), self._min_samples,
            )
            return False

        df = TradeFeatureExtractor.extract_batch(trades)

        # Store feature statistics for fallback and explanation
        for col in df.columns:
            self._feature_means[col] = float(df[col].mean())
            self._feature_stds[col] = float(df[col].std()) or 1.0

        # Try Isolation Forest
        try:
            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                contamination=self._contamination,
                n_estimators=100,
                random_state=42,
            )
            self._model.fit(df.values)
            self._is_trained = True
            logger.info(
                "Anomaly detector trained on %d trades (%d features)",
                len(trades), len(df.columns),
            )
            return True

        except ImportError:
            logger.info(
                "sklearn not available — using statistical fallback. "
                "Trained on %d trades.", len(trades),
            )
            self._is_trained = True
            return True

    def detect(self, trade: Trade) -> AnomalyResult:
        """Detect if a single trade is anomalous."""
        if not self._is_trained:
            self.train()

        features = TradeFeatureExtractor.extract_trade_features(trade)
        feature_array = np.array([list(features.values())])

        # Get deviations from mean (z-scores)
        deviations = {}
        for feat_name, val in features.items():
            mean = self._feature_means.get(feat_name, 0)
            std = self._feature_stds.get(feat_name, 1)
            z = abs(val - mean) / std if std > 0 else 0
            if z > 2.0:
                deviations[feat_name] = round(z, 2)

        # Determine anomaly
        if self._model is not None:
            score = float(self._model.score_samples(feature_array)[0])
            prediction = int(self._model.predict(feature_array)[0])
            is_anomaly = prediction == -1
        else:
            # Statistical fallback: anomaly if 3+ features are > 2 std devs
            score = -1.0 * len(deviations) / max(len(features), 1)
            is_anomaly = len(deviations) >= 3

        reasons = self._explain_anomaly(features, deviations)

        return AnomalyResult(
            trade_id=str(trade.id),
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            anomaly_reasons=reasons,
            feature_deviations=deviations,
        )

    def detect_batch(
        self,
        lookback_days: int = 30,
    ) -> AnomalyReport:
        """Run anomaly detection on recent trades."""
        if not self._is_trained:
            self.train()

        cutoff = datetime.now() - timedelta(days=lookback_days)
        trades = self.db.query(Trade).filter(
            and_(
                Trade.status == TradeStatus.CLOSED,
                Trade.entry_time >= cutoff,
            )
        ).all()

        results = []
        anomalies = 0
        feature_anomaly_counts: Dict[str, int] = {}

        for trade in trades:
            result = self.detect(trade)
            results.append(result)
            if result.is_anomaly:
                anomalies += 1
                for feat in result.feature_deviations:
                    feature_anomaly_counts[feat] = feature_anomaly_counts.get(feat, 0) + 1

        # Top anomalous features
        top_features = sorted(
            feature_anomaly_counts.keys(),
            key=lambda k: feature_anomaly_counts[k],
            reverse=True,
        )[:5]

        return AnomalyReport(
            total_trades=len(trades),
            anomalies_found=anomalies,
            anomaly_rate=anomalies / len(trades) if trades else 0,
            results=results,
            top_anomalous_features=top_features,
        )

    def _explain_anomaly(
        self,
        features: Dict[str, float],
        deviations: Dict[str, float],
    ) -> List[str]:
        """Generate human-readable reasons for anomaly."""
        reasons = []

        if "hold_minutes" in deviations:
            hold = features["hold_minutes"]
            avg = self._feature_means.get("hold_minutes", 0)
            if hold > avg * 2:
                reasons.append(
                    f"Unusually long hold: {hold:.0f}min vs avg {avg:.0f}min"
                )
            elif hold < avg * 0.3:
                reasons.append(
                    f"Unusually short hold: {hold:.0f}min vs avg {avg:.0f}min"
                )

        if "hour_of_day" in deviations:
            reasons.append(
                f"Trade at unusual hour: {features['hour_of_day']:.0f}:00"
            )

        if "position_value" in deviations:
            val = features["position_value"]
            avg = self._feature_means.get("position_value", 0)
            if val > avg:
                reasons.append(
                    f"Position size {val/avg:.1f}x larger than average"
                )

        if "is_risky_emotion" in deviations and features.get("is_risky_emotion"):
            reasons.append("Trade taken under risky emotional state")

        if "override_count" in deviations:
            count = features["override_count"]
            reasons.append(f"Unusual override count: {count:.0f}")

        if "checklist_score" in deviations:
            score = features["checklist_score"]
            avg = self._feature_means.get("checklist_score", 50)
            if score < avg:
                reasons.append(
                    f"Low checklist score: {score:.0f} vs avg {avg:.0f}"
                )

        if not reasons and deviations:
            top_devs = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:3]
            for feat, z in top_devs:
                reasons.append(f"{feat} deviates by {z:.1f} std deviations")

        return reasons
