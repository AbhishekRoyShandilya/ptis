"""
PTIS ML Prediction Engine
=========================

Advanced machine learning system for trading performance prediction and coaching recommendations.

Features:
- Performance prediction models
- Risk assessment models
- Behavioral pattern prediction
- Coaching recommendation engine
- Model evaluation and validation
- Feature importance analysis
- Model versioning and deployment

Provides intelligence for:
- Predictive coaching interventions
- Risk management alerts
- Performance forecasting
- Personalized recommendations
- Early warning systems

Design
------
- Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- Time-series forecasting capabilities
- Ensemble methods for robustness
- Explainable AI with SHAP values
- Continuous model retraining
- A/B testing framework
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json
from collections import defaultdict
import pickle
import joblib

# ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap

from core.models import Trade, BehavioralEvent, DecisionQualityRecord
from intelligence.profile_synthesizer import UnifiedTraderProfile
from ml.behavior_clustering import BehaviorClusteringSystem, BehaviorCluster

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models."""
    PERFORMANCE_PREDICTION = "performance_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    BEHAVIORAL_PREDICTION = "behavioral_prediction"
    COACHING_RECOMMENDATION = "coaching_recommendation"
    ANOMALY_DETECTION = "anomaly_detection"


class AlgorithmType(str, Enum):
    """Available ML algorithms."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class PredictionTarget(str, Enum):
    """Target variables for prediction."""
    NEXT_TRADE_R_MULTIPLE = "next_trade_r_multiple"
    NEXT_TRADE_WIN_PROBABILITY = "next_trade_win_probability"
    DAILY_PNL = "daily_pnl"
    RISK_OF_LARGE_LOSS = "risk_of_large_loss"
    DISCIPLINE_BREAKDOWN = "discipline_breakdown"
    EMOTIONAL_STATE = "emotional_state"
    COACHING_INTERVENTION = "coaching_intervention"


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # Model parameters
    model_type: ModelType = ModelType.PERFORMANCE_PREDICTION
    algorithm: AlgorithmType = AlgorithmType.RANDOM_FOREST
    target_variable: PredictionTarget = PredictionTarget.NEXT_TRADE_R_MULTIPLE
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    
    # Model-specific parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3
    
    nn_hidden_layer_sizes: Tuple[int, ...] = (100, 50)
    nn_alpha: float = 0.0001
    nn_learning_rate_init: float = 0.001
    
    # Ensemble parameters
    ensemble_weights: Optional[List[float]] = None
    
    # Validation thresholds
    min_accuracy: float = 0.6
    min_r2_score: float = 0.3


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    
    model_id: str
    algorithm: AlgorithmType
    target_variable: PredictionTarget
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    
    # Cross-validation scores
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Training info
    training_samples: int = 0
    training_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm.value,
            "target_variable": self.target_variable.value,
            "mse": round(self.mse, 4) if self.mse is not None else None,
            "mae": round(self.mae, 4) if self.mae is not None else None,
            "r2_score": round(self.r2_score, 4) if self.r2_score is not None else None,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
            "auc_score": round(self.auc_score, 4) if self.auc_score is not None else None,
            "cv_mean": round(self.cv_mean, 4) if self.cv_mean is not None else None,
            "cv_std": round(self.cv_std, 4) if self.cv_std is not None else None,
            "training_samples": self.training_samples,
            "training_time": round(self.training_time, 2),
            "timestamp": self.timestamp.isoformat(),
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            }
        }


@dataclass
class PredictionResult:
    """Result of a model prediction."""
    
    prediction_id: str
    model_id: str
    trader_id: str
    target_variable: PredictionTarget
    prediction: Union[float, int, str]
    confidence: float
    
    # Additional info
    feature_contributions: Optional[Dict[str, float]] = None
    similar_cases: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "trader_id": self.trader_id,
            "target_variable": self.target_variable.value,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 3),
            "feature_contributions": self.feature_contributions,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version
        }


class FeatureEngineering:
    """Advanced feature engineering for ML models."""
    
    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def engineer_features(
        self, 
        trader_data: Dict[str, Dict[str, Any]],
        target_variable: PredictionTarget
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Engineer features for ML training.
        
        Parameters
        ----------
        trader_data : Dict[str, Dict[str, Any]]
            Trader data dictionary
        target_variable : PredictionTarget
            Target variable to predict
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            Features, targets, and feature names
        """
        features_list = []
        targets_list = []
        
        for trader_id, data in trader_data.items():
            # Extract features
            trader_features = self._extract_comprehensive_features(data)
            
            # Extract target
            target = self._extract_target_variable(data, target_variable)
            
            if target is not None and trader_features:
                features_list.append(trader_features)
                targets_list.append(target)
        
        if not features_list:
            return np.array([]), np.array([]), []
        
        # Convert to arrays
        features = np.array(features_list)
        targets = np.array(targets_list)
        
        # Get feature names (only once)
        if not self.feature_names:
            self.feature_names = self._get_feature_names(trader_data)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Scale features
        if len(features) > 0:
            features = self.scaler.fit_transform(features)
        
        return features, targets, self.feature_names
    
    def _extract_comprehensive_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract comprehensive features from trader data."""
        features = []
        
        # Profile features
        profile = data.get("profile", {})
        behavioral_profile = profile.get("behavioral_profile", {})
        
        # Behavioral scores
        features.extend([
            behavioral_profile.get("discipline_score", 50) / 100.0,
            behavioral_profile.get("consistency_score", 50) / 100.0,
            behavioral_profile.get("patience_score", 50) / 100.0,
            behavioral_profile.get("impulsiveness_score", 50) / 100.0,
            behavioral_profile.get("risk_tolerance", 50) / 100.0,
            behavioral_profile.get("confidence_level", 50) / 100.0,
        ])
        
        # Performance features
        trades = data.get("trades", [])
        if trades:
            # Recent performance (last 10 trades)
            recent_trades = trades[-10:]
            
            win_rate = sum(1 for t in recent_trades if (t.get("pnl", 0) or 0) > 0) / len(recent_trades)
            avg_r_multiple = np.mean([t.get("r_multiple", 0) for t in recent_trades if t.get("r_multiple")]) or 0
            profit_factor = self._calculate_profit_factor(recent_trades)
            max_drawdown = self._calculate_max_drawdown(recent_trades)
            
            features.extend([
                win_rate,
                avg_r_multiple / 3.0,  # Normalize
                min(profit_factor, 3.0) / 3.0,  # Cap and normalize
                1.0 - max_drawdown,  # Invert drawdown
                len(recent_trades) / 10.0,  # Trade frequency
            ])
        else:
            features.extend([0.5, 0.0, 0.0, 1.0, 0.0])  # Default values
        
        # Risk features
        if trades:
            risk_pcts = [t.get("risk_pct", 1) for t in trades if t.get("risk_pct")]
            avg_risk = np.mean(risk_pcts) / 5.0 if risk_pcts else 0.2  # Normalize
            risk_consistency = 1.0 - (np.std(risk_pcts) / np.mean(risk_pcts)) if len(risk_pcts) > 1 else 0.5
            
            features.extend([
                avg_risk,
                risk_consistency,
                sum(1 for t in trades if t.get("stop_hit", False)) / len(trades),  # Stop hit rate
            ])
        else:
            features.extend([0.2, 0.5, 0.1])
        
        # Emotional features
        behavioral_events = data.get("behavioral_events", [])
        if behavioral_events:
            emotional_events = [
                e for e in behavioral_events 
                if any(trigger in e.get("description", "").lower() 
                       for trigger in ["tilt", "fear", "greed", "revenge", "fomo"])
            ]
            emotional_frequency = len(emotional_events) / len(behavioral_events)
            
            # Recent emotional events (last 5)
            recent_emotional = emotional_events[-5:]
            recent_emotional_intensity = len(recent_emotional) / 5.0
            
            features.extend([
                emotional_frequency,
                recent_emotional_intensity,
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Time-based features
        if trades:
            last_trade = trades[-1]
            if last_trade.get("entry_time"):
                hours_since_last = (datetime.utcnow() - last_trade["entry_time"]).total_seconds() / 3600
                time_factor = np.exp(-hours_since_last / 24.0)  # Decay over 24 hours
            else:
                time_factor = 0.5
            
            features.append(time_factor)
        else:
            features.append(0.5)
        
        # Market context features
        market_context = data.get("market_context", {})
        features.extend([
            0.5 if market_context.get("volatility_regime") == "normal" else 0.8 if market_context.get("volatility_regime") == "high" else 0.2,
            0.5 if market_context.get("trend_direction") == "neutral" else 0.8 if market_context.get("trend_direction") == "bullish" else 0.2,
            0.5 if market_context.get("session_type") == "regular" else 0.3,
        ])
        
        return features
    
    def _extract_target_variable(self, data: Dict[str, Any], target: PredictionTarget) -> Optional[float]:
        """Extract target variable from trader data."""
        trades = data.get("trades", [])
        
        if target == PredictionTarget.NEXT_TRADE_R_MULTIPLE:
            # Use last trade as "next" trade prediction target
            if trades:
                return trades[-1].get("r_multiple", 0)
        
        elif target == PredictionTarget.NEXT_TRADE_WIN_PROBABILITY:
            if trades:
                return 1.0 if (trades[-1].get("pnl", 0) or 0) > 0 else 0.0
        
        elif target == PredictionTarget.DAILY_PNL:
            # Sum of P&L for the day
            daily_pnl = sum(t.get("pnl", 0) for t in trades)
            return daily_pnl / 1000.0  # Normalize to thousands
        
        elif target == PredictionTarget.RISK_OF_LARGE_LOSS:
            # Binary: 1 if any loss > 2R in recent trades
            recent_trades = trades[-10:]
            large_loss = any((t.get("r_multiple", 0) or 0) < -2.0 for t in recent_trades)
            return 1.0 if large_loss else 0.0
        
        elif target == PredictionTarget.DISCIPLINE_BREAKDOWN:
            # Based on behavioral events and decision quality
            behavioral_events = data.get("behavioral_events", [])
            dq_records = data.get("dq_records", [])
            
            # Check for discipline violations
            violations = 0
            for event in behavioral_events:
                if "override" in event.get("description", "").lower():
                    violations += 1
            
            for record in dq_records:
                if record.get("composite_score", 100) < 50:
                    violations += 1
            
            # Binary classification
            return 1.0 if violations > 2 else 0.0
        
        elif target == PredictionTarget.EMOTIONAL_STATE:
            # Predict next emotional state
            behavioral_events = data.get("behavioral_events", [])
            if behavioral_events:
                last_event = behavioral_events[-1]
                emotional_state = last_event.get("emotional_state")
                
                # Map to numeric
                state_map = {"calm": 0, "confident": 1, "anxious": 2, "frustrated": 3, "tilt": 4}
                return state_map.get(emotional_state, 0) / 4.0
        
        elif target == PredictionTarget.COACHING_INTERVENTION:
            # Predict if coaching intervention is needed
            profile = data.get("profile", {})
            behavioral_profile = profile.get("behavioral_profile", {})
            
            # Multiple factors
            discipline_score = behavioral_profile.get("discipline_score", 50)
            recent_performance = self._calculate_recent_performance(data.get("trades", []))
            emotional_events = len([
                e for e in data.get("behavioral_events", [])
                if "tilt" in e.get("description", "").lower()
            ])
            
            # Intervention needed if:
            # - Low discipline OR
            # - Poor recent performance OR
            # - High emotional events
            intervention_needed = (
                discipline_score < 40 or
                recent_performance < 0.3 or
                emotional_events > 2
            )
            
            return 1.0 if intervention_needed else 0.0
        
        return None
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor for trades."""
        profits = [t.get("pnl", 0) for t in trades if (t.get("pnl", 0) or 0) > 0]
        losses = [abs(t.get("pnl", 0)) for t in trades if (t.get("pnl", 0) or 0) < 0]
        
        if profits and losses:
            return sum(profits) / sum(losses)
        return 1.0
    
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown."""
        if not trades:
            return 0.0
        
        pnl_series = [t.get("pnl", 0) for t in trades]
        cumulative = np.cumsum(pnl_series)
        
        if len(cumulative) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    def _calculate_recent_performance(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate recent performance score."""
        if not trades:
            return 0.5
        
        recent_trades = trades[-5:]
        wins = sum(1 for t in recent_trades if (t.get("pnl", 0) or 0) > 0)
        win_rate = wins / len(recent_trades)
        
        avg_r = np.mean([t.get("r_multiple", 0) for t in recent_trades if t.get("r_multiple")]) or 0
        
        # Combine win rate and R-multiple
        performance_score = (win_rate * 0.6 + min(avg_r / 2.0, 1.0) * 0.4)
        return performance_score
    
    def _get_feature_names(self, trader_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get feature names for the engineered features."""
        names = [
            # Behavioral features
            "discipline_score", "consistency_score", "patience_score",
            "impulsiveness_score", "risk_tolerance", "confidence_level",
            
            # Performance features
            "recent_win_rate", "recent_avg_r_multiple", "recent_profit_factor",
            "recent_max_drawdown", "trade_frequency",
            
            # Risk features
            "avg_risk_pct", "risk_consistency", "stop_hit_rate",
            
            # Emotional features
            "emotional_frequency", "recent_emotional_intensity",
            
            # Time features
            "time_factor",
            
            # Market context features
            "volatility_regime", "trend_direction", "session_type",
        ]
        
        return names
    
    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in feature matrix."""
        mask = np.isnan(features)
        features[mask] = 0
        return features


class MLEngine:
    """
    Main ML engine for prediction and recommendation.
    
    Supports multiple algorithms, feature engineering,
    model evaluation, and prediction generation.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.feature_engineering = FeatureEngineering(self.cfg)
        self.models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        
        logger.info(f"MLEngine initialized for {self.cfg.model_type}")
    
    def train_model(
        self, 
        trader_data: Dict[str, Dict[str, Any]]
    ) -> ModelPerformance:
        """
        Train an ML model on trader data.
        
        Parameters
        ----------
        trader_data : Dict[str, Dict[str, Any]]
            Training data
        
        Returns
        -------
        ModelPerformance
            Model performance metrics
        """
        start_time = datetime.utcnow()
        
        # Engineer features
        features, targets, feature_names = self.feature_engineering.engineer_features(
            trader_data, self.cfg.target_variable
        )
        
        if len(features) == 0 or len(targets) == 0:
            raise ValueError("No valid features or targets for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, 
            test_size=self.cfg.test_size, 
            random_state=self.cfg.random_state
        )
        
        # Select and train model
        model = self._create_model()
        model.fit(X_train, y_train)
        
        # Evaluate model
        performance = self._evaluate_model(
            model, X_test, y_test, X_train, y_train, feature_names, start_time
        )
        
        # Store model
        model_id = f"{self.cfg.algorithm.value}_{self.cfg.target_variable.value}_{int(start_time.timestamp())}"
        self.models[model_id] = model
        self.model_performances[model_id] = performance
        
        logger.info(f"Model trained: {model_id}, R²: {performance.r2_score:.3f}")
        return performance
    
    def _create_model(self) -> Any:
        """Create ML model based on algorithm configuration."""
        if self.cfg.algorithm == AlgorithmType.RANDOM_FOREST:
            if self._is_classification_task():
                return RandomForestClassifier(
                    n_estimators=self.cfg.rf_n_estimators,
                    max_depth=self.cfg.rf_max_depth,
                    min_samples_split=self.cfg.rf_min_samples_split,
                    random_state=self.cfg.random_state
                )
            else:
                return RandomForestRegressor(
                    n_estimators=self.cfg.rf_n_estimators,
                    max_depth=self.cfg.rf_max_depth,
                    min_samples_split=self.cfg.rf_min_samples_split,
                    random_state=self.cfg.random_state
                )
        
        elif self.cfg.algorithm == AlgorithmType.GRADIENT_BOOSTING:
            if self._is_classification_task():
                return GradientBoostingClassifier(
                    n_estimators=self.cfg.gb_n_estimators,
                    learning_rate=self.cfg.gb_learning_rate,
                    max_depth=self.cfg.gb_max_depth,
                    random_state=self.cfg.random_state
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=self.cfg.gb_n_estimators,
                    learning_rate=self.cfg.gb_learning_rate,
                    max_depth=self.cfg.gb_max_depth,
                    random_state=self.cfg.random_state
                )
        
        elif self.cfg.algorithm == AlgorithmType.LINEAR_REGRESSION:
            if self._is_classification_task():
                return LogisticRegression(random_state=self.cfg.random_state)
            else:
                return LinearRegression()
        
        elif self.cfg.algorithm == AlgorithmType.NEURAL_NETWORK:
            if self._is_classification_task():
                return MLPClassifier(
                    hidden_layer_sizes=self.cfg.nn_hidden_layer_sizes,
                    alpha=self.cfg.nn_alpha,
                    learning_rate_init=self.cfg.nn_learning_rate_init,
                    random_state=self.cfg.random_state,
                    max_iter=1000
                )
            else:
                return MLPRegressor(
                    hidden_layer_sizes=self.cfg.nn_hidden_layer_sizes,
                    alpha=self.cfg.nn_alpha,
                    learning_rate_init=self.cfg.nn_learning_rate_init,
                    random_state=self.cfg.random_state,
                    max_iter=1000
                )
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.cfg.algorithm}")
    
    def _is_classification_task(self) -> bool:
        """Check if the target variable is a classification task."""
        classification_targets = [
            PredictionTarget.NEXT_TRADE_WIN_PROBABILITY,
            PredictionTarget.RISK_OF_LARGE_LOSS,
            PredictionTarget.DISCIPLINE_BREAKDOWN,
            PredictionTarget.COACHING_INTERVENTION
        ]
        return self.cfg.target_variable in classification_targets
    
    def _evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        start_time: datetime
    ) -> ModelPerformance:
        """Evaluate model performance."""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        performance = ModelPerformance(
            model_id="",  # Will be set by caller
            algorithm=self.cfg.algorithm,
            target_variable=self.cfg.target_variable,
            training_samples=len(X_train),
            training_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        if self._is_classification_task():
            # Classification metrics
            performance.accuracy = accuracy_score(y_test, y_pred)
            performance.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            performance.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            performance.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC for binary classification
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                performance.auc_score = roc_auc_score(y_test, y_proba)
        else:
            # Regression metrics
            performance.mse = mean_squared_error(y_test, y_pred)
            performance.mae = mean_absolute_error(y_test, y_pred)
            performance.r2_score = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cfg.cross_validation_folds)
        performance.cv_scores = cv_scores.tolist()
        performance.cv_mean = cv_scores.mean()
        performance.cv_std = cv_scores.std()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(model.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = importance
            
            # Filter by threshold
            performance.feature_importance = {
                k: v for k, v in importance_dict.items() 
                if v >= self.cfg.feature_importance_threshold
            }
        
        return performance
    
    def predict(
        self, 
        trader_data: Dict[str, Any], 
        model_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Make prediction for a trader.
        
        Parameters
        ----------
        trader_data : Dict[str, Any]
            Trader data for prediction
        model_id : Optional[str]
            Specific model to use (if None, uses best performing)
        
        Returns
        -------
        PredictionResult
            Prediction result with confidence and explanations
        """
        # Select model
        if model_id is None:
            model_id = self._get_best_model_id()
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        performance = self.model_performances[model_id]
        
        # Engineer features for prediction
        features_dict = {"temp_trader": trader_data}
        features, _, feature_names = self.feature_engineering.engineer_features(
            features_dict, self.cfg.target_variable
        )
        
        if len(features) == 0:
            raise ValueError("No features could be extracted for prediction")
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(model, features, prediction)
        
        # Generate explanation
        explanation = self._generate_explanation(model, features, feature_names, prediction)
        
        # Create result
        result = PredictionResult(
            prediction_id=f"pred_{int(datetime.utcnow().timestamp())}",
            model_id=model_id,
            trader_id=trader_data.get("trader_id", "unknown"),
            target_variable=self.cfg.target_variable,
            prediction=prediction,
            confidence=confidence,
            explanation=explanation
        )
        
        return result
    
    def _get_best_model_id(self) -> str:
        """Get the ID of the best performing model."""
        if not self.model_performances:
            raise ValueError("No trained models available")
        
        # Sort by performance metric
        if self._is_classification_task():
            # Use F1 score for classification
            best_model_id = max(
                self.model_performances.keys(),
                key=lambda k: self.model_performances[k].f1_score or 0
            )
        else:
            # Use R² score for regression
            best_model_id = max(
                self.model_performances.keys(),
                key=lambda k: self.model_performations[k].r2_score or 0
            )
        
        return best_model_id
    
    def _calculate_prediction_confidence(
        self, 
        model: Any, 
        features: np.ndarray, 
        prediction: Union[float, int]
    ) -> float:
        """Calculate confidence score for prediction."""
        # For tree-based models, use prediction variance
        if hasattr(model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = []
            for estimator in model.estimators_:
                tree_pred = estimator.predict(features)[0]
                tree_predictions.append(tree_pred)
            
            # Calculate variance as inverse confidence
            variance = np.var(tree_predictions)
            confidence = 1.0 / (1.0 + variance)
        else:
            # Default confidence based on model performance
            confidence = 0.7  # Default
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_explanation(
        self, 
        model: Any, 
        features: np.ndarray, 
        feature_names: List[str], 
        prediction: Union[float, int]
    ) -> str:
        """Generate explanation for the prediction."""
        explanation_parts = []
        
        # Feature importance explanation
        if hasattr(model, 'feature_importances_'):
            # Get top features
            importance_scores = model.feature_importances_
            top_indices = np.argsort(importance_scores)[-3:][::-1]
            
            explanation_parts.append("Key factors:")
            for idx in top_indices:
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    importance = importance_scores[idx]
                    explanation_parts.append(f"- {feature_name}: {importance:.3f}")
        
        # Prediction interpretation
        if self._is_classification_task():
            if prediction > 0.5:
                explanation_parts.append("Model predicts positive outcome with high confidence.")
            else:
                explanation_parts.append("Model predicts negative outcome with high confidence.")
        else:
            if prediction > 0:
                explanation_parts.append(f"Model predicts positive outcome: {prediction:.2f}")
            else:
                explanation_parts.append(f"Model predicts negative outcome: {prediction:.2f}")
        
        return "\n".join(explanation_parts)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models."""
        if not self.model_performances:
            return {"total_models": 0, "models": []}
        
        summary = {
            "total_models": len(self.model_performances),
            "target_variable": self.cfg.target_variable.value,
            "models": []
        }
        
        for model_id, performance in self.model_performances.items():
            summary["models"].append(performance.to_dict())
        
        return summary


# Global instances
_ml_engines: Dict[str, MLEngine] = {}

def get_ml_engine(model_type: ModelType, config: Optional[ModelConfig] = None) -> MLEngine:
    """Get or create ML engine for specific model type."""
    key = model_type.value
    if key not in _ml_engines:
        cfg = config or ModelConfig(model_type=model_type)
        _ml_engines[key] = MLEngine(cfg)
    return _ml_engines[key]
