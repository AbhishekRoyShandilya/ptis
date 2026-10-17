"""
Performance Predictor
======================
Predicts trade outcome probability (win/loss) and expected R-multiple
before trade entry, based on current conditions + behavioral state.

Used to surface "confidence score" warnings: "Based on your history,
trades in this emotional state + market regime have a 34% win rate."

Two models:
1. WinProbabilityModel  — binary classification (win / loss)
2. ExpectedRModel       — regression (predicted R-multiple)

Both fall back to statistical base rates when ML is unavailable.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradePrediction:
    win_probability: float      # 0-1
    expected_r: float           # predicted R-multiple
    confidence: float           # model confidence 0-1
    method: str                 # "ml" | "statistical"
    warning: Optional[str]      # human-readable warning if conditions are unfavourable
    base_win_rate: float        # historical overall win rate for context


@dataclass
class TradeConditions:
    """Input features for the predictor."""
    emotion_pre: Optional[str] = None     # e.g. "calm", "fomo", "revenge"
    market_regime: Optional[str] = None   # e.g. "trending_up", "ranging"
    session: Optional[str] = None         # "morning" | "afternoon" | "late"
    discipline_score: Optional[float] = None
    conviction_score: Optional[float] = None
    setup_type: Optional[str] = None
    day_of_week: Optional[int] = None     # 0=Mon … 4=Fri
    is_after_loss_streak: bool = False
    override_count: int = 0               # overrides already taken before entry


_RISKY_EMOTIONS = {"revenge", "fomo", "frustrated", "euphoric", "anxious", "fearful"}
_CALM_EMOTIONS  = {"calm", "focused", "neutral", "confident"}


def _condition_vector(c: TradeConditions) -> list[float]:
    """Convert TradeConditions to a numeric feature vector."""
    em_risk = 1.0 if (c.emotion_pre or "").lower() in _RISKY_EMOTIONS else 0.0
    em_calm = 1.0 if (c.emotion_pre or "").lower() in _CALM_EMOTIONS else 0.0
    is_trending = 1.0 if "trending" in (c.market_regime or "") else 0.0
    is_ranging  = 1.0 if "ranging" in (c.market_regime or "") else 0.0
    is_volatile = 1.0 if "volatile" in (c.market_regime or "") else 0.0
    is_morning  = 1.0 if c.session == "morning" else 0.0
    disc = (c.discipline_score or 50.0) / 100.0
    conv = (c.conviction_score or 5.0) / 10.0
    dow  = (c.day_of_week or 2) / 4.0   # normalise Mon-Fri
    after_loss = 1.0 if c.is_after_loss_streak else 0.0
    ovr_norm = min(c.override_count / 3.0, 1.0)
    return [em_risk, em_calm, is_trending, is_ranging, is_volatile,
            is_morning, disc, conv, dow, after_loss, ovr_norm]


FEATURE_NAMES = [
    "emotion_risk", "emotion_calm", "is_trending", "is_ranging", "is_volatile",
    "is_morning", "discipline_norm", "conviction_norm", "day_of_week_norm",
    "after_loss_streak", "override_count_norm",
]


class StatisticalPredictor:
    """
    Base-rate predictor — uses historical trade slices to estimate win probability.
    No ML library required.
    """

    def __init__(self):
        self._history: list[tuple[TradeConditions, float, float]] = []  # (cond, win, r)

    def fit(self, trades: list) -> None:
        """Load historical trades to build base rates."""
        self._history = []
        for t in trades:
            pnl = getattr(t, "pnl", None)
            if pnl is None:
                continue
            r = getattr(t, "r_multiple", None) or (1.0 if pnl > 0 else -1.0)
            em = getattr(t, "emotion_pre", None)
            em_str = str(em.value if hasattr(em, "value") else em) if em else None
            reg = getattr(t, "market_regime", None)
            reg_str = str(reg.value if hasattr(reg, "value") else reg) if reg else None
            et = getattr(t, "entry_time", None)
            sess = None
            if et:
                h = et.hour
                sess = "morning" if h < 11 else "afternoon" if h < 14 else "late"
            cond = TradeConditions(
                emotion_pre=em_str,
                market_regime=reg_str,
                session=sess,
                discipline_score=getattr(t, "discipline_score", None),
                day_of_week=et.weekday() if et else None,
            )
            self._history.append((cond, 1.0 if pnl > 0 else 0.0, float(r)))

    def predict(self, conditions: TradeConditions) -> TradePrediction:
        if not self._history:
            return TradePrediction(0.5, 0.0, 0.0, "statistical", None, 0.5)

        wins = [w for _, w, _ in self._history]
        base_wr = statistics.mean(wins)
        base_r  = statistics.mean(r for _, _, r in self._history)

        # Narrow to matching conditions
        def _match(h_cond: TradeConditions) -> int:
            score = 0
            if conditions.emotion_pre and h_cond.emotion_pre == conditions.emotion_pre:
                score += 3
            if conditions.market_regime and h_cond.market_regime == conditions.market_regime:
                score += 2
            if conditions.session and h_cond.session == conditions.session:
                score += 1
            return score

        scored = sorted(self._history, key=lambda x: -_match(x[0]))
        subset = scored[:max(20, len(scored) // 4)]
        sub_wins = [w for _, w, _ in subset]
        sub_rs   = [r for _, _, r in subset]
        wr   = statistics.mean(sub_wins) if sub_wins else base_wr
        ex_r = statistics.mean(sub_rs)   if sub_rs   else base_r

        warning = self._build_warning(conditions, wr, base_wr)
        conf = min(len(subset) / 30, 1.0)   # confidence grows with sample size

        return TradePrediction(
            win_probability=round(wr, 3),
            expected_r=round(ex_r, 3),
            confidence=round(conf, 3),
            method="statistical",
            warning=warning,
            base_win_rate=round(base_wr, 3),
        )

    @staticmethod
    def _build_warning(c: TradeConditions, wr: float, base_wr: float) -> Optional[str]:
        warnings = []
        if c.emotion_pre and c.emotion_pre.lower() in _RISKY_EMOTIONS:
            warnings.append(f"Entering in '{c.emotion_pre}' state — historically weak performance")
        if c.is_after_loss_streak:
            warnings.append("Consecutive losses detected — revenge trading risk elevated")
        if c.override_count >= 2:
            warnings.append(f"{c.override_count} overrides already taken — discipline flagged")
        if wr < base_wr * 0.80:
            warnings.append(
                f"Win rate in these conditions ({wr:.0%}) is below your average ({base_wr:.0%})"
            )
        return " | ".join(warnings) if warnings else None


class MLPerformancePredictor:
    """
    Gradient-boosted predictor using scikit-learn.
    Falls back to StatisticalPredictor if scikit-learn is unavailable.
    """

    def __init__(self):
        self._win_model = None
        self._r_model   = None
        self._stat = StatisticalPredictor()
        self._trained = False

    def fit(self, trades: list) -> bool:
        self._stat.fit(trades)
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        except ImportError:
            logger.warning("scikit-learn not installed; using statistical predictor only")
            return False

        X_cls, y_cls, X_reg, y_reg = [], [], [], []
        for t in trades:
            pnl = getattr(t, "pnl", None)
            if pnl is None:
                continue
            r = getattr(t, "r_multiple", None) or (1.0 if pnl > 0 else -1.0)
            em = getattr(t, "emotion_pre", None)
            em_str = str(em.value if hasattr(em, "value") else em) if em else None
            reg = getattr(t, "market_regime", None)
            reg_str = str(reg.value if hasattr(reg, "value") else reg) if reg else None
            et = getattr(t, "entry_time", None)
            sess = None
            if et:
                h = et.hour
                sess = "morning" if h < 11 else "afternoon" if h < 14 else "late"
            cond = TradeConditions(
                emotion_pre=em_str, market_regime=reg_str, session=sess,
                discipline_score=getattr(t, "discipline_score", None),
                conviction_score=getattr(t, "conviction_score", None),
                day_of_week=et.weekday() if et else None,
                override_count=len(getattr(t, "overrides", [])),
            )
            fv = _condition_vector(cond)
            X_cls.append(fv); y_cls.append(1 if pnl > 0 else 0)
            X_reg.append(fv); y_reg.append(float(r))

        if len(X_cls) < 20:
            return False

        self._win_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self._win_model.fit(X_cls, y_cls)
        self._r_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self._r_model.fit(X_reg, y_reg)
        self._trained = True
        logger.info(f"Performance predictor trained on {len(X_cls)} trades")
        return True

    def predict(self, conditions: TradeConditions) -> TradePrediction:
        stat_pred = self._stat.predict(conditions)
        if not self._trained:
            return stat_pred

        fv = [_condition_vector(conditions)]
        win_prob = float(self._win_model.predict_proba(fv)[0][1])
        ex_r     = float(self._r_model.predict(fv)[0])
        warning  = StatisticalPredictor._build_warning(conditions, win_prob, stat_pred.base_win_rate)

        return TradePrediction(
            win_probability=round(win_prob, 3),
            expected_r=round(ex_r, 3),
            confidence=0.75,
            method="ml",
            warning=warning,
            base_win_rate=stat_pred.base_win_rate,
        )

    def save(self, path: str) -> None:
        if not self._trained:
            raise RuntimeError("Model not trained")
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"win": self._win_model, "r": self._r_model}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._win_model = data["win"]
        self._r_model   = data["r"]
        self._trained = True


def get_performance_predictor(use_ml: bool = True):
    return MLPerformancePredictor() if use_ml else StatisticalPredictor()
