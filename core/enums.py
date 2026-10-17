# core/enums.py
"""PTIS application-wide enumerations."""

from enum import Enum, auto


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class MarketRegime(str, Enum):
    # Trend-based regimes
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    TRENDING_BULLISH = "trending_bullish"   # used by strategy templates + RegimeType
    TRENDING_BEARISH = "trending_bearish"   # used by strategy templates + RegimeType
    # Range/consolidation regimes
    RANGE = "range"
    RANGING = "ranging"          # alias used by regime_detector (same concept)
    MEAN_REVERTING = "mean_reverting"
    # Volatility-based regimes
    VOLATILE = "volatile"
    HIGH_VOLATILITY = "high_volatility"
    VOLATILE_EXPANSION = "volatile_expansion"        # RegimeType alias
    LOW_VOL = "low_vol"
    LOW_VOLATILITY = "low_volatility"
    LOW_VOLATILITY_COMPRESSION = "low_volatility_compression"  # RegimeType alias
    # Transitional
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    UNKNOWN = "unknown"


class SetupType(str, Enum):
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    MOMENTUM = "momentum"
    RANGE = "range"
    CUSTOM = "custom"


class VolatilityRegime(str, Enum):
    LOW = "low"
    AVERAGE = "average"
    HIGH = "high"
    EXTREME = "extreme"


class EmotionalState(str, Enum):
    CALM = "calm"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    FRUSTRATED = "frustrated"
    REVENGE = "revenge"
    FOMO = "fomo"
    BORED = "bored"
    EUPHORIC = "euphoric"
    NEUTRAL = "neutral"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SessionType(str, Enum):
    PRE_MARKET = "pre_market"
    MORNING = "morning"           # 9:30 - 11:30
    MIDDAY = "midday"             # 11:30 - 14:00
    AFTERNOON = "afternoon"       # 14:00 - 15:30
    CLOSE = "close"               # 15:30 - 16:00
    AFTER_HOURS = "after_hours"


# ── Strategy Attribution ──

class AttributionTier(str, Enum):
    TIER1_DECLARED = "tier1_declared"
    TIER2_AUTO_MATCHED = "tier2_auto_matched"
    TIER3_OVERRIDE = "tier3_override"


class ConditionStatus(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    PARTIALLY_MET = "partially_met"
    INCONCLUSIVE = "inconclusive"   # insufficient data
    ERROR = "error"


class OverrideType(str, Enum):
    EARLY_ENTRY = "early_entry"
    LATE_ENTRY = "late_entry"
    WRONG_DIRECTION = "wrong_direction"
    MISSING_CONFIRMATION = "missing_confirmation"
    AVOID_CONDITION_VIOLATED = "avoid_condition_violated"
    OVERSIZING = "oversizing"
    UNDERSIZING = "undersizing"
    STOP_TOO_WIDE = "stop_too_wide"
    STOP_TOO_TIGHT = "stop_too_tight"
    NO_STOP_SET = "no_stop_set"
    STOP_MOVED = "stop_moved"
    EARLY_EXIT = "early_exit"
    LATE_EXIT = "late_exit"
    TARGET_SKIPPED = "target_skipped"
    TIME_WINDOW_VIOLATION = "time_window_violation"
    REGIME_MISMATCH = "regime_mismatch"
    INVALIDATION_IGNORED = "invalidation_ignored"
    RAPID_RE_ENTRY = "rapid_re_entry"


class OverrideSeverity(str, Enum):
    LOW = "low"             # minor deviation, likely acceptable
    MEDIUM = "medium"       # notable deviation, worth tracking
    HIGH = "high"           # significant rule break
    CRITICAL = "critical"   # major discipline failure


class BehavioralEventType(str, Enum):
    FOMO_ENTRY = "fomo_entry"
    REVENGE_TRADE = "revenge_trade"
    IMPULSE_ENTRY = "impulse_entry"
    EMOTIONAL_EXIT = "emotional_exit"
    OVERTRADE = "overtrade"
    RAPID_RE_ENTRY = "rapid_re_entry"
    LOSS_CHASING = "loss_chasing"
    PROFIT_GIVING_BACK = "profit_giving_back"
    STOP_WIDENING = "stop_widening"
    POSITION_AVERAGING = "position_averaging"
    DISCIPLINE_HOLD = "discipline_hold"      # positive event
    CLEAN_EXECUTION = "clean_execution"      # positive event
    PATIENCE_REWARD = "patience_reward"      # positive event


class ShadowTradeType(str, Enum):
    MIRROR = "mirror"           # exact copy of real trade
    STRATEGY = "strategy"       # what strategy rules dictate
    VARIANT_TIGHT = "variant_tight"
    VARIANT_WIDE = "variant_wide"
    VARIANT_TRAIL = "variant_trail"


# ── Strategy Knowledge Engine (SKE) ──

class SKOStatus(str, Enum):
    """Strategy Knowledge Object lifecycle tiers."""
    HYPOTHESIS = "hypothesis"      # 0-9 trades
    DEVELOPING = "developing"      # 10-19 trades
    VALIDATED = "validated"        # 20-49 trades
    PROVEN = "proven"              # 50+ trades
    ARCHIVED = "archived"


class ConvictionBand(str, Enum):
    """Fuzzy conviction score → sizing band mapping."""
    SKIP = "skip"
    SMALL = "small"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAX_CONVICTION = "max"


class ScoringFunctionType(str, Enum):
    """Types of fuzzy scoring functions for feature evaluation."""
    GAUSSIAN = "gaussian"
    TRAPEZOID = "trapezoid"
    SIGMOID = "sigmoid"
    ASYMMETRIC = "asymmetric"


class AutomationPhase(str, Enum):
    """Automation readiness progression phases."""
    OBSERVATION = "observation"
    AWARENESS = "awareness"
    ALIGNMENT = "alignment"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"