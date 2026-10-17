# constants.py
# config/constants.py
"""
Application-wide constants.
"""

# ── Emotional States (ordered by risk) ──
RISKY_EMOTIONS = {"revenge", "fomo", "frustrated", "euphoric", "anxious"}
NEUTRAL_EMOTIONS = {"calm", "neutral", "confident"}
POSITIVE_EMOTIONS = {"calm", "confident"}

# ── Discipline Thresholds ──
DISCIPLINE_EXCELLENT = 85
DISCIPLINE_GOOD = 70
DISCIPLINE_POOR = 50
DISCIPLINE_CRITICAL = 30

# ── Regime Fitness Thresholds ──
REGIME_FITNESS_EXCELLENT = 75
REGIME_FITNESS_GOOD = 55
REGIME_FITNESS_MARGINAL = 35

# ── Session Classification (Eastern Time) ──
SESSION_TIMES = {
    "premarket": (4.0, 9.5),
    "open": (9.5, 10.5),
    "midday": (10.5, 14.0),
    "power_hour": (14.0, 16.0),
    "after_hours": (16.0, 20.0),
}

# ── Default Limits ──
MAX_DAILY_TRADES_DEFAULT = 8
MAX_CONSECUTIVE_LOSSES_DEFAULT = 3
MAX_DAILY_LOSS_PERCENT_DEFAULT = 3.0
MAX_STRATEGIES_PER_DAY = 3

# ── Shadow Trade Variants ──
SHADOW_VARIANT_CONFIGS = {
    "tight_stop_3R": {
        "stop_multiplier": 0.75,
        "r_target": 3.0,
        "trailing": False,
        "description": "Tighter stop, higher R target"
    },
    "wide_stop_trailing": {
        "stop_multiplier": 1.5,
        "r_target": 2.0,
        "trailing": True,
        "trailing_activation_r": 1.0,
        "description": "Wider stop with trailing"
    },
    "breakeven_trail": {
        "stop_multiplier": 1.0,
        "r_target": 3.0,
        "trailing": True,
        "trailing_activation_r": 1.0,
        "move_to_breakeven_at_r": 1.0,
        "description": "Move to breakeven at 1R, trail for 3R"
    },
}