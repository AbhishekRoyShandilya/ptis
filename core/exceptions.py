# exceptions.py
# core/exceptions.py
"""
Custom exception classes for PTIS.
"""


class PTISError(Exception):
    """Base exception for PTIS."""
    pass


class TradeNotFoundError(PTISError):
    """Trade not found in database."""
    def __init__(self, trade_id: str):
        self.trade_id = trade_id
        super().__init__(f"Trade not found: {trade_id}")


class StrategyNotFoundError(PTISError):
    """Strategy not found."""
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        super().__init__(f"Strategy not found: {strategy_name}")


class InsufficientDataError(PTISError):
    """Not enough data for analysis."""
    pass


class RiskLimitExceededError(PTISError):
    """Risk limit exceeded."""
    def __init__(self, limit_type: str, current: float, maximum: float):
        self.limit_type = limit_type
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"Risk limit exceeded: {limit_type} "
            f"(current: {current}, max: {maximum})"
        )


class ShadowTradeError(PTISError):
    """Error in shadow trade operations."""
    pass


class ValidationError(PTISError):
    """Input validation error."""
    pass