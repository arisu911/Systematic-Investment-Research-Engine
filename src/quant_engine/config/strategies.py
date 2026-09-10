"""Strategy configuration models and parameter containers."""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """Enumeration of standard strategy families."""
    
    MA_MOMENTUM = "MovingAverageMomentum"
    TS_MOMENTUM = "TimeSeriesMomentum"
    MEAN_REVERSION = "MeanReversion"
    BREAKOUT = "Breakout"
    FACTOR = "FactorComposite"
    MACRO_MOMENTUM = "MacroAwareMomentum"


class StrategyConfig(BaseModel):
    """Configuration container for strategy parameters and execution constraints."""
    
    name: str = "Quantitative Strategy"
    strategy_type: StrategyType = StrategyType.TS_MOMENTUM
    parameters: Dict[str, Any] = Field(default_factory=dict)
    long_only: bool = True
    allow_cash: bool = True
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    description: Optional[str] = None
    
    @classmethod
    def default_ma_momentum(cls) -> "StrategyConfig":
        return cls(
            name="MA Crossover Momentum",
            strategy_type=StrategyType.MA_MOMENTUM,
            parameters={"fast_period": 20, "slow_period": 100, "signal_type": "crossover"},
            long_only=True,
            description="Dual Moving Average Crossover (Fast vs Slow SMA)",
        )
        
    @classmethod
    def default_ts_momentum(cls) -> "StrategyConfig":
        return cls(
            name="Time-Series Momentum",
            strategy_type=StrategyType.TS_MOMENTUM,
            parameters={"lookback_days": 120, "holding_days": 20, "vol_scale": True},
            long_only=True,
            description="Absolute time-series momentum with volatility scaling",
        )
        
    @classmethod
    def default_mean_reversion(cls) -> "StrategyConfig":
        return cls(
            name="Z-Score Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            parameters={"window": 20, "entry_z": 2.0, "exit_z": 0.5},
            long_only=True,
            description="Statistical z-score reversion to rolling mean",
        )
        
    @classmethod
    def default_breakout(cls) -> "StrategyConfig":
        return cls(
            name="Donchian Channel Breakout",
            strategy_type=StrategyType.BREAKOUT,
            parameters={"lookback": 50, "atr_filter": True, "atr_period": 14},
            long_only=True,
            description="Classic Donchian Channel breakout with ATR volatility filter",
        )
