"""Market definitions, asset universes, and cost model specifications."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field
from quant_engine.config.settings import get_settings


class TransactionCostConfig(BaseModel):
    """Cost configuration in basis points (1 bp = 0.01% = 0.0001)."""
    
    brokerage_bps: float = Field(default=5.0, description="Brokerage commission in bps")
    clearing_fee_bps: float = Field(default=1.0, description="Exchange clearing fee in bps")
    stamp_duty_bps: float = Field(default=0.0, description="Regulatory stamp duty in bps")
    bid_ask_spread_bps: float = Field(default=5.0, description="Typical bid-ask spread in bps")
    default_slippage_bps: float = Field(default=5.0, description="Slippage assumption in bps")
    minimum_commission: float = Field(default=0.0, description="Minimum commission per order in local currency")
    stamp_duty_cap: Optional[float] = Field(default=None, description="Maximum stamp duty per trade in local currency")
    
    @property
    def total_fixed_fee_bps(self) -> float:
        """Total deterministic fees (brokerage + clearing + stamp duty)."""
        return self.brokerage_bps + self.clearing_fee_bps + self.stamp_duty_bps
    
    @property
    def total_cost_per_trade_bps(self) -> float:
        """Total expected one-way cost including half-spread and slippage."""
        return self.total_fixed_fee_bps + (self.bid_ask_spread_bps / 2.0) + self.default_slippage_bps


class UniverseAsset(BaseModel):
    """Specification of an investable or benchmark security."""
    
    symbol: str
    name: str
    asset_class: str = "equity"
    sector: Optional[str] = None
    category: Optional[str] = None


class MarketConfig(BaseModel):
    """Market specifications and transaction parameters."""
    
    market_code: str
    market_name: str
    currency: str
    timezone: str = "UTC"
    annual_trading_days: int = 252
    benchmark_symbol: str
    universe: List[UniverseAsset] = Field(default_factory=list)
    macro_proxies: List[UniverseAsset] = Field(default_factory=list)
    transaction_costs: TransactionCostConfig = Field(default_factory=TransactionCostConfig)

    @property
    def trading_days_per_year(self) -> int:
        """Alias for annual_trading_days."""
        return self.annual_trading_days

    @property
    def costs(self) -> TransactionCostConfig:
        """Alias for transaction_costs."""
        return self.transaction_costs


def load_market_config(market_code: str) -> MarketConfig:
    """Load market configuration from YAML file by market code (MY, US, JP, EU)."""
    settings = get_settings()
    filename_map = {
        "MY": "malaysia.yaml",
        "US": "usa.yaml",
        "JP": "japan.yaml",
        "EU": "europe.yaml",
    }
    fname = filename_map.get(market_code.upper(), f"{market_code.lower()}.yaml")
    config_path = settings.configs_dir / fname
    
    if not config_path.exists():
        # Fallback default configuration
        return MarketConfig(
            market_code=market_code.upper(),
            market_name=f"Market {market_code.upper()}",
            currency="USD",
            benchmark_symbol="SPY",
            universe=[UniverseAsset(symbol="SPY", name="S&P 500 ETF", asset_class="equity_index")],
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MarketConfig(**raw)


def load_global_config() -> Dict[str, Any]:
    """Load global cross-market setup from configs/global.yaml."""
    settings = get_settings()
    config_path = settings.configs_dir / "global.yaml"
    if not config_path.exists():
        return {"global_markets": [], "cross_market_pairs": []}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
