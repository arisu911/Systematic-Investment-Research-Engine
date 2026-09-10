"""Engine configuration settings and directory path resolvers."""

import os
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


class EngineSettings(BaseModel):
    """Global configuration settings for the quantitative engine."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")
    cache_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "cache")
    configs_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "configs")
    experiments_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "experiments")
    
    default_risk_free_rate: float = 0.03
    default_annual_trading_days: int = 252
    cache_expiry_hours: int = 24
    allow_demo_fallback: bool = True


_settings = None


def get_settings() -> EngineSettings:
    """Return singleton EngineSettings instance with verified directories."""
    global _settings
    if _settings is None:
        _settings = EngineSettings()
        # Ensure directories exist
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
        _settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _settings.experiments_dir.mkdir(parents=True, exist_ok=True)
    return _settings
