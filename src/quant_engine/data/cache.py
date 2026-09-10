"""Disk and in-memory Parquet cache manager."""

import os
import time
import hashlib
from pathlib import Path
from typing import Optional
import pandas as pd
from quant_engine.config.settings import get_settings


class DataCache:
    """Parquet disk-based cache with Time-To-Live (TTL) expiration."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or get_settings().cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _get_cache_path(self, key: str) -> Path:
        sanitized = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{sanitized}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        path = self._get_cache_path(key)
        if not path.exists():
            return None
        
        # Check TTL
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) > self.ttl_seconds:
            return None

        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._get_cache_path(key)
        try:
            df.to_parquet(path)
        except Exception:
            pass

    def clear(self) -> int:
        """Clear all cached files, returns number of files deleted."""
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return count
