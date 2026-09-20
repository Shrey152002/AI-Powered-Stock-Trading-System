"""Market data access with a simple on-disk cache.

The cache is a poor-man's data versioning layer: every (symbol, start, end)
request is pinned to a CSV file, so a training run can always be reproduced
from the same input data without re-hitting the network.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(symbol: str, start: str, end: str, cache_dir: str = "data/cache", use_cache: bool = True) -> pd.DataFrame:
    """Fetch OHLCV data for `symbol` between `start` and `end`, caching to disk."""
    cache_path = Path(cache_dir) / f"{symbol}_{start}_{end}.csv"

    if use_cache and cache_path.exists():
        logger.info(f"Loading cached data: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    logger.info(f"Downloading {symbol} data from {start} to {end}")
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} between {start} and {end}")

    if isinstance(df.columns, pd.MultiIndex):
        # Recent yfinance versions return (field, ticker) columns even for a single symbol.
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLUMNS].dropna()

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"Cached data to {cache_path}")

    return df
