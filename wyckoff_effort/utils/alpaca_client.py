"""
Alpaca Markets data client for live and historical market data.

Uses the SIP feed for real-time tick-level trade data.
Provides historical bars and trades for backtesting.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest,
        StockTradesRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Run: pip install alpaca-py")


def _get_client():
    """Create an Alpaca historical data client from environment variables."""
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
            "Copy .env.example to .env and fill in your keys."
        )

    return StockHistoricalDataClient(api_key, secret_key)


def _timeframe_from_string(tf_str):
    """Convert a string like '1min', '5min', '1h', '1d' to Alpaca TimeFrame."""
    mapping = {
        '1min': TimeFrame(1, TimeFrameUnit.Minute),
        '5min': TimeFrame(5, TimeFrameUnit.Minute),
        '15min': TimeFrame(15, TimeFrameUnit.Minute),
        '30min': TimeFrame(30, TimeFrameUnit.Minute),
        '1h': TimeFrame(1, TimeFrameUnit.Hour),
        '1d': TimeFrame(1, TimeFrameUnit.Day),
    }
    tf = mapping.get(tf_str.lower())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {tf_str}. Use one of {list(mapping.keys())}")
    return tf


def get_historical_bars(symbol, timeframe='1min', start_date=None, end_date=None, limit=10000):
    """
    Fetch historical OHLCV bars from Alpaca.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'QQQ' for NQ proxy)
        timeframe: '1min', '5min', '15min', '30min', '1h', '1d'
        start_date: datetime or string 'YYYY-MM-DD'
        end_date: datetime or string 'YYYY-MM-DD'
        limit: Max number of bars

    Returns:
        pd.DataFrame with OHLCV columns indexed by timestamp
    """
    if not ALPACA_AVAILABLE:
        raise ImportError("alpaca-py is required. Install with: pip install alpaca-py")

    client = _get_client()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_timeframe_from_string(timeframe),
        start=start_date,
        end=end_date,
        limit=limit,
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel('symbol')

    # Rename columns to match the project's conventions (lowercase)
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'trade_count': 'num_trades',
        'vwap': 'vwap',
    })

    logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
    return df


def get_historical_trades(symbol, start_date=None, end_date=None, limit=50000):
    """
    Fetch tick-level trade data from Alpaca SIP feed.
    Each trade has price, size, and exchange — no bid/ask split, but
    we can classify trades using tick rule.

    Args:
        symbol: Ticker symbol
        start_date / end_date: datetime or string
        limit: Max trades to fetch

    Returns:
        pd.DataFrame with columns: Price, Size, Exchange, Delta (estimated)
    """
    if not ALPACA_AVAILABLE:
        raise ImportError("alpaca-py is required. Install with: pip install alpaca-py")

    client = _get_client()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None:
        start_date = datetime.now() - timedelta(days=1)
    if end_date is None:
        end_date = datetime.now()

    request = StockTradesRequest(
        symbol_or_symbols=symbol,
        start=start_date,
        end=end_date,
        limit=limit,
    )

    trades = client.get_stock_trades(request)
    df = trades.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel('symbol')

    df = df.rename(columns={
        'price': 'price',
        'size': 'size',
        'exchange': 'exchange',
    })

    # Classify trades using tick rule:
    #   If price > previous price → buyer-initiated (positive delta)
    #   If price < previous price → seller-initiated (negative delta)
    #   If price == previous → same as previous classification
    prices = df['price'].values
    sizes = df['size'].values
    tick_delta = np.zeros(len(df), dtype=np.int64)

    for i in range(1, len(df)):
        if prices[i] > prices[i - 1]:
            tick_delta[i] = sizes[i]
        elif prices[i] < prices[i - 1]:
            tick_delta[i] = -sizes[i]
        else:
            tick_delta[i] = tick_delta[i - 1]  # inherit previous direction

    df['delta'] = tick_delta

    logger.info(f"Fetched {len(df)} trades for {symbol}")
    return df


def trades_to_bars(trades_df, timeframe='1min'):
    """
    Aggregate tick-level trade data into OHLCV bars with real delta.

    Args:
        trades_df: DataFrame from get_historical_trades()
        timeframe: Resample frequency ('1min', '5min', etc.)

    Returns:
        pd.DataFrame with OHLCV + Delta + CVD
    """
    if trades_df.empty:
        return trades_df

    price = trades_df['price']
    size = trades_df['size']

    resampled = trades_df.resample(timeframe).agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum',
        'delta': 'sum',
    })

    # Flatten MultiIndex columns
    resampled.columns = ['open', 'high', 'low', 'close', 'volume', 'delta']
    resampled = resampled.dropna(subset=['open'])
    resampled = resampled[resampled['volume'] > 0]
    resampled['cvd'] = resampled['delta'].cumsum()

    return resampled
