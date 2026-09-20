import numpy as np
import pandas as pd

from trading_agent.env import make_env


def _synthetic_ohlcv(n=100):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1000, 5000, size=n),
        }
    )


def test_make_env_resets_and_steps():
    df = _synthetic_ohlcv()
    env = make_env(df, window_size=10)

    # window_size x (price_change, normalized_volume, rsi, trend, volatility)
    obs, _info = env.reset()
    assert obs.shape == (10, 5)

    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (10, 5)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))


def test_trade_incurs_a_fee_penalty():
    """Closing a trade should cost something, even with zero price movement."""
    df = _synthetic_ohlcv(n=60)
    df["Close"] = 100.0  # flat price: any reward must come purely from the fee
    df["Open"], df["High"], df["Low"] = 100.0, 101.0, 99.0
    env = make_env(df, window_size=10)

    env.reset()
    # Buy.value == 1 flips the initial Short position -> Long: a trade fires.
    _obs, reward, *_ = env.step(1)
    assert reward < 0


def test_make_env_default_frame_bound_covers_full_series():
    df = _synthetic_ohlcv(n=50)
    env = make_env(df, window_size=5)
    assert env.frame_bound == (5, 50)
