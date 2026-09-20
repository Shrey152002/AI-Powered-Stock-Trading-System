"""The custom trading environment the PPO agent is trained and evaluated on."""

import numpy as np
from gym_anytrading.envs import Actions, Positions, StocksEnv


class MyStockEnv(StocksEnv):
    """StocksEnv with richer features and a trade-cost-aware reward.

    Features: price % change, normalized volume, RSI(14), price vs. its 20-day
    moving average (trend), and rolling volatility — instead of just the first two.

    Reward: the base class only rewards realized price movement and never
    penalizes the act of trading itself, even though `trade_fee_ask_percent` /
    `trade_fee_bid_percent` already silently reduce `total_profit` at close.
    That mismatch gives a training agent no reason to avoid overtrading. Here the
    same fee is subtracted from the reward at the moment a trade fires, so the
    incentive during training matches the cost actually paid during evaluation.
    """

    def __init__(self, df, window_size, frame_bound):
        self.custom_df = df
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)

    def _process_data(self):
        close = self.df["Close"]
        volume = self.df["Volume"]
        prices = close.to_numpy()

        pct_change = close.pct_change().fillna(0)
        norm_volume = (volume / volume.max()).fillna(0)

        # RSI(14), scaled to 0-1. Manual Wilder's RSI to avoid pulling in `ta`
        # just for one indicator and to keep control over the warm-up fill value.
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(50) / 100.0

        # Trend: how far price sits from its own 20-day average.
        sma20 = close.rolling(window=20).mean()
        trend = ((close - sma20) / sma20).fillna(0)

        # Volatility: how choppy the last 10 days of returns have been.
        volatility = pct_change.rolling(window=10).std().fillna(0)

        signal_features = np.column_stack(
            [
                pct_change.to_numpy(),
                norm_volume.to_numpy(),
                rsi.to_numpy(),
                trend.to_numpy(),
                volatility.to_numpy(),
            ]
        )
        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0.0

        trade = (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        )

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

            round_trip_fee_pct = (self.trade_fee_ask_percent + self.trade_fee_bid_percent) / 2
            step_reward -= current_price * round_trip_fee_pct

        return step_reward


def make_env(df, window_size: int = 10, frame_bound: tuple | None = None) -> MyStockEnv:
    if frame_bound is None:
        frame_bound = (window_size, len(df))
    return MyStockEnv(df=df, window_size=window_size, frame_bound=frame_bound)
