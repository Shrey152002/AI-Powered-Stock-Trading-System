"""Paper-trading loop: polls the market, asks the trained PPO model for an
action, and submits orders through Alpaca. Intentionally simple — no queueing,
no async — a plain loop is easy to reason about and easy to explain.

Run: python -m trading_agent.cli trade
"""

import time
from datetime import datetime

import alpaca_trade_api as alpaca
import numpy as np
from stable_baselines3 import PPO

from .config import Settings
from .logging_utils import setup_logging
from .risk import is_trading_time

logger = setup_logging()


class LiveTrading:
    """Tracks portfolio state and submits orders to Alpaca to match a target strategy."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.symbol = settings.symbol

        self.portfolio = {
            "cash": settings.initial_capital,
            "positions": {},
            "history": [],
            "trades": [],
        }

        self.alpaca = alpaca.REST(
            settings.alpaca_api_key,
            settings.alpaca_api_secret,
            base_url=settings.alpaca_base_url,
        )

        self.clock = self.alpaca.get_clock()
        self.is_market_open = self.clock.is_open
        self._sync_portfolio()

        logger.info(f"Live Trading initialized. Market is {'open' if self.is_market_open else 'closed'}")

    def _sync_portfolio(self):
        try:
            account = self.alpaca.get_account()
            self.portfolio["cash"] = float(account.cash)

            positions = self.alpaca.list_positions()
            self.portfolio["positions"] = {p.symbol: float(p.qty) for p in positions}
            self.portfolio["total_value"] = float(account.portfolio_value)

            logger.info(f"Portfolio synced: ${self.portfolio['total_value']:.2f} total value")
        except Exception:
            logger.error("Error syncing portfolio", exc_info=True)

    def execute_strategy(self, strategy_func) -> bool:
        if not self.is_market_open:
            logger.info("Market is closed. Skipping execution.")
            return False

        current_date = datetime.now()
        weight = strategy_func()

        try:
            price = self.alpaca.get_latest_trade(self.symbol).price
        except Exception:
            logger.error(f"Error getting price for {self.symbol}", exc_info=True)
            return False

        current_shares = self.portfolio["positions"].get(self.symbol, 0)
        positions_value = current_shares * price
        portfolio_value = self.portfolio["cash"] + positions_value

        target_value = weight * portfolio_value
        target_shares = target_value / price if price else 0

        self._submit_rebalance_order(price, current_shares, target_shares)

        self.portfolio["history"].append(
            {
                "timestamp": current_date,
                "portfolio_value": portfolio_value,
                "cash": self.portfolio["cash"],
                "positions_value": positions_value,
            }
        )
        return True

    def _submit_rebalance_order(self, price: float, current_shares: float, target_shares: float):
        shares_to_trade = abs(target_shares - current_shares)
        if shares_to_trade < 0.01:
            return

        side = "buy" if target_shares > current_shares else "sell"

        try:
            logger.info(f"Placing order: {side} {shares_to_trade:.4f} shares of {self.symbol}")
            self.alpaca.submit_order(
                symbol=self.symbol,
                qty=shares_to_trade,
                side=side,
                type="market",
                time_in_force="day",
            )
            self.portfolio["trades"].append(
                {
                    "timestamp": datetime.now(),
                    "symbol": self.symbol,
                    "shares": shares_to_trade if side == "buy" else -shares_to_trade,
                    "price": price,
                    "value": shares_to_trade * price,
                }
            )
        except Exception:
            logger.error(f"Error executing order for {self.symbol}", exc_info=True)

        self._sync_portfolio()


def build_rl_strategy(model: PPO, alpaca_client: alpaca.REST, symbol: str, window_size: int):
    """Return a zero-arg strategy function that turns recent bars into a target weight."""

    def strategy() -> float:
        try:
            bars = alpaca_client.get_bars(symbol, "1Min", limit=window_size + 1).df
            if len(bars) < window_size + 1:
                return 0.0

            price_changes = bars["close"].pct_change().fillna(0).values[-window_size:]
            normalized_volume = (bars["volume"] / bars["volume"].max()).fillna(0).values[-window_size:]
            features = np.column_stack([price_changes, normalized_volume])
            state = features.reshape(1, window_size, 2)

            action, _ = model.predict(state, deterministic=True)
            return 1.0 if action == 1 else 0.0
        except Exception:
            logger.error("Error computing RL strategy action", exc_info=True)
            return 0.0

    return strategy


def run_live_trading_system(settings: Settings):
    """Poll the market every `update_interval_seconds` and trade using the trained model."""
    logger.info("=== Starting Live Trading System ===")

    model = PPO.load(settings.model_path)
    trader = LiveTrading(settings)
    strategy = build_rl_strategy(model, trader.alpaca, settings.symbol, settings.window_size)

    logger.info(f"System running. Trading every {settings.update_interval_seconds} seconds...")
    last_run = None

    try:
        while True:
            now = datetime.now()
            if last_run is None or (now - last_run).total_seconds() >= settings.update_interval_seconds:
                logger.info(f"--- Trading cycle at {now} ---")

                trader.clock = trader.alpaca.get_clock()
                trader.is_market_open = trader.clock.is_open

                if trader.is_market_open and is_trading_time():
                    success = trader.execute_strategy(strategy)
                    logger.info("Trading cycle completed" if success else "Trading cycle skipped")
                else:
                    logger.info("Market is closed. Waiting...")

                last_run = now

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    finally:
        logger.info("Trading system shut down")


if __name__ == "__main__":
    run_live_trading_system(Settings())
