"""Backtest a trained model and compare it against a buy-and-hold baseline.

Run: python -m trading_agent.cli evaluate
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a window, always saves to disk
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import Settings
from .data import fetch_ohlcv
from .env import make_env
from .logging_utils import setup_logging

logger = setup_logging()

DEFAULT_METRICS = {
    "total_return": 0.0,
    "percent_return": 0.0,
    "win_rate": 0.0,
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "final_portfolio_value": 1.0,
    "total_trades": 0,
}


def evaluate_model_accuracy(
    model_path: str,
    symbol: str,
    test_start: str,
    test_end: str,
    window_size: int = 10,
    plot_path: str = "reports/performance.png",
    data_cache_dir: str = "data/cache",
) -> dict:
    """Backtest `model_path` on out-of-sample data and return performance metrics."""
    try:
        model = PPO.load(model_path)

        logger.info(f"Downloading test data from {test_start} to {test_end}")
        test_df = fetch_ohlcv(symbol, test_start, test_end, cache_dir=data_cache_dir)
        test_df = test_df.reset_index()
        original_dates = test_df["Date"].copy()
        test_df = test_df.drop(columns=["Date"])

        if len(test_df) < 20:
            raise ValueError("Not enough test data available")

        test_env = make_env(test_df, window_size=window_size, frame_bound=(window_size, len(test_df)))
        test_env = DummyVecEnv([lambda: test_env])

        actions, portfolio_values = [], []
        obs = test_env.reset()
        done = False

        while not done:
            # deterministic=True: PPO.predict() samples stochastically by default,
            # which made repeated backtests of the *same* frozen model disagree.
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action[0]))

            obs, _reward, done, info = test_env.step(action)

            # gym_anytrading's info dict exposes `total_profit` (starts at 1.0,
            # compounds correctly, and already bakes in trade fees) — not
            # `portfolio_value`. An earlier version of this code looked for the
            # wrong key, silently fell back to treating each raw dollar reward as
            # a 1% return, and produced impossible results (e.g. 0% drawdown).
            current_portfolio_value = info[0]["total_profit"] if info else 1.0
            portfolio_values.append(current_portfolio_value)

        if not portfolio_values:
            logger.warning("No portfolio values were tracked during evaluation")
            portfolio_values = [1.0, 1.0]

        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        percent_return = (final_value / initial_value - 1) * 100

        # A "trade" is a change in desired position, not an environment step —
        # counting steps (as an earlier version did) produced the same trade
        # count for every symbol, since every 2025 test window has ~238 days.
        total_trades = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])

        daily_returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-9)
        profitable_days = int(np.sum(daily_returns > 0))
        win_rate = profitable_days / len(daily_returns) if len(daily_returns) else 0.0

        returns = daily_returns
        sharpe_ratio = (
            float(returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0 and len(returns) > 0
            else 0.0
        )

        if len(portfolio_values) > 1:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0
        else:
            max_drawdown = 0.0

        metrics = {
            "total_return": final_value - initial_value,
            "percent_return": percent_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_portfolio_value": final_value,
            "total_trades": total_trades,
        }

        logger.info("=== Model Accuracy Metrics ===")
        logger.info(f"Test Period: {test_start} to {test_end}")
        logger.info(f"Total Return: {percent_return:.2f}%")
        logger.info(f"Final Portfolio Value: ${final_value:.2f}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Trades: {total_trades}")

        test_df.index = original_dates
        plot_performance(test_df, portfolio_values, actions, window_size, plot_path)

        if mlflow.active_run():
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
            if Path(plot_path).exists():
                mlflow.log_artifact(plot_path)

        return metrics

    except Exception:
        logger.error("Error during model evaluation", exc_info=True)
        return dict(DEFAULT_METRICS)


def plot_performance(test_df, portfolio_values, actions, window_offset, plot_path="reports/performance.png"):
    """Render price/portfolio comparison, buy/sell signals and drawdown to a PNG file."""
    if not portfolio_values:
        logger.warning("No portfolio values available to plot")
        return

    plt.figure(figsize=(15, 12))

    ax1 = plt.subplot(3, 1, 1)
    price_data = test_df["Close"].values[window_offset:]
    max_len = min(len(price_data), len(portfolio_values))
    if max_len == 0:
        logger.warning("No data to plot after alignment")
        return

    dates = pd.to_datetime(test_df.index)[window_offset : window_offset + max_len]
    price_data = price_data[:max_len]
    portfolio_values = portfolio_values[:max_len]

    normed_price = price_data / price_data[0] if len(price_data) > 0 else []
    normed_portfolio = np.array(portfolio_values) / portfolio_values[0] if portfolio_values else []

    if len(normed_price) > 0:
        ax1.plot(dates[: len(normed_price)], normed_price, "b-", alpha=0.7, label="Price (normalized)")
    if len(normed_portfolio) > 0:
        ax1.plot(dates[: len(normed_portfolio)], normed_portfolio, "g-", linewidth=2, label="Portfolio Value (normalized)")

    ax1.set_title("Model Performance Comparison")
    ax1.set_ylabel("Normalized Value")
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    actions = actions[:max_len] if len(actions) > max_len else actions
    buy_signals = [i for i, a in enumerate(actions) if a == 1]
    sell_signals = [i for i, a in enumerate(actions) if a == 0]

    ax2.plot(dates[: len(price_data)], price_data, "k-", alpha=0.3)
    if buy_signals:
        buy_dates = [dates[i] for i in buy_signals if i < len(dates)]
        buy_prices = [price_data[i] for i in buy_signals if i < len(price_data)]
        if buy_dates and buy_prices:
            ax2.scatter(buy_dates, buy_prices, color="green", marker="^", s=100, label="Buy")
    if sell_signals:
        sell_dates = [dates[i] for i in sell_signals if i < len(dates)]
        sell_prices = [price_data[i] for i in sell_signals if i < len(price_data)]
        if sell_dates and sell_prices:
            ax2.scatter(sell_dates, sell_prices, color="red", marker="v", s=100, label="Sell")

    ax2.set_title("Trading Signals")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True)

    if portfolio_values:
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        ax3.fill_between(dates[: len(drawdown)], 0, drawdown, color="red", alpha=0.3)
        ax3.set_title("Portfolio Drawdown")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_ylim(bottom=0)
        ax3.invert_yaxis()
        ax3.grid(True)

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved performance plot to {plot_path}")


def compare_to_baseline(model_metrics: dict, symbol: str, test_start: str, test_end: str, data_cache_dir: str = "data/cache") -> dict:
    """Compare model performance against a simple buy-and-hold strategy."""
    try:
        test_df = fetch_ohlcv(symbol, test_start, test_end, cache_dir=data_cache_dir)
        if len(test_df) < 2:
            logger.warning("Not enough data to compare to baseline")
            return {}

        start_price = float(test_df["Close"].iloc[0])
        end_price = float(test_df["Close"].iloc[-1])
        buy_hold_return = (end_price / start_price - 1) * 100

        daily_returns = test_df["Close"].pct_change().dropna()
        std = float(daily_returns.std())
        mean = float(daily_returns.mean())
        buy_hold_sharpe = mean / std * np.sqrt(252) if std else 0.0

        peak = test_df["Close"].cummax()
        drawdown = (peak - test_df["Close"]) / peak
        buy_hold_max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        outperformance = model_metrics["percent_return"] - buy_hold_return

        logger.info("=== Strategy Comparison ===")
        logger.info(f"Period: {test_start} to {test_end}")
        logger.info(f"Model Return: {model_metrics['percent_return']:.2f}% | Buy & Hold: {buy_hold_return:.2f}%")
        logger.info(f"Model Sharpe: {model_metrics['sharpe_ratio']:.2f} | Buy & Hold: {buy_hold_sharpe:.2f}")
        logger.info(f"Model Max DD: {model_metrics['max_drawdown']:.2%} | Buy & Hold: {buy_hold_max_dd:.2%}")
        logger.info(
            f"Model {'outperformed' if outperformance > 0 else 'underperformed'} buy & hold by {abs(outperformance):.2f}%"
        )

        result = {
            "buy_hold_return": buy_hold_return,
            "buy_hold_sharpe": buy_hold_sharpe,
            "buy_hold_max_dd": buy_hold_max_dd,
            "outperformance": outperformance,
        }

        if mlflow.active_run():
            mlflow.log_metrics({f"baseline_{k}": v for k, v in result.items()})

        return result
    except Exception:
        logger.error("Error comparing to baseline", exc_info=True)
        return {}


def run_evaluation(settings: Settings) -> tuple[dict, dict]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"eval-{settings.symbol}"):
        metrics = evaluate_model_accuracy(
            model_path=settings.model_path,
            symbol=settings.symbol,
            test_start=settings.test_start,
            test_end=settings.test_end,
            window_size=settings.window_size,
            plot_path=settings.plot_path,
            data_cache_dir=settings.data_cache_dir,
        )
        baseline = compare_to_baseline(
            metrics,
            symbol=settings.symbol,
            test_start=settings.test_start,
            test_end=settings.test_end,
            data_cache_dir=settings.data_cache_dir,
        )
        return metrics, baseline


if __name__ == "__main__":
    run_evaluation(Settings())
