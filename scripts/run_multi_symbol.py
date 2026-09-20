"""Train + evaluate the PPO strategy across multiple tickers and print a summary.

Every symbol trains and evaluates on the same windows and timestep budget, so the
result answers "does this generalize past AAPL?" rather than "did it get lucky on
one stock's 2025 path?". Each run is still logged to MLflow individually (tagged
by its `symbol` param), so the underlying per-symbol detail isn't lost.

Run: python scripts/run_multi_symbol.py
"""

from trading_agent.config import Settings
from trading_agent.evaluate import run_evaluation
from trading_agent.train import train

SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

TRAIN_START = "2015-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
TOTAL_TIMESTEPS = 300_000


def main():
    results = []

    for symbol in SYMBOLS:
        print(f"\n=== {symbol}: training ===", flush=True)
        settings = Settings(
            symbol=symbol,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END,
            total_timesteps=TOTAL_TIMESTEPS,
        )
        train(settings)

        print(f"=== {symbol}: evaluating ===", flush=True)
        metrics, baseline = run_evaluation(settings)
        row = {"symbol": symbol, **metrics, **baseline}
        results.append(row)
        print(
            f"{symbol}: return={row['percent_return']:.2f}% "
            f"sharpe={row['sharpe_ratio']:.2f} "
            f"outperformance={row.get('outperformance', 0):.2f}%",
            flush=True,
        )

    print("\n\n=== Summary across symbols (2025 out-of-sample) ===")
    print(f"{'Symbol':<8}{'Return%':>10}{'Sharpe':>9}{'MaxDD%':>9}{'Trades':>8}{'B&H Ret%':>11}{'Outperf%':>10}")
    for r in results:
        print(
            f"{r['symbol']:<8}"
            f"{r['percent_return']:>10.2f}"
            f"{r['sharpe_ratio']:>9.2f}"
            f"{r['max_drawdown'] * 100:>9.2f}"
            f"{r['total_trades']:>8}"
            f"{r.get('buy_hold_return', 0):>11.2f}"
            f"{r.get('outperformance', 0):>10.2f}"
        )

    wins = sum(1 for r in results if r.get("outperformance", 0) > 0)
    avg_outperf = sum(r.get("outperformance", 0) for r in results) / len(results)
    print(f"\nBeat buy & hold in {wins}/{len(results)} symbols. Average outperformance: {avg_outperf:.2f} points.")


if __name__ == "__main__":
    main()
