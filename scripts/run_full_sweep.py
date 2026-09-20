"""Full sweep: 3 seeds x remaining symbols, same treatment as scripts/run_seeds.py.

AAPL's 3 seeds were already run via run_seeds.py — reused here instead of
re-training, so this only pays for MSFT/AMZN/GOOGL/META.

Run: python scripts/run_full_sweep.py
"""

import statistics

from trading_agent.config import Settings
from trading_agent.evaluate import run_evaluation
from trading_agent.train import train

SYMBOLS = ["MSFT", "AMZN", "GOOGL", "META"]
SEEDS = [1, 2, 3]

TRAIN_START = "2015-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
TOTAL_TIMESTEPS = 500_000

# Already computed by scripts/run_seeds.py.
AAPL_RESULTS = [
    {"seed": 1, "percent_return": 18.13, "sharpe_ratio": 1.72, "total_trades": 6, "outperformance": 5.64},
    {"seed": 2, "percent_return": 4.72, "sharpe_ratio": 1.42, "total_trades": 6, "outperformance": -7.77},
    {"seed": 3, "percent_return": 17.93, "sharpe_ratio": 1.35, "total_trades": 4, "outperformance": 5.44},
]


def run_symbol(symbol):
    rows = []
    for seed in SEEDS:
        print(f"\n=== {symbol} seed {seed}: training ===", flush=True)
        settings = Settings(
            symbol=symbol,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=seed,
        )
        train(settings)

        print(f"=== {symbol} seed {seed}: evaluating ===", flush=True)
        metrics, baseline = run_evaluation(settings)
        row = {"seed": seed, **metrics, **baseline}
        rows.append(row)
        print(
            f"{symbol} seed {seed}: return={row['percent_return']:.2f}% sharpe={row['sharpe_ratio']:.2f} "
            f"trades={row['total_trades']} outperformance={row.get('outperformance', 0):.2f}",
            flush=True,
        )
    return rows


def summarize(symbol, rows):
    returns = [r["percent_return"] for r in rows]
    sharpes = [r["sharpe_ratio"] for r in rows]
    outperf = [r.get("outperformance", 0) for r in rows]
    trades = [r["total_trades"] for r in rows]
    wins = sum(1 for o in outperf if o > 0)
    return {
        "symbol": symbol,
        "return_mean": statistics.mean(returns),
        "return_std": statistics.pstdev(returns),
        "sharpe_mean": statistics.mean(sharpes),
        "sharpe_std": statistics.pstdev(sharpes),
        "outperf_mean": statistics.mean(outperf),
        "outperf_std": statistics.pstdev(outperf),
        "trades_mean": statistics.mean(trades),
        "wins": wins,
        "n": len(rows),
    }


def main():
    all_summaries = [summarize("AAPL", AAPL_RESULTS)]

    for symbol in SYMBOLS:
        rows = run_symbol(symbol)
        all_summaries.append(summarize(symbol, rows))

    print("\n\n=== Full sweep summary (mean +/- std across 3 seeds, 2025 out-of-sample) ===")
    print(f"{'Symbol':<8}{'Return%':>16}{'Sharpe':>14}{'Trades':>8}{'Outperf%':>16}{'Wins':>7}")
    for s in all_summaries:
        print(
            f"{s['symbol']:<8}"
            f"{s['return_mean']:>10.2f} +/-{s['return_std']:<5.2f}"
            f"{s['sharpe_mean']:>9.2f} +/-{s['sharpe_std']:<4.2f}"
            f"{s['trades_mean']:>8.1f}"
            f"{s['outperf_mean']:>10.2f} +/-{s['outperf_std']:<5.2f}"
            f"{s['wins']:>4}/{s['n']}"
        )

    total_wins = sum(s["wins"] for s in all_summaries)
    total_runs = sum(s["n"] for s in all_summaries)
    print(f"\nBeat buy & hold in {total_wins}/{total_runs} symbol-seed runs across {len(all_summaries)} symbols.")


if __name__ == "__main__":
    main()
