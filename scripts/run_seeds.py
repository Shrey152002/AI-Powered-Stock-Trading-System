"""Train the same config across multiple random seeds and report mean/std.

RL training is noisy — a single run can get lucky or unlucky. This trains N
seeds for one symbol/config and summarizes the spread, instead of reporting one
run's number as if it were the whole story.

Run: python scripts/run_seeds.py
"""

import statistics

from trading_agent.config import Settings
from trading_agent.evaluate import run_evaluation
from trading_agent.train import train

SYMBOL = "AAPL"
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
TOTAL_TIMESTEPS = 500_000
SEEDS = [1, 2, 3]


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n=== seed {seed}: training ===", flush=True)
        settings = Settings(
            symbol=SYMBOL,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=seed,
        )
        train(settings)

        print(f"=== seed {seed}: evaluating ===", flush=True)
        metrics, baseline = run_evaluation(settings)
        row = {"seed": seed, **metrics, **baseline}
        rows.append(row)
        print(
            f"seed {seed}: return={row['percent_return']:.2f}% sharpe={row['sharpe_ratio']:.2f} "
            f"trades={row['total_trades']} outperformance={row.get('outperformance', 0):.2f}",
            flush=True,
        )

    returns = [r["percent_return"] for r in rows]
    sharpes = [r["sharpe_ratio"] for r in rows]
    outperf = [r.get("outperformance", 0) for r in rows]

    print(f"\n\n=== {SYMBOL}: {len(SEEDS)} seeds, {TOTAL_TIMESTEPS} timesteps ===")
    print(f"{'Seed':<6}{'Return%':>10}{'Sharpe':>9}{'MaxDD%':>9}{'Trades':>8}{'Outperf%':>10}")
    for r in rows:
        print(
            f"{r['seed']:<6}"
            f"{r['percent_return']:>10.2f}"
            f"{r['sharpe_ratio']:>9.2f}"
            f"{r['max_drawdown'] * 100:>9.2f}"
            f"{r['total_trades']:>8}"
            f"{r.get('outperformance', 0):>10.2f}"
        )

    print(f"\nReturn:  mean={statistics.mean(returns):.2f}%  std={statistics.pstdev(returns):.2f}")
    print(f"Sharpe:  mean={statistics.mean(sharpes):.2f}  std={statistics.pstdev(sharpes):.2f}")
    print(f"Outperf: mean={statistics.mean(outperf):.2f}  std={statistics.pstdev(outperf):.2f}")
    wins = sum(1 for o in outperf if o > 0)
    print(f"Beat buy & hold in {wins}/{len(SEEDS)} seeds.")


if __name__ == "__main__":
    main()
