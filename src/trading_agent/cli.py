"""Single entry point for every workflow: train, evaluate, trade.

Examples:
    python -m trading_agent.cli train
    python -m trading_agent.cli evaluate
    python -m trading_agent.cli trade
"""

import argparse

from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="AI-powered stock trading agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the PPO agent and log the run to MLflow")
    subparsers.add_parser("evaluate", help="Backtest the trained model against buy-and-hold")
    subparsers.add_parser("trade", help="Run the live (paper trading) loop")

    args = parser.parse_args()
    settings = Settings()

    if args.command == "train":
        from .train import train

        train(settings)
    elif args.command == "evaluate":
        from .evaluate import run_evaluation

        run_evaluation(settings)
    elif args.command == "trade":
        from .live_trading import run_live_trading_system

        run_live_trading_system(settings)


if __name__ == "__main__":
    main()
