"""Train the PPO trading agent, tracking every run with MLflow.

Run: python -m trading_agent.cli train
"""

from pathlib import Path

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import Settings
from .data import fetch_ohlcv
from .env import make_env
from .logging_utils import setup_logging

logger = setup_logging()


def train(settings: Settings) -> str:
    """Train a PPO agent on `settings.symbol` and return the saved model path."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = fetch_ohlcv(
        settings.symbol,
        settings.train_start,
        settings.train_end,
        cache_dir=settings.data_cache_dir,
    )
    env = DummyVecEnv([lambda: make_env(df, settings.window_size)])

    run_name = f"ppo-{settings.symbol}" + (f"-seed{settings.seed}" if settings.seed is not None else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "symbol": settings.symbol,
                "window_size": settings.window_size,
                "train_start": settings.train_start,
                "train_end": settings.train_end,
                "total_timesteps": settings.total_timesteps,
                "seed": settings.seed,
                "algo": "PPO",
                "policy": "MlpPolicy",
                "train_rows": len(df),
            }
        )

        logger.info(f"Training PPO on {settings.symbol} for {settings.total_timesteps} timesteps (seed={settings.seed})")
        model = PPO("MlpPolicy", env, verbose=1, seed=settings.seed)
        model.learn(total_timesteps=settings.total_timesteps)

        Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
        model.save(settings.model_path)
        model_zip = f"{settings.model_path}.zip"

        mlflow.log_artifact(model_zip)
        logger.info(f"Model trained and saved to {model_zip}")

        return model_zip


if __name__ == "__main__":
    train(Settings())
