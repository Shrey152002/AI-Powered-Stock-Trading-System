"""Central configuration, loaded from environment variables / a .env file.

Nothing here is hardcoded — secrets and tunables come from the environment so the
same code runs unchanged across a laptop, CI, and a container.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Trading ---
    symbol: str = "AAPL"
    window_size: int = 10

    # --- Training ---
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    total_timesteps: int = 40_000
    seed: int | None = None

    # --- Evaluation ---
    test_start: str = "2023-01-01"
    test_end: str = "2023-12-31"

    # --- Risk management ---
    initial_capital: float = 100_000.0
    max_position_size: float = 0.25
    max_daily_loss: float = 0.03
    update_interval_seconds: int = 60

    # --- Storage ---
    models_dir: str = "models"
    artifact_name: str = ""  # empty = auto-derive from symbol, e.g. "ppo_msft_trading"
    data_cache_dir: str = "data/cache"
    reports_dir: str = "reports"
    logs_dir: str = "logs"

    # --- MLflow ---
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "ppo-trading"

    # --- Broker / data provider credentials (set via .env, never hardcode) ---
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpha_vantage_api_key: str = ""

    @property
    def model_path(self) -> str:
        name = self.artifact_name or f"ppo_{self.symbol.lower()}_trading"
        if self.seed is not None:
            name = f"{name}_seed{self.seed}"
        return f"{self.models_dir}/{name}"

    @property
    def plot_path(self) -> str:
        suffix = f"_seed{self.seed}" if self.seed is not None else ""
        return f"{self.reports_dir}/performance_{self.symbol.lower()}{suffix}.png"
