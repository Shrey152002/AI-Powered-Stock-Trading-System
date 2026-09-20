from trading_agent.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.symbol == "AAPL"
    assert settings.window_size == 10
    assert settings.max_position_size == 0.25


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("SYMBOL", "MSFT")
    monkeypatch.setenv("TOTAL_TIMESTEPS", "1000")
    settings = Settings(_env_file=None)
    assert settings.symbol == "MSFT"
    assert settings.total_timesteps == 1000


def test_model_path_combines_dir_and_name():
    settings = Settings(_env_file=None, models_dir="models", artifact_name="ppo_aapl_trading")
    assert settings.model_path == "models/ppo_aapl_trading"


def test_model_path_auto_derives_from_symbol():
    settings = Settings(_env_file=None, symbol="MSFT", artifact_name="")
    assert settings.model_path == "models/ppo_msft_trading"


def test_plot_path_is_symbol_specific():
    settings = Settings(_env_file=None, symbol="GOOGL", reports_dir="reports")
    assert settings.plot_path == "reports/performance_googl.png"
