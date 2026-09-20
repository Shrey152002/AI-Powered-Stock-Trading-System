from datetime import datetime, timezone

from trading_agent.risk import check_risk_limits, is_trading_time


def test_check_risk_limits_passes_within_bounds():
    portfolio = {"cash": 50_000, "positions": {"AAPL": 10}}
    trades = {
        "prices": {"AAPL": 100.0},
        "target_positions": {"AAPL": 100},  # 100 * 100 = 10,000 out of 51,000
    }
    assert check_risk_limits(portfolio, trades, max_position_pct=0.25) is True


def test_check_risk_limits_blocks_oversized_position():
    portfolio = {"cash": 1_000, "positions": {}}
    trades = {
        "prices": {"AAPL": 100.0},
        "target_positions": {"AAPL": 100},  # 10,000 value vs 1,000 cash -> way over 25%
    }
    assert check_risk_limits(portfolio, trades, max_position_pct=0.25) is False


def test_check_risk_limits_blocks_daily_loss():
    portfolio = {
        "cash": 900,
        "positions": {},
        "start_of_day_value": 1000,
    }
    trades = {"prices": {}, "target_positions": {}}
    # 900 / 1000 - 1 = -10%, breaches a 3% daily loss limit
    assert check_risk_limits(portfolio, trades, max_daily_loss_pct=0.03) is False


def test_is_trading_time_weekday_market_hours():
    # Tuesday 2024-01-02, 10:00 Eastern -> 15:00 UTC
    dt = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    assert is_trading_time(dt) is True


def test_is_trading_time_outside_market_hours():
    # Tuesday 2024-01-02, 20:00 Eastern -> well after close
    dt = datetime(2024, 1, 3, 1, 0, tzinfo=timezone.utc)
    assert is_trading_time(dt) is False


def test_is_trading_time_weekend():
    # Saturday
    dt = datetime(2024, 1, 6, 15, 0, tzinfo=timezone.utc)
    assert is_trading_time(dt) is False
