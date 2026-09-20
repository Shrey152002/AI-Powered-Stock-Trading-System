"""Pure risk-management checks, kept dependency-free so they're trivial to unit test."""

import logging
from datetime import datetime, time, timedelta, timezone

logger = logging.getLogger(__name__)


def check_risk_limits(portfolio: dict, trades: dict, max_position_pct: float = 0.25, max_daily_loss_pct: float = 0.03) -> bool:
    """Return False if the proposed trades would breach position-size or daily-loss limits."""
    portfolio_value = portfolio["cash"] + sum(
        portfolio["positions"].get(symbol, 0) * price for symbol, price in trades["prices"].items()
    )

    for symbol, shares in trades["target_positions"].items():
        position_value = shares * trades["prices"].get(symbol, 0)
        position_pct = position_value / portfolio_value if portfolio_value else 0

        if position_pct > max_position_pct:
            logger.warning(f"Position size limit exceeded for {symbol}: {position_pct:.1%}")
            return False

    if "start_of_day_value" in portfolio and portfolio["start_of_day_value"]:
        current_loss_pct = (portfolio_value / portfolio["start_of_day_value"]) - 1
        if current_loss_pct < -max_daily_loss_pct:
            logger.warning(f"Daily loss limit exceeded: {-current_loss_pct:.1%}")
            return False

    return True


def is_trading_time(now: datetime | None = None) -> bool:
    """True during US equity market hours (9:30-16:00 Eastern, weekdays)."""
    now = now or datetime.now(timezone.utc)
    now_et = now.astimezone(timezone(timedelta(hours=-5)))

    if now_et.weekday() >= 5:
        return False

    return time(9, 30) <= now_et.time() <= time(16, 0)
