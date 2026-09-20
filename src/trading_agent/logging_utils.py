"""Shared logging setup, replacing scattered print() calls with real logs."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(logs_dir: str = "logs", name: str = "trading_system") -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"trading_log_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls (e.g. in tests)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
