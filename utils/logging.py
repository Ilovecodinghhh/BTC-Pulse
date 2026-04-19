"""
Logging setup for BTC-Pulse using loguru.
"""

import sys
from loguru import logger
from utils.config import get_log_dir


def setup_logger(module_name: str = "btcpulse"):
    """Configure loguru logger with file and console output."""
    log_dir = get_log_dir()

    # Remove default handler
    logger.remove()

    # Console output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
               "<level>{message}</level>",
        level="INFO",
    )

    # File output with rotation
    logger.add(
        log_dir / f"{module_name}_{{time:YYYY-MM-DD}}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
    )

    return logger
