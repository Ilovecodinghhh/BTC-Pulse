"""
Retry and error handling utilities.
"""

import time
from functools import wraps
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import requests


# Decorator for API calls with exponential backoff
api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)


def safe_api_call(func):
    """Decorator that wraps API calls with retry + error logging."""
    @wraps(func)
    @api_retry
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API call failed in {func.__name__}: {e}")
            raise
    return wrapper
