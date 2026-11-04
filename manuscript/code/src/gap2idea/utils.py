"""Cross-cutting helpers: logging, seeding, retry."""
from __future__ import annotations

import logging
import os
import random
import time
from functools import wraps
from typing import Callable, TypeVar

import numpy as np

LOG_FMT = "%(asctime)s %(levelname)s %(name)s :: %(message)s"

T = TypeVar("T")


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Module-level logger configured once, idempotent."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FMT))
        logger.addHandler(h)
    logger.setLevel(level or int(os.getenv("GAP2IDEA_LOG_LEVEL", logging.INFO)))
    logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed all sources of randomness we control. Determinism is best-effort."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def retry(
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    tries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """Exponential-backoff retry decorator. Logs each failure."""
    log = get_logger("gap2idea.retry")

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        raise
                    log.warning("%s attempt %d/%d failed: %s", fn.__name__, attempt, tries, e)
                    time.sleep(min(delay, max_delay))
                    delay *= 2
            # unreachable
            raise RuntimeError("retry: exhausted without return")

        return wrapper

    return deco
