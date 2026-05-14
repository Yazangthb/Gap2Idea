"""Unit tests for gap2idea.utils."""
from __future__ import annotations

import time

import pytest

from gap2idea.utils import get_logger, retry, set_seed


def test_set_seed_is_idempotent():
    set_seed(123)
    import random
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b


def test_get_logger_idempotent():
    log1 = get_logger("test.alpha")
    log2 = get_logger("test.alpha")
    assert log1 is log2
    assert len(log1.handlers) == 1


def test_retry_succeeds_after_failures():
    attempts = {"n": 0}

    @retry(exceptions=(ValueError,), tries=3, base_delay=0.0)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_retry_exhausts_and_reraises():
    @retry(exceptions=(ValueError,), tries=2, base_delay=0.0)
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fails()
