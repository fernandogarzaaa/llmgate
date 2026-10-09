import pytest

from llmgate.resilience import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    STATE_VALUE,
    CircuitBreaker,
    RetryPolicy,
    is_retryable_status,
    parse_retry_after,
)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_state_values_for_metrics():
    assert STATE_VALUE == {"closed": 0, "half_open": 1, "open": 2}


def test_breaker_defaults():
    b = CircuitBreaker()
    assert (b.failure_threshold, b.recovery_s, b.half_open_max) == (5, 30.0, 1)
    assert b.state == CLOSED and b.failures == 0


@pytest.mark.parametrize("kw", [{"failure_threshold": 0}, {"recovery_s": -0.1}, {"half_open_max": 0}])
def test_breaker_validation(kw):
    with pytest.raises(ValueError):
        CircuitBreaker(**kw)


def test_breaker_zero_recovery_is_allowed():
    c = Clock()
    b = CircuitBreaker(1, 0, clock=c)
    b.record_failure()
    assert b.state == HALF_OPEN  # immediately eligible for a trial


def test_breaker_opens_exactly_at_threshold():
    c = Clock()
    b = CircuitBreaker(3, 10, clock=c)
    b.record_failure()
    b.record_failure()
    assert b.state == CLOSED and b.failures == 2 and b.allow()
    b.record_failure()
    assert b.state == OPEN and not b.allow()


def test_success_resets_consecutive_failures():
    b = CircuitBreaker(2, 10, clock=Clock())
    b.record_failure()
    b.record_success()
    assert b.failures == 0
    b.record_failure()
    assert b.state == CLOSED


def test_open_until_recovery_elapsed_then_half_open():
    c = Clock()
    b = CircuitBreaker(1, 10, clock=c)
    b.record_failure()
    c.t += 9.99
    assert b.state == OPEN and not b.allow()
    c.t += 0.01
    assert b.state == HALF_OPEN


def test_half_open_allows_limited_trials():
    c = Clock()
    b = CircuitBreaker(1, 10, half_open_max=2, clock=c)
    b.record_failure()
    c.t += 10
    assert b.allow() is True
    assert b.allow() is True
    assert b.allow() is False
    assert b.state == HALF_OPEN


def test_half_open_success_closes_and_resets():
    c = Clock()
    b = CircuitBreaker(2, 10, clock=c)
    b.record_failure()
    b.record_failure()
    c.t += 10
    assert b.allow()
    b.record_success()
    assert b.state == CLOSED and b.failures == 0
    b.record_failure()  # needs the full threshold again
    assert b.state == CLOSED


def test_half_open_failure_reopens_for_full_recovery():
    c = Clock()
    b = CircuitBreaker(3, 10, clock=c)
    for _ in range(3):
        b.record_failure()
    c.t += 10
    assert b.allow()
    b.record_failure()  # a single failure in half-open re-opens
    assert b.state == OPEN
    c.t += 9
    assert b.state == OPEN
    c.t += 1
    assert b.state == HALF_OPEN and b.allow()  # trial counter was reset


def test_reopen_resets_trials():
    c = Clock()
    b = CircuitBreaker(1, 5, clock=c)
    b.record_failure()
    c.t += 5
    assert b.allow()
    b.record_failure()
    c.t += 5
    assert b.allow()  # a new half-open window gets a fresh trial


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 529, 599])
def test_retryable_statuses(status):
    assert is_retryable_status(status)


@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 410, 413, 422, 424, 426, 428, 430, 499])
def test_non_retryable_statuses(status):
    assert not is_retryable_status(status)


def test_retry_policy_defaults_and_validation():
    p = RetryPolicy()
    assert (p.max_retries, p.base_s, p.max_s) == (2, 0.2, 5.0)
    RetryPolicy(0, 0, 0)
    for kw in ({"max_retries": -1}, {"base_s": -0.1}, {"max_s": -0.1}):
        with pytest.raises(ValueError):
            RetryPolicy(**kw)


def test_retry_delay_exponential_with_cap_and_jitter():
    p = RetryPolicy(4, 0.5, 3.0)
    assert [p.delay(i, lambda: 1.0) for i in range(4)] == [0.5, 1.0, 2.0, 3.0]
    assert p.delay(2, lambda: 0.5) == 1.0
    assert p.delays(lambda: 0.0) == [0.0, 0.0, 0.0, 0.0]
    assert len(RetryPolicy(0).delays()) == 0
    assert 0 <= p.delay(1) < 1.0  # real jitter stays within [0, cap)


def test_parse_retry_after():
    assert parse_retry_after(None) is None
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("100") == 30.0
    assert parse_retry_after("100", max_s=5) == 5
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after("-1") is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
