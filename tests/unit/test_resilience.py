from llmgate.resilience import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, RetryPolicy, is_retryable_status


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_after_threshold():
    c = Clock()
    b = CircuitBreaker(3, 10, clock=c)
    for _ in range(3):
        assert b.allow()
        b.record_failure()
    assert b.state == OPEN
    assert not b.allow()


def test_breaker_half_open_then_closes():
    c = Clock()
    b = CircuitBreaker(1, 10, clock=c)
    b.record_failure()
    c.t = 10
    assert b.state == HALF_OPEN
    assert b.allow()
    b.record_success()
    assert b.state == CLOSED


def test_retryable_statuses():
    assert is_retryable_status(429)
    assert is_retryable_status(503)
    assert not is_retryable_status(400)


def test_retry_delays_capped():
    p = RetryPolicy(3, 1.0, 2.0)
    assert p.delays(lambda: 0.999) == [0.999, 1.998, 1.998]
