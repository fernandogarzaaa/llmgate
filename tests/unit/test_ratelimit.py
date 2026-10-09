import pytest

from llmgate.ratelimit import Limits, RateLimiter, TokenBucket


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_allows_up_to_capacity_then_rejects():
    c = Clock()
    b = TokenBucket(3, 1.0, c)
    assert all(b.try_acquire().allowed for _ in range(3))
    d = b.try_acquire()
    assert not d.allowed
    assert d.retry_after_s == pytest.approx(1.0)


def test_bucket_refills_over_time():
    c = Clock()
    b = TokenBucket(2, 1.0, c)
    b.try_acquire(2)
    c.t = 1.0
    assert b.try_acquire().allowed
    assert not b.try_acquire().allowed


async def test_rpm_limit():
    c = Clock()
    rl = RateLimiter(c)
    lim = Limits(rpm=2)
    assert (await rl.check("t", lim, 0))[0].allowed
    assert (await rl.check("t", lim, 0))[0].allowed
    d, which = await rl.check("t", lim, 0)
    assert not d.allowed and which == "rpm"


async def test_tenants_are_isolated():
    rl = RateLimiter(Clock())
    lim = Limits(rpm=1)
    assert (await rl.check("a", lim, 0))[0].allowed
    assert (await rl.check("b", lim, 0))[0].allowed


async def test_unlimited():
    rl = RateLimiter(Clock())
    for _ in range(100):
        assert (await rl.check("t", Limits(), 10_000))[0].allowed
