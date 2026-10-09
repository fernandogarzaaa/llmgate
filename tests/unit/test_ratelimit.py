import math

import pytest

from llmgate.ratelimit import Decision, Limits, RateLimiter, RedisRateLimiter, TokenBucket, make_limiter


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ----------------------------------------------------------------------------- TokenBucket
def test_decision_defaults():
    d = Decision(True)
    assert d.retry_after_s == 0.0 and d.remaining == 0.0


@pytest.mark.parametrize("cap,rate", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_bucket_rejects_non_positive_parameters(cap, rate):
    with pytest.raises(ValueError):
        TokenBucket(cap, rate, Clock())


def test_bucket_accepts_tiny_positive_parameters():
    b = TokenBucket(0.5, 0.01, Clock())
    assert b.capacity == 0.5 and b.refill_per_s == 0.01


def test_bucket_starts_full_and_drains():
    b = TokenBucket(3, 1.0, Clock())
    assert b.tokens == 3
    d = b.try_acquire()
    assert d.allowed and d.remaining == 2 and d.retry_after_s == 0.0
    assert b.try_acquire(2).allowed
    assert b.tokens == 0


def test_bucket_rejection_reports_exact_wait():
    c = Clock()
    b = TokenBucket(10, 2.0, c)  # 2 tokens/s
    b.try_acquire(9)  # 1 left
    d = b.try_acquire(4)  # needs 3 more -> 1.5 s
    assert not d.allowed
    assert d.retry_after_s == pytest.approx(1.5)
    assert d.remaining == pytest.approx(1.0)
    assert b.tokens == pytest.approx(1.0)  # a rejection does not consume


def test_bucket_refill_rate_and_cap():
    c = Clock()
    b = TokenBucket(10, 2.0, c)
    b.try_acquire(10)
    c.t = 1.5
    assert b.tokens == pytest.approx(3.0)  # 1.5 s * 2 tokens/s
    c.t = 100
    assert b.tokens == 10  # never above capacity


def test_bucket_clock_going_backwards_does_not_drain():
    c = Clock(10)
    b = TokenBucket(5, 1.0, c)
    b.try_acquire(5)
    c.t = 5  # clock jumped back
    assert b.tokens == 0
    c.t = 6
    assert b.tokens == pytest.approx(1.0)


def test_bucket_exact_amount_is_allowed():
    b = TokenBucket(2, 1.0, Clock())
    assert b.try_acquire(2).allowed


def test_bucket_zero_request_is_allowed_negative_is_error():
    b = TokenBucket(1, 1.0, Clock())
    b.try_acquire(1)
    assert b.try_acquire(0).allowed
    with pytest.raises(ValueError):
        b.try_acquire(-1)


def test_bucket_request_larger_than_capacity_never_fits():
    b = TokenBucket(5, 1.0, Clock())
    d = b.try_acquire(6)
    assert not d.allowed and math.isinf(d.retry_after_s)
    assert b.tokens == 5


def test_consume_can_go_negative_and_refund_is_capped():
    c = Clock()
    b = TokenBucket(10, 1.0, c)
    b.consume(15)
    assert b.tokens == -5
    d = b.try_acquire(1)
    assert not d.allowed and d.retry_after_s == pytest.approx(6.0)
    b.consume(-100)
    assert b.tokens == 10


# ----------------------------------------------------------------------------- RateLimiter
async def test_rpm_limit_and_refill():
    c = Clock()
    rl = RateLimiter(c)
    lim = Limits(rpm=2)
    assert (await rl.check("t", lim, 0)) == (Decision(True, 0.0, 0.0), "")
    assert (await rl.check("t", lim, 0))[0].allowed
    d, which = await rl.check("t", lim, 0)
    assert not d.allowed and which == "rpm"
    assert d.retry_after_s == pytest.approx(30.0)  # 2/min -> one token every 30 s
    c.t = 30
    assert (await rl.check("t", lim, 0))[0].allowed


async def test_tenants_are_isolated():
    rl = RateLimiter(Clock())
    lim = Limits(rpm=1)
    assert (await rl.check("a", lim, 0))[0].allowed
    assert (await rl.check("b", lim, 0))[0].allowed
    assert not (await rl.check("a", lim, 0))[0].allowed


async def test_changed_limits_get_a_fresh_bucket():
    rl = RateLimiter(Clock())
    assert (await rl.check("a", Limits(rpm=1), 0))[0].allowed
    assert (await rl.check("a", Limits(rpm=2), 0))[0].allowed


async def test_unlimited():
    rl = RateLimiter(Clock())
    for _ in range(100):
        assert (await rl.check("t", Limits(), 10_000))[0].allowed


async def test_tpm_limit_uses_estimate():
    c = Clock()
    rl = RateLimiter(c)
    lim = Limits(tpm=100)
    assert (await rl.check("t", lim, 60))[0].allowed
    d, which = await rl.check("t", lim, 60)
    assert not d.allowed and which == "tpm"
    assert d.retry_after_s == pytest.approx(12.0)  # needs 20 tokens at 100/60 per s
    assert (await rl.check("t", lim, 40))[0].allowed


async def test_tpm_zero_estimate_is_not_charged():
    rl = RateLimiter(Clock())
    lim = Limits(tpm=1)
    for _ in range(5):
        assert (await rl.check("t", lim, 0))[0].allowed


async def test_tpm_estimate_larger_than_limit_is_clamped():
    rl = RateLimiter(Clock())
    lim = Limits(tpm=50)
    assert (await rl.check("t", lim, 500))[0].allowed  # a single huge prompt is not unservable
    assert not (await rl.check("t", lim, 1))[0].allowed


async def test_tpm_rejection_refunds_the_request_slot():
    rl = RateLimiter(Clock())
    lim = Limits(rpm=2, tpm=10)
    assert (await rl.check("t", lim, 10))[0].allowed
    d, which = await rl.check("t", lim, 10)
    assert which == "tpm"
    # the rejected request did not use an RPM slot, so one request is still allowed
    assert (await rl.check("t", Limits(rpm=2), 0))[0].allowed
    assert not (await rl.check("t", Limits(rpm=2), 0))[0].allowed


async def test_settle_charges_the_difference():
    c = Clock()
    rl = RateLimiter(c)
    lim = Limits(tpm=100)
    await rl.check("t", lim, 10)
    await rl.settle("t", lim, 10, 100)  # actually used 100: bucket at 0
    d, _ = await rl.check("t", lim, 1)
    assert not d.allowed
    await rl.settle("t", lim, 10, 0)  # over-estimate of 10 is refunded
    assert (await rl.check("t", lim, 10))[0].allowed


async def test_settle_with_clamped_estimate():
    rl = RateLimiter(Clock())
    lim = Limits(tpm=50)
    await rl.check("t", lim, 500)  # charged 50, not 500
    await rl.settle("t", lim, 500, 50)
    b = rl._bucket("t", "tpm", 50)
    assert b.tokens == pytest.approx(0.0)


async def test_settle_negative_estimate_treated_as_zero():
    rl = RateLimiter(Clock())
    lim = Limits(tpm=50)
    await rl.settle("t", lim, -5, 20)
    assert rl._bucket("t", "tpm", 50).tokens == pytest.approx(30.0)


async def test_settle_without_tpm_is_noop():
    rl = RateLimiter(Clock())
    await rl.settle("t", Limits(rpm=5), 10, 1000)
    assert rl._buckets == {}


def test_make_limiter_picks_backend():
    assert isinstance(make_limiter(None), RateLimiter)


# ----------------------------------------------------------------------------- Redis (fakeredis + lupa)
@pytest.fixture
async def redis():
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    r = fakeredis.FakeAsyncRedis()
    yield r
    await r.aclose()


async def test_redis_limiter_matches_memory_semantics(redis):
    c = Clock(1000.0)
    rl = RedisRateLimiter(redis, clock=c)
    assert isinstance(make_limiter(redis), RedisRateLimiter)
    lim = Limits(rpm=2)
    d, which = await rl.check("a", lim, 0)
    assert d.allowed and which == "" and d.retry_after_s == 0.0
    assert (await rl.check("a", lim, 0))[0].allowed
    d, which = await rl.check("a", lim, 0)
    assert not d.allowed and which == "rpm"
    assert d.retry_after_s == pytest.approx(30.0, abs=0.001)
    assert (await rl.check("b", lim, 0))[0].allowed
    c.t += 30
    d = await rl._call("a", "rpm", 2, 1, False)
    assert d.allowed and d.remaining == pytest.approx(0.0, abs=0.001)


async def test_redis_tpm_settle_and_refund(redis):
    c = Clock(50.0)
    rl = RedisRateLimiter(redis, prefix="x", clock=c)
    lim = Limits(rpm=2, tpm=100)
    d, _ = await rl.check("a", lim, 60)
    assert d.allowed
    assert float((await redis.hget("x:a:tpm:100", "t")).decode()) == pytest.approx(40.0)
    d, which = await rl.check("a", lim, 60)
    assert not d.allowed and which == "tpm" and d.retry_after_s == pytest.approx(12.0, abs=0.001)
    # the tpm rejection gave the rpm slot back
    d, _ = await rl.check("a", Limits(rpm=2), 0)
    assert d.allowed
    await rl.settle("a", lim, 60, 100)
    d, _ = await rl.check("a", Limits(tpm=100), 1)
    assert not d.allowed
    await rl.settle("a", Limits(rpm=1), 0, 1000)  # no tpm: no-op
    assert await redis.exists("x:a:rpm:1") == 0
    assert 0 < await redis.ttl("x:a:tpm:100") <= 120


async def test_redis_oversized_and_zero_estimates(redis):
    rl = RedisRateLimiter(redis, clock=Clock(1.0))
    lim = Limits(tpm=10)
    assert (await rl.check("a", lim, 0))[0].allowed
    assert (await rl.check("a", lim, 1000))[0].allowed  # clamped to capacity
    d = await rl._call("a", "tpm", 10, 11, False)
    assert not d.allowed and math.isinf(d.retry_after_s)
    await rl.settle("a", lim, 1000, 10)
    assert float((await redis.hget("llmgate:rl:a:tpm:10", "t")).decode()) == pytest.approx(0.0)


# ----------------------------------------------------------------------------- boundaries
def test_consume_refills_before_charging():
    c = Clock()
    b = TokenBucket(10, 1.0, c)
    b.consume(10)  # empty at t=0
    c.t = 20  # bucket would be full again (capped at 10)
    b.consume(5)
    assert b.tokens == pytest.approx(5.0)


async def test_limits_of_one_are_enforced():
    rl = RateLimiter(Clock())
    assert (await rl.check("t", Limits(rpm=1), 0))[0].allowed
    assert (await rl.check("t", Limits(rpm=1), 0))[1] == "rpm"
    assert (await rl.check("u", Limits(tpm=1), 1))[0].allowed
    assert (await rl.check("u", Limits(tpm=1), 1))[1] == "tpm"


async def test_tpm_refund_with_rpm_of_one():
    rl = RateLimiter(Clock())
    lim = Limits(rpm=1, tpm=5)
    rl._bucket("t", "tpm", 5).consume(5)  # TPM exhausted
    d, which = await rl.check("t", lim, 1)
    assert which == "tpm"
    assert rl._bucket("t", "rpm", 1).tokens == pytest.approx(1.0)  # slot refunded


async def test_settle_with_tpm_of_one():
    rl = RateLimiter(Clock())
    await rl.settle("t", Limits(tpm=1), 0, 3)
    assert rl._bucket("t", "tpm", 1).tokens == pytest.approx(-2.0)


async def test_zero_estimate_creates_no_tpm_bucket():
    rl = RateLimiter(Clock())
    await rl.check("t", Limits(tpm=10), 0)
    assert rl._buckets == {}


async def test_redis_call_decision_fields(redis):
    rl = RedisRateLimiter(redis, clock=Clock(10.0))
    d = await rl._call("a", "tpm", 60, 15, False)  # capacity 60, 1 token/s
    assert d == Decision(True, 0.0, 45.0)
    d = await rl._call("a", "tpm", 60, 50, False)
    assert not d.allowed and d.retry_after_s == pytest.approx(5.0) and d.remaining == pytest.approx(45.0)


async def test_redis_force_can_go_negative(redis):
    rl = RedisRateLimiter(redis, clock=Clock(10.0))
    lim = Limits(tpm=100)
    assert (await rl.check("a", lim, 10))[0].allowed
    await rl.settle("a", lim, 10, 200)  # 190 more than estimated -> -100
    assert (await rl._call("a", "tpm", 100, 0, False)).remaining == pytest.approx(-100.0)
    assert not (await rl.check("a", lim, 1))[0].allowed


async def test_redis_limits_of_one(redis):
    c = Clock(10.0)
    rl = RedisRateLimiter(redis, clock=c)
    assert (await rl.check("r", Limits(rpm=1), 0))[0].allowed
    assert (await rl.check("r", Limits(rpm=1), 0))[1] == "rpm"
    assert (await rl.check("t", Limits(tpm=1), 1))[0].allowed
    assert (await rl.check("t", Limits(tpm=1), 1))[1] == "tpm"
    await rl.settle("s", Limits(tpm=1), 0, 3)
    assert (await rl._call("s", "tpm", 1, 0, False)).remaining == pytest.approx(-2.0)
    # TPM rejection refunds an RPM limit of one
    lim = Limits(rpm=1, tpm=1)
    await rl._call("q", "tpm", 1, 1, True)
    assert (await rl.check("q", lim, 1))[1] == "tpm"
    assert (await rl._call("q", "rpm", 1, 0, False)).remaining == pytest.approx(1.0)


async def test_redis_zero_estimate_and_no_tpm_touch_nothing(redis):
    rl = RedisRateLimiter(redis, prefix="z", clock=Clock(10.0))
    await rl.check("a", Limits(tpm=10), 0)
    await rl.check("a", Limits(), 5)
    await rl.settle("a", Limits(), 5, 50)
    assert await redis.keys("z:*") == []
