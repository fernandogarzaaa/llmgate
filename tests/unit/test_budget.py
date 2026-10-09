from datetime import datetime, timedelta, timezone

import pytest

from llmgate.budget import PERIODS, Budget, BudgetStatus, BudgetTracker, period_start

UTC = timezone.utc


class Clock:
    def __init__(self, dt):
        self.dt = dt

    def __call__(self):
        return self.dt


def test_period_start_variants():
    now = datetime(2026, 10, 9, 15, 30, 12, 999, tzinfo=UTC)
    assert period_start(now, "monthly") == datetime(2026, 10, 1, tzinfo=UTC)
    assert period_start(now, "daily") == datetime(2026, 10, 9, tzinfo=UTC)
    assert period_start(now, "total") == datetime(1970, 1, 1, tzinfo=UTC)
    assert PERIODS == ("daily", "monthly", "total")


def test_period_start_converts_to_utc():
    manila = timezone(timedelta(hours=8))
    # 2026-11-01 05:00 in Manila is still October 31 in UTC
    now = datetime(2026, 11, 1, 5, 0, tzinfo=manila)
    assert period_start(now, "monthly") == datetime(2026, 10, 1, tzinfo=UTC)
    assert period_start(now, "daily") == datetime(2026, 10, 31, tzinfo=UTC)


def test_period_start_errors():
    with pytest.raises(ValueError, match="timezone-aware"):
        period_start(datetime(2026, 1, 1), "daily")
    with pytest.raises(ValueError, match="unknown budget period"):
        period_start(datetime(2026, 1, 1, tzinfo=UTC), "weekly")


def test_budget_validation_and_defaults():
    assert Budget() == Budget(0.0, "monthly")
    assert Budget().unlimited and not Budget(0.01).unlimited
    Budget(0)  # zero is fine (unlimited)
    with pytest.raises(ValueError):
        Budget(-0.01)
    with pytest.raises(ValueError):
        Budget(1, "weekly")


def test_check_blocks_at_and_after_limit():
    bt = BudgetTracker(Clock(datetime(2026, 10, 9, tzinfo=UTC)))
    b = Budget(1.0)
    assert bt.check("t", b) == BudgetStatus(True, 0.0, 1.0, 1.0)
    assert bt.add("t", b, 0.25) == pytest.approx(0.25)
    s = bt.check("t", b)
    assert s.allowed and s.spent_usd == pytest.approx(0.25) and s.remaining_usd == pytest.approx(0.75)
    assert bt.add("t", b, 0.75) == pytest.approx(1.0)
    s = bt.check("t", b)
    assert not s.allowed and s.remaining_usd == 0.0  # spent == limit is exhausted
    bt.add("t", b, 0.5)
    s = bt.check("t", b)
    assert not s.allowed and s.remaining_usd == 0.0 and s.spent_usd == pytest.approx(1.5)


def test_unlimited_budget_reports_infinite_remaining():
    bt = BudgetTracker()
    b = Budget(0)
    bt.add("t", b, 100)
    s = bt.check("t", b)
    assert s.allowed and s.limit_usd == 0.0 and s.remaining_usd == float("inf") and s.spent_usd == 100


def test_add_validation_and_zero_cost():
    bt = BudgetTracker()
    with pytest.raises(ValueError):
        bt.add("t", Budget(1), -0.01)
    assert bt.add("t", Budget(1), 0.0) == 0.0


def test_periods_roll_over():
    clock = Clock(datetime(2026, 10, 31, 23, 0, tzinfo=UTC))
    bt = BudgetTracker(clock)
    monthly, daily, total = Budget(1, "monthly"), Budget(1, "daily"), Budget(1, "total")
    for b in (monthly, daily, total):
        bt.add("t", b, 2.0)
    clock.dt = datetime(2026, 11, 1, 0, 30, tzinfo=UTC)
    assert bt.spent("t", monthly) == 0.0
    assert bt.spent("t", daily) == 0.0
    assert bt.spent("t", total) == 2.0
    assert bt.since(monthly) == datetime(2026, 11, 1, tzinfo=UTC)


def test_tenants_are_separate():
    bt = BudgetTracker()
    bt.add("a", Budget(1), 5)
    assert bt.spent("b", Budget(1)) == 0.0


def test_seed_replaces_and_includes_pending():
    bt = BudgetTracker()
    b = Budget(10)
    bt.add("t", b, 3)
    bt.seed("t", b, 1.5)
    assert bt.spent("t", b) == 1.5
    bt.seed("t", b, 1.5, pending=0.5)
    assert bt.spent("t", b) == 2.0
    bt.seed("t", b, 0.0)
    assert bt.spent("t", b) == 0.0
    with pytest.raises(ValueError):
        bt.seed("t", b, -1)
    with pytest.raises(ValueError):
        bt.seed("t", b, 1, pending=-1)


def test_prune_drops_old_periods_only():
    clock = Clock(datetime(2026, 9, 15, tzinfo=UTC))
    bt = BudgetTracker(clock)
    bt.add("t", Budget(1, "monthly"), 1)
    bt.add("t", Budget(1, "daily"), 1)
    bt.add("t", Budget(1, "total"), 1)
    clock.dt = datetime(2026, 10, 2, tzinfo=UTC)
    bt.add("t", Budget(1, "daily"), 1)
    assert bt.prune() == 2  # September monthly + Sept 15 daily; "total" and today kept
    assert bt.spent("t", Budget(1, "total")) == 1
    assert bt.spent("t", Budget(1, "daily")) == 1
    assert bt.prune() == 0
    assert bt.prune(keep_after=datetime(2026, 10, 3, tzinfo=UTC)) == 1
    assert bt.spent("t", Budget(1, "daily")) == 0
