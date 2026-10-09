from datetime import datetime, timezone

import pytest

from llmgate.budget import Budget, BudgetTracker, period_start


def test_period_start_monthly():
    now = datetime(2026, 10, 9, 15, 30, tzinfo=timezone.utc)
    assert period_start(now, "monthly") == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert period_start(now, "daily") == datetime(2026, 10, 9, tzinfo=timezone.utc)


def test_budget_blocks_after_limit():
    bt = BudgetTracker(lambda: datetime(2026, 10, 9, tzinfo=timezone.utc))
    b = Budget(1.0)
    assert bt.check("t", b).allowed
    bt.add("t", b, 0.6)
    assert bt.check("t", b).allowed
    bt.add("t", b, 0.6)
    assert not bt.check("t", b).allowed


def test_unlimited_budget():
    bt = BudgetTracker()
    b = Budget(0)
    bt.add("t", b, 100)
    assert bt.check("t", b).allowed


def test_invalid_period():
    with pytest.raises(ValueError):
        Budget(1, "weekly")
