import pytest

from llmgate.cost import Price, cost_usd, estimate_tokens, message_text, usage_from_response


def test_cost_per_million():
    assert cost_usd(1_000_000, 0, Price(2.5, 10)) == pytest.approx(2.5)
    assert cost_usd(1000, 500, Price(2.5, 10)) == pytest.approx(0.0025 + 0.005)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_message_text_parts():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}]}]
    assert message_text(msgs) == "sys\na"


def test_usage_from_response_prefers_upstream():
    u = usage_from_response({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, "x" * 100, "y")
    assert u == {"prompt_tokens": 7, "completion_tokens": 3, "estimated": False}


def test_usage_from_response_estimates():
    u = usage_from_response({}, "x" * 8, "y" * 4)
    assert u["estimated"] is True
    assert u["prompt_tokens"] == 2
