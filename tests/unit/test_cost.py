import pytest

from llmgate.cost import Price, cost_usd, estimate_tokens, message_text, usage_from_response


def test_price_defaults_and_validation():
    assert Price() == Price(0.0, 0.0)
    Price(0, 0)
    with pytest.raises(ValueError):
        Price(-1, 0)
    with pytest.raises(ValueError):
        Price(0, -1)


def test_cost_per_million_exact():
    assert cost_usd(1_000_000, 0, Price(2.5, 10)) == 2.5
    assert cost_usd(0, 1_000_000, Price(2.5, 10)) == 10.0
    assert cost_usd(1000, 500, Price(2.5, 10)) == pytest.approx(0.0075, rel=1e-12)
    assert cost_usd(0, 0, Price(2.5, 10)) == 0.0
    assert cost_usd(1, 1, Price()) == 0.0


@pytest.mark.parametrize("pt,ct", [(-1, 0), (0, -1)])
def test_cost_rejects_negative_tokens(pt, ct):
    with pytest.raises(ValueError):
        cost_usd(pt, ct, Price(1, 1))


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


def test_message_text_parts():
    msgs = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {}},
                {"type": "text", "text": 5},
                "junk",
            ],
        },
        {"role": "assistant", "content": None},
    ]
    assert message_text(msgs) == "sys\na"
    assert message_text([]) == ""


def test_usage_from_response_prefers_upstream():
    u = usage_from_response({"usage": {"prompt_tokens": 7, "completion_tokens": 3}}, "x" * 100, "y")
    assert u == {"prompt_tokens": 7, "completion_tokens": 3, "estimated": False}


def test_usage_zero_counts_are_valid():
    u = usage_from_response({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}, "x" * 100, "y" * 100)
    assert u == {"prompt_tokens": 0, "completion_tokens": 0, "estimated": False}


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 7},
        {"completion_tokens": 3},
        {"prompt_tokens": "7", "completion_tokens": 3},
        {"prompt_tokens": 7, "completion_tokens": "3"},
        {"prompt_tokens": -1, "completion_tokens": 3},
        {"prompt_tokens": 7, "completion_tokens": -1},
        "nope",
    ],
)
def test_usage_falls_back_to_estimates(usage):
    u = usage_from_response({"usage": usage}, "x" * 8, "y" * 4)
    assert u == {"prompt_tokens": 2, "completion_tokens": 1, "estimated": True}


def test_usage_non_mapping_payload():
    assert usage_from_response([], "abcd", "")["prompt_tokens"] == 1
