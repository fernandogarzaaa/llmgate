import pytest

from llmgate.routing import CostRouteConfig, RouteContext, Router, RoutingError, Rule, rules_from_config


class FixedScorer:
    name = "fixed"

    def __init__(self, v):
        self.v = v

    def score(self, prompts):
        return [self.v for _ in prompts]


MODELS = {"strong": ["backup"], "weak": [], "backup": []}


def test_direct_route_with_fallbacks():
    r = Router(MODELS)
    d = r.route(RouteContext("strong", "hi"))
    assert d.model == "strong" and d.fallbacks == ["backup"] and d.strategy == "direct"


def test_unknown_model():
    with pytest.raises(RoutingError):
        Router(MODELS).route(RouteContext("nope", "hi"))


def test_costroute_threshold():
    cr = CostRouteConfig("auto", "strong", "weak", 0.5)
    assert Router(MODELS, costroute=cr, scorer=FixedScorer(0.7)).route(RouteContext("auto", "x")).model == "strong"
    assert Router(MODELS, costroute=cr, scorer=FixedScorer(0.2)).route(RouteContext("auto", "x")).model == "weak"


def test_rule_matches_length():
    rules = [Rule("long", "strong", min_chars=10)]
    r = Router(MODELS, rules)
    assert r.route(RouteContext("weak", "x" * 20)).model == "strong"
    assert r.route(RouteContext("weak", "x")).model == "weak"


def test_rules_from_config_rejects_unknown_condition():
    with pytest.raises(ValueError):
        rules_from_config([{"route_to": "weak", "when": {"color": "red"}}])
