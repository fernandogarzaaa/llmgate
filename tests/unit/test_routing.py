import pytest

from llmgate.routing import (
    CostRouteConfig,
    RouteContext,
    RouteDecision,
    Router,
    RoutingError,
    Rule,
    parse_threshold,
    rules_from_config,
)


class FixedScorer:
    name = "fixed"

    def __init__(self, v):
        self.v = v
        self.calls = 0

    def score(self, prompts):
        self.calls += 1
        return [self.v for _ in prompts]


MODELS = {"strong": ["backup"], "weak": [], "backup": []}
CR = CostRouteConfig("auto", "strong", "weak", 0.5)


def ctx(model="weak", prompt="hi", tenant="", headers=None):
    return RouteContext(model, prompt, tenant, headers or {})


# ------------------------------------------------------------------------------- Rule
def test_rule_unset_conditions_match_everything():
    assert Rule("r", "weak").matches(ctx("anything", ""))


def test_rule_model_condition():
    r = Rule("r", "weak", model="strong")
    assert r.matches(ctx("strong")) and not r.matches(ctx("weak"))
    assert Rule("r", "weak", model="*").matches(ctx("x"))


def test_rule_tenant_condition():
    r = Rule("r", "weak", tenant="acme")
    assert r.matches(ctx(tenant="acme")) and not r.matches(ctx(tenant="other"))


def test_rule_length_bounds_are_inclusive():
    r = Rule("r", "weak", min_chars=3, max_chars=5)
    assert not r.matches(ctx(prompt="ab"))
    assert r.matches(ctx(prompt="abc"))
    assert r.matches(ctx(prompt="abcde"))
    assert not r.matches(ctx(prompt="abcdef"))
    assert Rule("eq", "weak", min_chars=4, max_chars=4).matches(ctx(prompt="abcd"))


def test_rule_min_greater_than_max_rejected():
    with pytest.raises(ValueError):
        Rule("r", "weak", min_chars=5, max_chars=4)
    Rule("r", "weak", min_chars=5)
    Rule("r", "weak", max_chars=0)


def test_rule_regex_is_case_insensitive():
    r = Rule("r", "weak", contains=r"\bsql\b")
    assert r.matches(ctx(prompt="Write SQL please")) and not r.matches(ctx(prompt="sequel"))


def test_rule_headers_case_insensitive_names_exact_values():
    r = Rule("r", "weak", header={"X-Team": "ml"})
    assert r.matches(ctx(headers={"x-team": "ml"}))
    assert not r.matches(ctx(headers={"x-team": "ML"}))
    assert not r.matches(ctx(headers={}))
    r2 = Rule("r", "weak", header={"a": "1", "b": "2"})
    assert not r2.matches(ctx(headers={"a": "1"})) and r2.matches(ctx(headers={"A": "1", "b": "2"}))


def test_rule_non_match_is_false_not_falsy():
    c = ctx("weak", "abc", "acme", {"x": "1"})
    for r in (
        Rule("r", "w", model="strong"),
        Rule("r", "w", tenant="other"),
        Rule("r", "w", min_chars=9),
        Rule("r", "w", max_chars=1),
        Rule("r", "w", contains="zzz"),
        Rule("r", "w", header={"x": "2"}),
    ):
        assert r.matches(c) is False
    assert Rule("r", "w").matches(c) is True


def test_rule_conditions_are_anded():
    r = Rule("r", "weak", model="strong", tenant="acme", contains="x")
    assert r.matches(ctx("strong", "x", "acme"))
    assert not r.matches(ctx("strong", "x", "other"))
    assert not r.matches(ctx("weak", "x", "acme"))
    assert not r.matches(ctx("strong", "y", "acme"))


# ------------------------------------------------------------------------ config bits
def test_costroute_config_threshold_bounds():
    assert CostRouteConfig("a", "s", "w").threshold == 0.5
    CostRouteConfig("a", "s", "w", 0.0)
    CostRouteConfig("a", "s", "w", 1.0)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            CostRouteConfig("a", "s", "w", bad)


def test_parse_threshold():
    assert parse_threshold(None, 0.5) == 0.5
    assert parse_threshold("", 0.5) == 0.5
    assert parse_threshold("0.7", 0.5) == 0.7
    assert parse_threshold("0", 0.5) == 0.0
    assert parse_threshold("1", 0.5) == 1.0
    for bad in ("x", "-0.1", "1.5"):
        with pytest.raises(ValueError):
            parse_threshold(bad, 0.5)


def test_candidates_dedupe_preserves_order():
    assert RouteDecision("a", ["b", "a", "c", "b"], "direct").candidates() == ["a", "b", "c"]
    assert RouteDecision("a", [], "direct").candidates() == ["a"]


def test_rules_from_config():
    rules = rules_from_config(
        [{"name": "n", "route_to": "weak", "when": {"min_chars": 3}, "fallbacks": ["x"]}, {"route_to": "strong"}]
    )
    assert rules[0] == Rule("n", "weak", min_chars=3, fallbacks=["x"])
    assert rules[1].name == "rule-1" and rules[1].route_to == "strong"
    with pytest.raises(ValueError, match="unknown condition"):
        rules_from_config([{"route_to": "weak", "when": {"color": "red"}}])
    with pytest.raises(ValueError, match="missing route_to"):
        rules_from_config([{"when": {}}])


# ----------------------------------------------------------------------------- Router
def test_router_validates_references():
    with pytest.raises(ValueError, match="unknown model"):
        Router(MODELS, [Rule("r", "nope")])
    Router(MODELS, [Rule("r", "auto")], CR, FixedScorer(0))  # alias is a valid rule target
    with pytest.raises(ValueError, match="unknown model"):
        Router(MODELS, [Rule("r", "auto")])  # ... but only with a cost router
    with pytest.raises(ValueError, match="costroute model"):
        Router(MODELS, costroute=CostRouteConfig("auto", "strong", "missing"))
    with pytest.raises(ValueError, match="costroute model"):
        Router(MODELS, costroute=CostRouteConfig("auto", "missing", "weak"))
    with pytest.raises(ValueError, match="default_model"):
        Router(MODELS, default_model="nope")
    Router(MODELS, default_model="weak")
    Router(MODELS, costroute=CR, scorer=FixedScorer(0), default_model="auto")
    with pytest.raises(ValueError, match="default_model"):
        Router(MODELS, default_model="auto")


def test_public_models():
    assert Router(MODELS).public_models() == ["strong", "weak", "backup"]
    assert Router(MODELS, costroute=CR, scorer=FixedScorer(0)).public_models() == ["strong", "weak", "backup", "auto"]


def test_direct_route_with_fallbacks():
    d = Router(MODELS).route(ctx("strong"))
    assert (d.model, d.fallbacks, d.strategy, d.reason, d.score) == ("strong", ["backup"], "direct", "requested", None)


def test_default_model_used_when_none_requested():
    assert Router(MODELS, default_model="weak").route(ctx("")).model == "weak"
    with pytest.raises(RoutingError):
        Router(MODELS).route(ctx(""))


def test_unknown_model():
    with pytest.raises(RoutingError):
        Router(MODELS).route(ctx("nope"))


def test_costroute_threshold_boundary_and_fallbacks():
    hi = Router(MODELS, costroute=CR, scorer=FixedScorer(0.5)).route(ctx("auto"))
    assert hi.model == "strong" and hi.strategy == "costroute" and hi.score == 0.5
    assert hi.fallbacks == ["backup", "weak"]  # own fallbacks, then the other tier
    assert hi.reason == "score=0.5000 threshold=0.50"
    lo = Router(MODELS, costroute=CR, scorer=FixedScorer(0.49)).route(ctx("auto"))
    assert lo.model == "weak" and lo.fallbacks == ["strong"]


def test_costroute_per_request_threshold_and_precomputed_score():
    s = FixedScorer(0.3)
    r = Router(MODELS, costroute=CR, scorer=s)
    assert r.route(ctx("auto"), threshold=0.2).model == "strong"
    assert r.route(ctx("auto"), score=0.9).model == "strong" and s.calls == 1


def test_costroute_without_scorer_errors():
    with pytest.raises(RoutingError):
        Router(MODELS, costroute=CR).route(ctx("auto"))


def test_needs_score():
    r = Router(MODELS, [Rule("long", "auto", min_chars=10)], CR, FixedScorer(0))
    assert r.needs_score(ctx("auto"))
    assert r.needs_score(ctx("weak", "x" * 10))  # a rule sends it to the alias
    assert not r.needs_score(ctx("weak"))
    assert not Router(MODELS).needs_score(ctx("auto"))


def test_first_matching_rule_wins():
    rules = [Rule("a", "strong", contains="x"), Rule("b", "backup", contains="x")]
    d = Router(MODELS, rules).route(ctx("weak", "x"))
    assert (d.model, d.strategy, d.reason, d.fallbacks) == ("strong", "rule", "a", ["backup"])
    assert Router(MODELS, rules).route(ctx("weak", "y")).strategy == "direct"


def test_rule_fallback_override():
    d = Router(MODELS, [Rule("r", "strong", fallbacks=["weak"])]).route(ctx("weak"))
    assert d.fallbacks == ["weak"]
    d = Router(MODELS, [Rule("r", "strong", fallbacks=[])]).route(ctx("weak"))
    assert d.fallbacks == []


def test_rule_to_cost_router():
    r = Router(MODELS, [Rule("long", "auto", min_chars=3, fallbacks=["backup"])], CR, FixedScorer(0.9))
    d = r.route(ctx("weak", "long prompt"))
    assert d.model == "strong" and d.strategy == "costroute" and d.fallbacks == ["backup"]
    assert d.reason == "long; score=0.9000 threshold=0.50"
    r2 = Router(MODELS, [Rule("long", "auto", min_chars=3)], CR, FixedScorer(0.1))
    assert r2.route(ctx("weak", "long prompt")).fallbacks == ["strong"]
