"""Request routing: rules first, then cost-aware scoring, then the requested model.

Order of precedence for a request asking for ``model``:

1. The first matching **rule** (``rules:`` in the config) picks the target.
   A rule's target may itself be the cost-router alias.
2. If the target is the **cost-router alias** (default ``auto``), the scorer
   (costroute) estimates P(needs strong model); ``score >= threshold`` goes to
   the strong model, otherwise to the weak one.
3. Otherwise the target must be a configured model name.

Every decision carries the ordered list of fallbacks to try if the chosen
model's provider fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence


class Scorer(Protocol):
    name: str

    def score(self, prompts: Sequence[str]) -> List[float]: ...


@dataclass(frozen=True)
class RouteContext:
    requested_model: str
    prompt: str
    tenant: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """Match conditions are ANDed; an unset condition always matches."""

    name: str
    route_to: str
    model: Optional[str] = None  # requested model must equal this ("*" or None = any)
    contains: Optional[str] = None  # case-insensitive regex searched in the prompt
    min_chars: Optional[int] = None  # prompt length >= min_chars
    max_chars: Optional[int] = None  # prompt length <= max_chars
    tenant: Optional[str] = None
    header: Optional[Dict[str, str]] = None  # every header must equal the given value
    fallbacks: Optional[List[str]] = None  # overrides the target model's own fallbacks
    _rx: Optional[re.Pattern[str]] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.min_chars is not None and self.max_chars is not None and self.min_chars > self.max_chars:
            raise ValueError(f"rule {self.name!r}: min_chars > max_chars")
        if self.contains is not None:
            object.__setattr__(self, "_rx", re.compile(self.contains, re.IGNORECASE))

    def matches(self, ctx: RouteContext) -> bool:
        if self.model not in (None, "*") and ctx.requested_model != self.model:
            return False
        if self.tenant is not None and ctx.tenant != self.tenant:
            return False
        n = len(ctx.prompt)
        if self.min_chars is not None and n < self.min_chars:
            return False
        if self.max_chars is not None and n > self.max_chars:
            return False
        if self._rx is not None and not self._rx.search(ctx.prompt):
            return False
        if self.header:
            lower = {k.lower(): v for k, v in ctx.headers.items()}
            for k, v in self.header.items():
                if lower.get(k.lower()) != v:
                    return False
        return True


@dataclass(frozen=True)
class CostRouteConfig:
    alias: str
    strong: str
    weak: str
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


@dataclass
class RouteDecision:
    model: str
    fallbacks: List[str]
    strategy: str  # "rule" | "costroute" | "direct"
    reason: str = ""
    score: Optional[float] = None

    def candidates(self) -> List[str]:
        """The chosen model followed by its fallbacks, without duplicates."""
        out: List[str] = []
        for m in [self.model, *self.fallbacks]:
            if m not in out:
                out.append(m)
        return out


class RoutingError(LookupError):
    pass


def parse_threshold(raw: Optional[str], default: float) -> float:
    """Per-request threshold override (header ``x-llmgate-threshold``)."""
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"threshold must be a number, got {raw!r}") from None
    if not 0.0 <= v <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return v


class Router:
    def __init__(
        self,
        models: Mapping[str, Sequence[str]],
        rules: Sequence[Rule] = (),
        costroute: Optional[CostRouteConfig] = None,
        scorer: Optional[Scorer] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """``models`` maps each configured model name to its default fallback list."""
        self.models = {k: list(v) for k, v in models.items()}
        self.rules = list(rules)
        self.costroute = costroute
        self.scorer = scorer
        self.default_model = default_model
        known = set(self.models)
        alias = costroute.alias if costroute else None
        for r in self.rules:
            if r.route_to not in known and r.route_to != alias:
                raise ValueError(f"rule {r.name!r} routes to unknown model {r.route_to!r}")
        if costroute is not None:
            for m in (costroute.strong, costroute.weak):
                if m not in known:
                    raise ValueError(f"costroute model {m!r} is not configured")
        if default_model is not None and default_model not in known and default_model != alias:
            raise ValueError(f"default_model {default_model!r} is not configured")

    def public_models(self) -> List[str]:
        names = list(self.models)
        if self.costroute is not None:
            names.append(self.costroute.alias)
        return names

    def needs_score(self, ctx: RouteContext) -> bool:
        """Whether :meth:`route` will call the scorer (so callers can run it off the event loop)."""
        return self.costroute is not None and self._target(ctx)[0] == self.costroute.alias

    def _target(self, ctx: RouteContext) -> tuple[str, Optional[Rule]]:
        for rule in self.rules:
            if rule.matches(ctx):
                return rule.route_to, rule
        target = ctx.requested_model or (self.default_model or "")
        return target, None

    def route(
        self, ctx: RouteContext, threshold: Optional[float] = None, score: Optional[float] = None
    ) -> RouteDecision:
        """Pick the model for ``ctx``. ``score`` may be pre-computed by the caller."""
        target, rule = self._target(ctx)
        strategy, reason = ("rule", rule.name) if rule is not None else ("direct", "requested")
        if self.costroute is not None and target == self.costroute.alias:
            cr = self.costroute
            if score is None:
                if self.scorer is None:
                    raise RoutingError("cost router alias requested but no scorer is configured")
                score = float(self.scorer.score([ctx.prompt])[0])
            thr = cr.threshold if threshold is None else threshold
            model = cr.strong if score >= thr else cr.weak
            other = cr.weak if model == cr.strong else cr.strong
            fallbacks = (
                rule.fallbacks if (rule is not None and rule.fallbacks is not None) else [*self.models[model], other]
            )
            why = f"score={score:.4f} threshold={thr:.2f}"
            return RouteDecision(
                model, list(fallbacks), "costroute", f"{reason}; {why}" if rule is not None else why, score
            )
        if target not in self.models:
            raise RoutingError(f"model {target!r} is not available on this gateway")
        fallbacks = rule.fallbacks if (rule is not None and rule.fallbacks is not None) else self.models[target]
        return RouteDecision(target, list(fallbacks), strategy, reason)


def rules_from_config(items: Sequence[Mapping[str, Any]]) -> List[Rule]:
    out = []
    for i, it in enumerate(items):
        when = dict(it.get("when") or {})
        unknown = set(when) - {"model", "contains", "min_chars", "max_chars", "tenant", "header"}
        if unknown:
            raise ValueError(f"rule #{i}: unknown condition(s) {sorted(unknown)}")
        if "route_to" not in it:
            raise ValueError(f"rule #{i}: missing route_to")
        out.append(
            Rule(
                name=str(it.get("name", f"rule-{i}")),
                route_to=str(it["route_to"]),
                fallbacks=it.get("fallbacks"),
                **when,
            )
        )
    return out
