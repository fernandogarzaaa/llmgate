"""Gateway configuration: a YAML file with ``${VAR}`` / ``${VAR:-default}`` interpolation.

The file is chosen by ``--config`` or ``LLMGATE_CONFIG``. A few top-level
settings can also be overridden by environment variables (see ``ENV_OVERRIDES``).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


def interpolate(value: Any, env: Mapping[str, str]) -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in every string of a nested structure."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set (use ${{{name}:-default}})")

        return _VAR.sub(sub, value)
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BreakerConfig(_Model):
    failure_threshold: int = Field(5, ge=1)
    recovery_s: float = Field(30.0, ge=0)
    half_open_max: int = Field(1, ge=1)


class ProviderConfig(_Model):
    name: str
    type: Literal["openai", "anthropic", "local"] = "openai"
    base_url: str
    api_key: str = ""
    timeout_s: float = Field(60.0, gt=0)
    connect_timeout_s: float = Field(5.0, gt=0)
    max_retries: int = Field(2, ge=0)
    retry_base_s: float = Field(0.2, ge=0)
    retry_max_s: float = Field(5.0, ge=0)
    breaker: BreakerConfig = BreakerConfig()
    headers: Dict[str, str] = {}
    anthropic_version: str = "2023-06-01"


class ModelConfig(_Model):
    name: str
    provider: str
    upstream_model: Optional[str] = None
    input_per_1m: float = Field(0.0, ge=0)
    output_per_1m: float = Field(0.0, ge=0)
    fallbacks: List[str] = []


class CostRouteSection(_Model):
    enabled: bool = False
    alias: str = "auto"
    router: str = "heuristic"
    threshold: float = Field(0.5, ge=0, le=1)
    strong: str = ""
    weak: str = ""


class RoutingConfig(_Model):
    default_model: Optional[str] = None
    rules: List[Dict[str, Any]] = []
    costroute: CostRouteSection = CostRouteSection()


class GuardConfig(_Model):
    enabled: bool = True
    input: Literal["block", "flag", "off"] = "block"
    output: Literal["redact", "block", "flag", "off"] = "redact"
    mode: Literal["any", "cascade"] = "any"
    classifier_path: Optional[str] = None
    stream_output: Literal["post_scan", "buffer"] = "post_scan"
    required: bool = False


class ExactCacheConfig(_Model):
    enabled: bool = True
    backend: Literal["memory", "redis"] = "memory"
    ttl_s: float = Field(3600, ge=0)
    max_entries: int = Field(10_000, ge=1)


class SemanticCacheConfig(_Model):
    enabled: bool = False
    backend: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "llmgate-semantic.db"
    threshold: float = Field(0.95, gt=0, le=1)
    ttl_s: float = Field(3600, ge=0)
    max_entries_per_namespace: int = Field(5000, ge=1)
    embedder: Literal["hashing", "fastembed", "openai"] = "hashing"
    embedder_model: Optional[str] = None
    embedder_base_url: Optional[str] = None
    embedder_api_key: str = ""


class CacheConfig(_Model):
    scope: Literal["tenant", "global"] = "tenant"
    only_deterministic: bool = False
    exact: ExactCacheConfig = ExactCacheConfig()
    semantic: SemanticCacheConfig = SemanticCacheConfig()


class BootstrapKey(_Model):
    tenant: str
    api_key: str = Field(min_length=8)
    rpm: int = Field(0, ge=0)
    tpm: int = Field(0, ge=0)
    budget_usd: float = Field(0.0, ge=0)
    budget_period: Literal["daily", "monthly", "total"] = "monthly"


class TenancyConfig(_Model):
    require_auth: bool = True
    default_rpm: int = Field(0, ge=0)
    default_tpm: int = Field(0, ge=0)
    key_cache_s: float = Field(30.0, ge=0)
    budget_refresh_s: float = Field(30.0, gt=0)
    bootstrap: List[BootstrapKey] = []


class TracingConfig(_Model):
    enabled: bool = False
    service_name: str = "llmgate"
    exporter: Literal["otlp", "console", "none"] = "otlp"
    endpoint: Optional[str] = None


class ServerConfig(_Model):
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)
    admin_key: str = ""
    max_body_bytes: int = Field(4_000_000, ge=1024)
    log_level: str = "info"


class GatewayConfig(_Model):
    server: ServerConfig = ServerConfig()
    database_url: str = "sqlite:///llmgate.db"
    redis_url: Optional[str] = None
    providers: List[ProviderConfig]
    models: List[ModelConfig]
    routing: RoutingConfig = RoutingConfig()
    guard: GuardConfig = GuardConfig()
    cache: CacheConfig = CacheConfig()
    tenancy: TenancyConfig = TenancyConfig()
    tracing: TracingConfig = TracingConfig()

    @field_validator("redis_url")
    @classmethod
    def _empty_redis(cls, v: Optional[str]) -> Optional[str]:
        return v or None

    @model_validator(mode="after")
    def _check_refs(self) -> GatewayConfig:
        pnames = [p.name for p in self.providers]
        if len(set(pnames)) != len(pnames):
            raise ValueError("provider names must be unique")
        mnames = [m.name for m in self.models]
        if len(set(mnames)) != len(mnames):
            raise ValueError("model names must be unique")
        for m in self.models:
            if m.provider not in pnames:
                raise ValueError(f"model {m.name!r} uses unknown provider {m.provider!r}")
            for f in m.fallbacks:
                if f not in mnames:
                    raise ValueError(f"model {m.name!r} has unknown fallback {f!r}")
        cr = self.routing.costroute
        if cr.enabled and (cr.strong not in mnames or cr.weak not in mnames):
            raise ValueError("routing.costroute.strong and .weak must be configured model names")
        if self.cache.exact.enabled and self.cache.exact.backend == "redis" and not self.redis_url:
            raise ValueError("cache.exact.backend=redis needs redis_url")
        return self

    def model(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    def provider(self, name: str) -> ProviderConfig:
        for p in self.providers:
            if p.name == name:
                return p
        raise KeyError(name)


ENV_OVERRIDES = {
    "LLMGATE_DATABASE_URL": ("database_url",),
    "LLMGATE_REDIS_URL": ("redis_url",),
    "LLMGATE_ADMIN_KEY": ("server", "admin_key"),
    "LLMGATE_PORT": ("server", "port"),
}


def load_config(
    path: Optional[str] = None, env: Optional[Mapping[str, str]] = None, data: Optional[Dict[str, Any]] = None
) -> GatewayConfig:
    env = dict(os.environ if env is None else env)
    if data is None:
        path = path or env.get("LLMGATE_CONFIG")
        if not path:
            raise ConfigError("no config file: pass --config or set LLMGATE_CONFIG")
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}") from None
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    data = interpolate(data, env)
    for var, dotted in ENV_OVERRIDES.items():
        if env.get(var):
            node = data
            for part in dotted[:-1]:
                node = node.setdefault(part, {})
            node[dotted[-1]] = env[var]
    try:
        return GatewayConfig.model_validate(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
