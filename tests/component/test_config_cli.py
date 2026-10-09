import json
import os

import pytest

from llmgate.cli import main, parse_since
from llmgate.config import ConfigError, interpolate, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK_CFG = os.path.join(ROOT, "examples", "config.mock.yaml")


def test_interpolation():
    env = {"A": "1", "EMPTY": ""}
    assert interpolate({"x": ["${A}", "${B:-d}", "${EMPTY:-e}"]}, env) == {"x": ["1", "d", "e"]}
    with pytest.raises(ConfigError):
        interpolate("${MISSING}", env)


def test_example_configs_validate():
    cfg = load_config(MOCK_CFG, env={})
    assert cfg.routing.costroute.enabled and len(cfg.models) == 4
    prod = load_config(
        os.path.join(ROOT, "examples", "config.production.yaml"),
        env={
            "LLMGATE_ADMIN_KEY": "a",
            "LLMGATE_DATABASE_URL": "sqlite://",
            "LLMGATE_REDIS_URL": "redis://r",
            "OPENAI_API_KEY": "k",
            "ANTHROPIC_API_KEY": "k",
        },
    )
    assert prod.cache.exact.backend == "redis"


def test_env_overrides():
    cfg = load_config(MOCK_CFG, env={"LLMGATE_DATABASE_URL": "sqlite:///x.db", "LLMGATE_PORT": "9999"})
    assert cfg.database_url == "sqlite:///x.db" and cfg.server.port == 9999


def test_config_reference_errors():
    base = {"providers": [{"name": "p", "base_url": "http://x"}], "models": [{"name": "m", "provider": "q"}]}
    with pytest.raises(ConfigError, match="unknown provider"):
        load_config(data=base, env={})
    base["models"] = [{"name": "m", "provider": "p", "fallbacks": ["zz"]}]
    with pytest.raises(ConfigError, match="unknown fallback"):
        load_config(data=base, env={})


def test_parse_since():
    assert parse_since("2026-01-02").year == 2026
    d = parse_since("7d")
    assert d.tzinfo is not None


def test_cli_tenants_keys_usage(tmp_path, capsys):
    db = f"sqlite:///{tmp_path}/cli.db"
    assert main(["tenants", "create", "acme", "--rpm", "10", "--budget", "5", "--database-url", db]) == 0
    assert main(["keys", "create", "--tenant", "acme", "--json", "--database-url", db]) == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    key = json.loads(out)
    assert key["key"].startswith("lg-")
    assert main(["keys", "list", "--json", "--database-url", db]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["id"] == key["id"] and not listed[0]["revoked"]
    assert main(["keys", "revoke", key["id"], "--database-url", db]) == 0
    assert main(["keys", "create", "--tenant", "ghost", "--database-url", db]) == 2
    assert main(["usage", "--json", "--database-url", db]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == []
    assert main(["tenants", "update", "acme", "--models", "a,b", "--database-url", db]) == 0
    capsys.readouterr()
    assert main(["tenants", "list", "--json", "--database-url", db]) == 0
    assert json.loads(capsys.readouterr().out)[0]["allowed_models"] == ["a", "b"]


def test_cli_check_config(capsys):
    assert main(["check-config", "--config", MOCK_CFG]) == 0
    assert "ok:" in capsys.readouterr().out
