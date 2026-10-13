"""costroute / injectguard must never be pulled from PyPI by name.

"injectguard" on PyPI is an unrelated third-party package and "costroute" is unclaimed, so an
extra such as ``llmgate[guard]`` would install someone else's code into the gateway process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[2]


def _all_requirements() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs.extend(extra)
    return reqs


@pytest.mark.parametrize("name", ["injectguard", "costroute"])
def test_no_requirement_resolves_the_name_from_pypi(name: str) -> None:
    for req in _all_requirements():
        if req.split("@")[0].split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() == name:
            assert "@ git+https://github.com/fernandogarzaaa/" in req, req


def test_missing_costroute_error_points_at_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmgate import costroute_adapter

    monkeypatch.setitem(sys.modules, "costroute", None)  # import costroute -> ImportError
    with pytest.raises(ImportError, match="git\\+https://github.com/fernandogarzaaa/costroute"):
        costroute_adapter.load_scorer("lora")


def test_missing_injectguard_warning_points_at_the_repository(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from llmgate import guard
    from llmgate.config import GuardConfig

    monkeypatch.setitem(sys.modules, "injectguard", None)
    with caplog.at_level("WARNING", logger="llmgate.guard"):
        assert guard.load_guard(GuardConfig(enabled=True)) is None
    assert "git+https://github.com/fernandogarzaaa/injectguard" in caplog.text
