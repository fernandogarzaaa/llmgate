"""Guardrails adapter for injectguard (https://github.com/fernandogarzaaa/injectguard).

injectguard's rules + PII layers are pure standard library. Its classifier
layer needs torch and weights; set ``guard.classifier_path`` to enable it.
When injectguard is not installed the gateway runs without guardrails (and
says so in ``/healthz``) unless ``guard.required`` is true, in which case it
refuses to start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from .config import GuardConfig

log = logging.getLogger("llmgate.guard")


@dataclass
class GuardResult:
    flagged: bool
    score: float = 0.0
    categories: List[str] = field(default_factory=list)
    redacted_text: Optional[str] = None


class GuardBackend(Protocol):
    name: str

    def scan_input(self, text: str) -> GuardResult: ...

    def scan_output(self, text: str) -> GuardResult: ...


class InjectguardBackend:
    def __init__(self, mode: str = "any", classifier_path: Optional[str] = None) -> None:
        from injectguard import Guard

        clf: Any = None
        if classifier_path:
            from injectguard import ClassifierDetector

            clf = ClassifierDetector(classifier_path)
        self._guard = Guard(classifier=clf, mode=mode)
        self.name = "injectguard" + ("+classifier" if clf is not None else "(rules+pii)")

    def scan_input(self, text: str) -> GuardResult:
        v = self._guard.scan_input(text)
        return GuardResult(bool(v.flagged), float(v.score), list(v.categories))

    def scan_output(self, text: str) -> GuardResult:
        v = self._guard.scan_output(text, redact=True)
        return GuardResult(bool(v.flagged), float(v.score), list(v.categories), v.redacted_text)


def load_guard(cfg: GuardConfig) -> Optional[GuardBackend]:
    if not cfg.enabled or (cfg.input == "off" and cfg.output == "off"):
        return None
    try:
        return InjectguardBackend(cfg.mode, cfg.classifier_path)
    except ImportError as exc:
        if cfg.required:
            raise RuntimeError(f"guard.required is true but injectguard is unavailable: {exc}") from None
        log.warning("injectguard is not installed; guardrails are DISABLED (pip install 'llmgate[guard]')")
        return None
