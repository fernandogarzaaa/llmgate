"""Upstream providers."""

from __future__ import annotations

import httpx

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import Provider, ProviderError
from .openai import LocalProvider, OpenAIProvider

__all__ = ["AnthropicProvider", "LocalProvider", "OpenAIProvider", "Provider", "ProviderError", "make_provider"]


def make_provider(cfg: ProviderConfig, client: httpx.AsyncClient) -> Provider:
    cls = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "local": LocalProvider}[cfg.type]
    return cls(cfg, client)
