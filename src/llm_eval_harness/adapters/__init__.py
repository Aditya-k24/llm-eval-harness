"""Adapter package — exports factory and all adapters."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import yaml

from .base import ModelAdapter, ModelResult, TimingInfo
from .anthropic_adapter import AnthropicAdapter
from .openai_adapter import OpenAIAdapter
from .gemini_adapter import GeminiAdapter

__all__ = [
    "ModelAdapter",
    "ModelResult",
    "TimingInfo",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "load_adapters",
]

_PROVIDER_MAP = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
}


def load_adapters(config_path: str = "configs/models.yaml") -> list:
    """Read models.yaml and instantiate one adapter per model entry."""
    data = yaml.safe_load(pathlib.Path(config_path).read_text())
    adapters = []
    for model_cfg in data.get("models", []):
        provider = model_cfg.get("provider")
        cls = _PROVIDER_MAP.get(provider)
        if cls is None:
            raise ValueError(f"Unknown provider '{provider}' in {config_path}")
        # Build kwargs — remove keys not accepted by the adapter constructors
        kwargs = {k: v for k, v in model_cfg.items() if k not in ("provider", "display_name")}
        kwargs["model_id"] = kwargs.pop("id")
        adapters.append(cls(**kwargs))
    return adapters
