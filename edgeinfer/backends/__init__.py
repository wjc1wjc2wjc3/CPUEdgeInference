"""后端工厂：按配置选择引擎，默认 mock（零依赖可跑）。"""
from __future__ import annotations

from .base import BaseBackend, GenResult, count_tokens
from .mock import MockBackend

__all__ = ["BaseBackend", "GenResult", "MockBackend", "count_tokens", "get_backend"]


def get_backend(cfg) -> BaseBackend:
    if cfg.backend == "mock":
        return MockBackend(cfg)
    if cfg.backend == "llama-cpp":
        from .llama_cpp import LlamaCppBackend
        return LlamaCppBackend(cfg)
    raise ValueError(f"未知后端：{cfg.backend}（可选：mock / llama-cpp）")
