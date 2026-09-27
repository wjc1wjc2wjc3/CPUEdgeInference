"""推理后端抽象。

无论底层是 llama.cpp 还是 mock，对外只暴露：load / unload / generate。
这样「换引擎」不影响上层的内存预算、调度与 OpenAI 兼容接口。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_LATIN = re.compile(r"[A-Za-z0-9_']+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")


def count_tokens(text: str) -> int:
    """粗略 token 估算（CJK 按字，拉丁按词）—— 用于 usage 统计，不追求精确。"""
    if not text:
        return 0
    return len(_LATIN.findall(text)) + len(_CJK.findall(text))


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str = "stop"

    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class BaseBackend:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.meta = None
        self._loaded = False

    # ---- 生命周期 ----
    def load(self, meta) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def est_ram_mb(self) -> float:
        return getattr(self.meta, "est_ram_mb", 0.0) if self.meta else 0.0

    @property
    def alias(self) -> str:
        return getattr(self.meta, "alias", "") if self.meta else ""

    # ---- 推理 ----
    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float = 0.7, stop=None) -> GenResult:
        raise NotImplementedError
