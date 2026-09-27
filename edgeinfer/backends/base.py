"""推理后端抽象。

无论底层是 llama.cpp 还是 mock，对外只暴露：load / unload / generate / embed。
这样「换引擎」不影响上层的内存预算、调度与 OpenAI 兼容接口。
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

_LATIN = re.compile(r"[A-Za-z0-9_']+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_WORD = re.compile(r"\w+", re.UNICODE)


def count_tokens(text: str) -> int:
    """粗略 token 估算（CJK 按字，拉丁按词）—— 用于 usage 统计，不追求精确。"""
    if not text:
        return 0
    return len(_LATIN.findall(text)) + len(_CJK.findall(text))


def hash_embed(text: str, dim: int = 256):
    """确定性哈希向量（零依赖，供 mock 后端的 /v1/embeddings 使用）。

    用 blake2b 而非内置 hash()，保证跨进程结果一致。
    """
    vec = [0.0] * dim
    toks = _WORD.findall(text.lower())
    for tok in toks:
        d = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        i = int.from_bytes(d[:8], "big") % dim
        s = 1 if (int.from_bytes(d[8:], "big") & 1) else -1
        vec[i] += s
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


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
                 temperature: float = 0.7, stop=None, **params) -> GenResult:
        raise NotImplementedError

    def stream_generate(self, prompt: str, max_tokens: int | None = None,
                        temperature: float = 0.7, stop=None, **params):
        """流式生成。默认实现：整块产出（保证任何后端都能被流式接口调用）。

        子类可覆盖为真正的增量生成。
        """
        r = self.generate(prompt, max_tokens, temperature, stop, **params)
        yield r.text

    # ---- 向量 ----
    def embed(self, texts) -> list:
        """文本向量化；不支持的后端会抛 NotImplementedError。"""
        raise NotImplementedError(f"{self.name} 后端不支持 embeddings")


def apply_stop(text: str, stop) -> tuple:
    """按 stop 序列截断，返回 (截断后文本, 是否命中)。"""
    if not stop:
        return text, None
    for s in stop:
        idx = text.find(s)
        if idx >= 0:
            return text[:idx], s
    return text, None
