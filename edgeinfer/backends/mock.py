"""Mock 后端：**零依赖默认引擎**。

让整条链路（HTTP → 调度 → 内存预算 → OpenAI 兼容响应 → 流式 → embeddings → metrics）
在**没有任何模型文件、没有 GPU、没有第三方库**的情况下也能跑通并被测试。
真实推理请切到 `backend=llama-cpp` 并提供本地 GGUF。
"""
from __future__ import annotations

import time

from .base import BaseBackend, GenResult, apply_stop, count_tokens, hash_embed


class MockBackend(BaseBackend):
    name = "mock"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_calls = 0

    def load(self, meta) -> None:
        self.meta = meta
        self._loaded = True
        self.load_calls += 1

    def unload(self) -> None:
        self._loaded = False
        self.meta = None

    def _compose(self, prompt: str, max_tokens, temperature, **params) -> str:
        text = (
            f"[mock:{self.alias}] 已收到 {count_tokens(prompt)} 个 token 的请求"
            f"（temperature={temperature}）。"
            f"当前为本地 mock 后端，用于验证链路与 OpenAI 兼容协议；"
            f"接入真实模型请把 backend 设为 llama-cpp 并放入 GGUF 文件。\n"
            f"提示结尾：{prompt.strip().replace(chr(10), ' ')[-160:]}"
        )
        mt = max_tokens or self.cfg.default_max_tokens
        if mt and count_tokens(text) > mt:
            text = text[: max(1, mt * 2)]
        return text

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float = 0.7, stop=None, **params) -> GenResult:
        if not self._loaded:
            raise RuntimeError("模型未加载")
        t0 = time.time()
        text = self._compose(prompt, max_tokens, temperature, **params)
        text, hit = apply_stop(text, stop)
        return GenResult(
            text=text,
            prompt_tokens=count_tokens(prompt),
            completion_tokens=count_tokens(text),
            latency_ms=(time.time() - t0) * 1000.0,
            finish_reason="stop" if hit else (
                "length" if (max_tokens or self.cfg.default_max_tokens)
                and count_tokens(text) >= (max_tokens or self.cfg.default_max_tokens) else "stop"),
        )

    def stream_generate(self, prompt: str, max_tokens: int | None = None,
                        temperature: float = 0.7, stop=None, **params):
        """按「词 / 汉字」切块增量产出，模拟真实流式。"""
        if not self._loaded:
            raise RuntimeError("模型未加载")
        text = self._compose(prompt, max_tokens, temperature, **params)
        text, _hit = apply_stop(text, stop)
        buf = ""
        for ch in text:
            buf += ch
            if ch in (" ", "\n") or len(buf) >= 2:  # 中英文都切成小块
                if self.cfg.stream_delay_s:
                    time.sleep(self.cfg.stream_delay_s)
                yield buf
                buf = ""
        if buf:
            yield buf

    def embed(self, texts) -> list:
        """确定性哈希向量（同一文本恒定结果，便于测试与联调）。"""
        if isinstance(texts, str):
            texts = [texts]
        return [hash_embed(t, self.cfg.embed_dim) for t in texts]
