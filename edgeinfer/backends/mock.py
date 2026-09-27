"""Mock 后端：**零依赖默认引擎**。

目的：让整条链路（HTTP → 调度 → 内存预算 → OpenAI 兼容响应 → metrics）
在**没有任何模型文件、没有 GPU、没有第三方库**的情况下也能跑通并被测试。
真实推理请切到 `backend=llama-cpp` 并提供本地 GGUF。
"""
from __future__ import annotations

import time

from .base import BaseBackend, GenResult, count_tokens


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

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float = 0.7, stop=None) -> GenResult:
        if not self._loaded:
            raise RuntimeError("模型未加载")
        t0 = time.time()
        mt = max_tokens or self.cfg.default_max_tokens
        tokens = count_tokens(prompt)
        # 确定性输出：回显提示要点 + 明确标注是 mock，避免被误当成真实答案
        tail = prompt.strip().replace("\n", " ")[-160:]
        text = (
            f"[mock:{self.alias}] 已收到 {tokens} 个 token 的请求（temperature={temperature}）。"
            f"当前为本地 mock 后端，用于验证链路与 OpenAI 兼容协议；"
            f"接入真实模型请把 backend 设为 llama-cpp 并放入 GGUF 文件。\n"
            f"提示结尾：{tail}"
        )
        if mt and count_tokens(text) > mt:  # 尊重 max_tokens 上限
            text = text[: max(1, mt * 2)]
        return GenResult(
            text=text,
            prompt_tokens=tokens,
            completion_tokens=count_tokens(text),
            latency_ms=(time.time() - t0) * 1000.0,
            finish_reason="length" if count_tokens(text) >= mt else "stop",
        )
