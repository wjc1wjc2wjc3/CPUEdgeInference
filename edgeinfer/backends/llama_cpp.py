"""llama.cpp 后端（**CPU only**）：真实本地推理。

强制 `n_gpu_layers=0` —— 本项目定位就是 CPU / 边缘，不依赖 GPU，
也因此能在没有显卡的设备上稳定复现。需可选依赖：

    pip install llama-cpp-python
"""
from __future__ import annotations

import time

from .base import BaseBackend, GenResult, count_tokens


class LlamaCppBackend(BaseBackend):
    name = "llama-cpp"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._llm = None

    def load(self, meta) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "真实推理需要可选依赖：pip install llama-cpp-python"
            ) from exc
        self.meta = meta
        self._llm = Llama(
            model_path=meta.path,
            n_ctx=self.cfg.context_size,
            n_threads=self.cfg.threads,
            n_gpu_layers=0,      # CPU / 边缘优先：不使用 GPU
            use_mmap=True,       # mmap 降低常驻内存
            verbose=False,
        )
        self._loaded = True

    def unload(self) -> None:
        self._llm = None
        self._loaded = False
        self.meta = None

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float = 0.7, stop=None) -> GenResult:
        if not self._loaded or self._llm is None:
            raise RuntimeError("模型未加载")
        t0 = time.time()
        out = self._llm(
            prompt,
            max_tokens=max_tokens or self.cfg.default_max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        text = out["choices"][0]["text"]
        usage = out.get("usage", {}) or {}
        return GenResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", count_tokens(prompt))),
            completion_tokens=int(usage.get("completion_tokens", count_tokens(text))),
            latency_ms=(time.time() - t0) * 1000.0,
            finish_reason=out["choices"][0].get("finish_reason", "stop") or "stop",
        )
