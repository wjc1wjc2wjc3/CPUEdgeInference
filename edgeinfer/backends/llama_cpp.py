"""llama.cpp 后端（**CPU only**）：真实本地推理。

强制 `n_gpu_layers=0` —— 本项目定位就是 CPU / 边缘，不依赖 GPU，
也因此能在没有显卡的设备上稳定复现。需可选依赖：

    pip install llama-cpp-python
"""
from __future__ import annotations

import time

from .base import BaseBackend, GenResult, apply_stop, count_tokens


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

    def _kwargs(self, max_tokens, temperature, stop, params):
        kw = {
            "max_tokens": max_tokens or self.cfg.default_max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        }
        # 透传常见采样参数（若底层不支持则忽略）
        for key in ("top_p", "top_k", "seed", "repeat_penalty", "mirostat_mode"):
            if params.get(key) is not None:
                kw[key] = params[key]
        return kw

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float = 0.7, stop=None, **params) -> GenResult:
        if not self._loaded or self._llm is None:
            raise RuntimeError("模型未加载")
        t0 = time.time()
        out = self._llm(prompt, **self._kwargs(max_tokens, temperature, stop, params))
        text = out["choices"][0]["text"]
        usage = out.get("usage", {}) or {}
        return GenResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", count_tokens(prompt))),
            completion_tokens=int(usage.get("completion_tokens", count_tokens(text))),
            latency_ms=(time.time() - t0) * 1000.0,
            finish_reason=out["choices"][0].get("finish_reason", "stop") or "stop",
        )

    def stream_generate(self, prompt: str, max_tokens: int | None = None,
                        temperature: float = 0.7, stop=None, **params):
        if not self._loaded or self._llm is None:
            raise RuntimeError("模型未加载")
        stream = self._llm(prompt, stream=True,
                           **self._kwargs(max_tokens, temperature, stop, params))
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if delta:
                yield delta

    def embed(self, texts) -> list:
        if not self._loaded or self._llm is None:
            raise RuntimeError("模型未加载")
        if isinstance(texts, str):
            texts = [texts]
        # 新版 llama-cpp-python 提供 create_embedding；旧版提供 embed
        if hasattr(self._llm, "create_embedding"):
            out = self._llm.create_embedding(input=texts)
            return [d["embedding"] for d in out.get("data", [])]
        return [list(self._llm.embed(t)) for t in texts]
