"""CPUEdgeInference —— CPU / 边缘优先的轻量本地推理服务。

核心（服务 / 调度 / 模型注册 / 内存预算）**零第三方依赖**，
用 Python 标准库即可启动一个 OpenAI 兼容的本地推理端点。
"""
from .config import Config
from .memory import MemoryBudget, pick_variant
from .metrics import Metrics
from .registry import ModelMeta, Registry
from .server import Engine, serve

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Engine",
    "serve",
    "Registry",
    "ModelMeta",
    "MemoryBudget",
    "pick_variant",
    "Metrics",
]
