"""CPUEdgeInference 配置：**CPU / 边缘优先**，默认离线。

与常见推理服务的区别：这里没有「GPU 显存」概念，一切围绕
**可用内存预算**来做模型选择与降级，目标是在树莓派 / 老旧笔记本 /
边缘盒子上也能把模型跑起来。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def auto_threads() -> int:
    n = os.cpu_count() or 2
    return max(1, min(4, n))  # 边缘设备上保守占用，避免打满 CPU


def total_ram_mb() -> int:
    """尽力探测总内存；失败时回退到保守值 4096MB。"""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys // (1024 * 1024))
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        return int(page * phys / (1024 * 1024))
    except Exception:
        return 4096


@dataclass
class Config:
    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = 8080

    # ---- 模型 ----
    model_dir: str = "./models"
    backend: str = "mock"          # mock | llama-cpp
    prefer_quant: str = "auto"     # auto | Q4_K_M | Q5_K_M | Q8_0 | F16 ...

    # ---- 资源预算（边缘核心）----
    memory_budget_mb: int = 0      # 0 = 自动取「总内存的 60%」
    threads: int = 0               # 0 = auto（min(4, cpu)）
    max_concurrency: int = 2       # 同时推理的请求数
    context_size: int = 2048
    default_max_tokens: int = 256
    request_timeout_s: float = 120.0
    idle_unload_s: float = 300.0   # 模型空闲多久后卸载，释放内存

    # ---- 离线 ----
    offline_only: bool = True
    allow_network: bool = False
    log_path: str = "serve.jsonl"

    def __post_init__(self) -> None:
        if self.memory_budget_mb <= 0:
            self.memory_budget_mb = max(512, int(total_ram_mb() * 0.6))
        if self.threads <= 0:
            self.threads = auto_threads()

    def apply_offline_env(self) -> None:
        if self.is_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["EDGEINFER_NO_NETWORK"] = "1"

    @property
    def is_offline(self) -> bool:
        return bool(self.offline_only and not self.allow_network)

    def to_dict(self) -> dict:
        return asdict(self)
