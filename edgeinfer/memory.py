"""内存预算与**自动量化降级**。

边缘设备的核心矛盾是「模型放不下」。这里的策略：
1. 给定内存预算，从同一模型家族的多个量化变体里挑**放得下的最好版本**；
2. 运行时按 LRU 卸载空闲模型，保证任意时刻用量不超预算；
3. 全部卸载事件都计入 metrics，便于观察降级是否频繁发生。
"""
from __future__ import annotations

import time

from .registry import ModelMeta, quality_of


class MemoryBudget:
    """跟踪已加载模型的内存占用，超过预算时按 LRU 卸载。"""

    def __init__(self, total_mb: int):
        self.total_mb = float(total_mb)
        self.loaded: dict = {}      # alias -> est_ram_mb
        self.last_used: dict = {}   # alias -> timestamp
        self.evictions = 0

    @property
    def used_mb(self) -> float:
        return round(sum(self.loaded.values()), 1)

    @property
    def free_mb(self) -> float:
        return round(self.total_mb - self.used_mb, 1)

    def fits(self, mb: float) -> bool:
        return mb <= self.total_mb

    def touch(self, alias: str) -> None:
        self.last_used[alias] = time.time()

    def reserve(self, alias: str, mb: float, protected: str | None = None) -> list:
        """为即将加载的模型腾出空间，返回被卸载的 alias 列表。

        若模型本身大于预算，则不做无谓卸载（返回空，交由上层降级到更小量化）。
        """
        evicted: list = []
        if not self.fits(mb):
            return evicted
        while self.used_mb + mb > self.total_mb:
            candidates = [a for a in self.loaded if a != protected]
            if not candidates:
                break
            victim = min(candidates, key=lambda a: self.last_used.get(a, 0.0))
            self.release(victim)
            evicted.append(victim)
            self.evictions += 1
        self.loaded[alias] = float(mb)
        self.touch(alias)
        return evicted

    def release(self, alias: str) -> None:
        self.loaded.pop(alias, None)
        self.last_used.pop(alias, None)

    def idle_aliases(self, idle_s: float):
        """返回空闲超过阈值、可安全卸载的模型。"""
        now = time.time()
        return [a for a, t in self.last_used.items() if now - t > idle_s]

    def snapshot(self) -> dict:
        return {
            "total_mb": round(self.total_mb, 1),
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "loaded": sorted(self.loaded.keys()),
            "evictions": self.evictions,
        }


def pick_variant(variants: list, budget_mb: float, prefer: str = "auto") -> ModelMeta | None:
    """在预算内挑选**质量最高**的量化变体；prefer 指定时优先该量化。

    返回 None 表示连最小变体都放不下 —— 调用方应据此拒绝加载并给出明确提示，
    而不是让进程 OOM。
    """
    if not variants:
        return None
    ordered = sorted(variants, key=lambda m: quality_of(m.quant), reverse=True)
    if prefer and prefer.lower() != "auto":
        forced = [m for m in ordered if m.quant.upper() == prefer.upper()]
        rest = [m for m in ordered if m.quant.upper() != prefer.upper()]
        ordered = forced + rest
    for m in ordered:
        if m.est_ram_mb <= budget_mb:
            return m
    return None
