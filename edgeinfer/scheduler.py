"""受限并发调度：边缘设备上「排队」比「同时跑爆内存」更重要。

- 用信号量把并发推理限制在 `max_concurrency`；
- 等待超过 `request_timeout_s` 直接失败，而不是无限堆积；
- 队列深度进 metrics，便于判断是否需要扩容或降级。
"""
from __future__ import annotations

import threading


class Scheduler:
    def __init__(self, cfg, metrics):
        self.cfg = cfg
        self.metrics = metrics
        self._sem = threading.BoundedSemaphore(max(1, cfg.max_concurrency))

    def submit(self, fn):
        """在并发额度内执行 fn；拿不到额度则 TimeoutError。"""
        if not self._sem.acquire(timeout=self.cfg.request_timeout_s):
            raise TimeoutError(
                f"等待推理槽位超时（max_concurrency={self.cfg.max_concurrency}）"
            )
        self.metrics.enter()
        try:
            return fn()
        finally:
            self.metrics.exit()
            self._sem.release()
