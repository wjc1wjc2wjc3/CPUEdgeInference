"""本地可观测：计数 / 延迟 / 吞吐 / 队列深度。

与云端可观测方案的区别：**数据不出本机**，无外发、无埋点，
一个 `/metrics` 就能看清这台边缘设备上发生了什么。
"""
from __future__ import annotations

import threading
import time


class Metrics:
    def __init__(self, keep: int = 200):
        self.requests = 0
        self.errors = 0
        self.tokens = 0
        self.depth = 0
        self.max_depth = 0
        self.latencies: list = []
        self.keep = keep
        self.started_at = time.time()
        self._lock = threading.Lock()

    def record(self, latency_ms: float, tokens: int = 0, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            self.tokens += tokens
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.keep:
                self.latencies.pop(0)

    def enter(self) -> None:
        with self._lock:
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)

    def exit(self) -> None:
        with self._lock:
            self.depth = max(0, self.depth - 1)

    @staticmethod
    def _pct(values: list, p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
        return round(s[idx], 2)

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self.latencies)
            uptime = time.time() - self.started_at
            return {
                "requests": self.requests,
                "errors": self.errors,
                "tokens_total": self.tokens,
                "tokens_per_s": round(self.tokens / uptime, 2) if uptime > 0 else 0.0,
                "latency_ms": {
                    "p50": self._pct(lat, 50),
                    "p95": self._pct(lat, 95),
                    "max": round(max(lat), 2) if lat else 0.0,
                },
                "queue_depth": self.depth,
                "max_queue_depth": self.max_depth,
                "uptime_s": round(uptime, 1),
            }
