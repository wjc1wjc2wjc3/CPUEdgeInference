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

    def to_prometheus(self, memory: dict | None = None) -> str:
        """导出为 Prometheus 文本格式（便于接入现有监控体系）。

        GET /metrics?format=prom
        """
        s = self.snapshot()
        lines = [
            "# HELP edgeinfer_requests_total 请求总数",
            "# TYPE edgeinfer_requests_total counter",
            f"edgeinfer_requests_total {s['requests']}",
            "# HELP edgeinfer_errors_total 错误总数",
            "# TYPE edgeinfer_errors_total counter",
            f"edgeinfer_errors_total {s['errors']}",
            "# HELP edgeinfer_tokens_total 累计 token 数",
            "# TYPE edgeinfer_tokens_total counter",
            f"edgeinfer_tokens_total {s['tokens_total']}",
            "# HELP edgeinfer_tokens_per_second 每秒 token 数",
            "# TYPE edgeinfer_tokens_per_second gauge",
            f"edgeinfer_tokens_per_second {s['tokens_per_s']}",
            "# HELP edgeinfer_latency_ms 请求延迟分位数（毫秒）",
            "# TYPE edgeinfer_latency_ms gauge",
            f'edgeinfer_latency_ms{{quantile="p50"}} {s["latency_ms"]["p50"]}',
            f'edgeinfer_latency_ms{{quantile="p95"}} {s["latency_ms"]["p95"]}',
            f'edgeinfer_latency_ms{{quantile="max"}} {s["latency_ms"]["max"]}',
            "# HELP edgeinfer_queue_depth 当前排队/在途请求数",
            "# TYPE edgeinfer_queue_depth gauge",
            f"edgeinfer_queue_depth {s['queue_depth']}",
            "# HELP edgeinfer_uptime_seconds 运行时长",
            "# TYPE edgeinfer_uptime_seconds counter",
            f"edgeinfer_uptime_seconds {s['uptime_s']}",
        ]
        if memory:
            lines += [
                "# HELP edgeinfer_memory_budget_mb 内存预算",
                "# TYPE edgeinfer_memory_budget_mb gauge",
                f"edgeinfer_memory_budget_mb {memory.get('total_mb', 0)}",
                "# HELP edgeinfer_memory_used_mb 已用内存（模型常驻估算）",
                "# TYPE edgeinfer_memory_used_mb gauge",
                f"edgeinfer_memory_used_mb {memory.get('used_mb', 0)}",
                "# HELP edgeinfer_evictions_total LRU 卸载次数",
                "# TYPE edgeinfer_evictions_total counter",
                f"edgeinfer_evictions_total {memory.get('evictions', 0)}",
            ]
        return "\n".join(lines) + "\n"
