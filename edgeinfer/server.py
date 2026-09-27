"""OpenAI 兼容的本地推理服务（**标准库 http.server，零第三方依赖**）。

端点：
    GET  /health                  健康检查 + 内存/线程/离线状态
    GET  /v1/models               列出本地模型（含量化与内存估算）
    POST /v1/chat/completions     OpenAI 兼容对话接口
    POST /v1/completions          OpenAI 兼容续写接口
    GET  /metrics                 请求数 / 延迟 p50-p95 / tokens / 队列深度
    POST /admin/unload            手动卸载模型释放内存

设计取舍：不上 FastAPI 是为了保证「装都不用装就能跑」，
在树莓派 / 内网盒子 / 离线环境里这一点比框架优雅更重要。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .backends import get_backend
from .config import Config
from .memory import MemoryBudget, pick_variant
from .metrics import Metrics
from .registry import Registry, quality_of
from .scheduler import Scheduler


def render_messages(messages) -> str:
    """把 OpenAI 的 messages 渲染成纯文本 prompt（无 chat template 依赖）。"""
    parts = []
    for m in messages or []:
        role = m.get("role") or "user"
        content = m.get("content") or ""
        if isinstance(content, list):  # 多模态内容块
            content = " ".join(
                str(c.get("text", "")) for c in content if isinstance(c, dict)
            )
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


class Engine:
    """模型选择 / 加载 / 内存预算 / 调度的核心。"""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.cfg.apply_offline_env()
        self.registry = Registry(self.cfg.model_dir, self.cfg.context_size)
        self.mem = MemoryBudget(self.cfg.memory_budget_mb)
        self.metrics = Metrics()
        self.scheduler = Scheduler(self.cfg, self.metrics)
        self.backends: dict = {}
        self._lock = threading.RLock()

    # ---------------- 模型 ----------------
    def list_models(self) -> list:
        return [m.to_dict() for m in self.registry.scan()]

    def resolve(self, requested: str | None):
        """解析模型：auto 时按内存预算自动挑**放得下的最好量化**。"""
        fams = self.registry.families()
        if requested and requested not in ("auto", "", None):
            if requested in fams:
                m = pick_variant(fams[requested], self.mem.total_mb, self.cfg.prefer_quant)
                if m:
                    return m
                raise RuntimeError(
                    f"模型家族 {requested} 没有任何量化变体能放进 "
                    f"{self.mem.total_mb}MB 预算"
                )
            m = self.registry.get(requested)
            if m:
                return m
            raise KeyError(f"未找到模型：{requested}（可用：{', '.join(sorted(fams)) or '无'}）")

        if not fams:
            raise RuntimeError(
                f"模型目录 {self.cfg.model_dir} 里没有本地模型（*.gguf/*.bin）。"
                "本项目不做任何联网下载，请把权重文件放到该目录。"
            )
        cands = []
        for _fam, variants in fams.items():
            m = pick_variant(variants, self.mem.total_mb, self.cfg.prefer_quant)
            if m:
                cands.append(m)
        if not cands:
            raise RuntimeError(
                f"没有任何模型能放进 {self.mem.total_mb}MB 内存预算，"
                "请换更小量化（如 Q4_K_M / Q3_K_M）或调高 --memory-budget-mb"
            )
        return max(cands, key=lambda m: quality_of(m.quant))

    def ensure_loaded(self, meta):
        with self._lock:
            be = self.backends.get(meta.alias)
            if be is not None and be.loaded:
                self.mem.touch(meta.alias)
                return be
            if meta.est_ram_mb > self.mem.total_mb:
                raise RuntimeError(
                    f"模型 {meta.alias} 约需 {meta.est_ram_mb}MB，"
                    f"超过内存预算 {self.mem.total_mb}MB"
                )
            for victim in self.mem.reserve(meta.alias, meta.est_ram_mb):
                old = self.backends.pop(victim, None)
                if old:
                    old.unload()
            be = get_backend(self.cfg)
            be.load(meta)
            self.backends[meta.alias] = be
            return be

    def unload(self, alias: str) -> None:
        with self._lock:
            be = self.backends.pop(alias, None)
            if be:
                be.unload()
            self.mem.release(alias)

    def reap_idle(self) -> list:
        """卸载空闲模型，主动释放内存（边缘设备关键能力）。"""
        with self._lock:
            out = []
            for alias in self.mem.idle_aliases(self.cfg.idle_unload_s):
                self.unload(alias)
                out.append(alias)
            return out

    # ---------------- 推理 ----------------
    def chat(self, model: str | None, messages, temperature: float = 0.7,
             max_tokens: int | None = None) -> dict:
        t0 = time.time()
        try:
            meta = self.resolve(model)
            be = self.ensure_loaded(meta)
            prompt = render_messages(messages)
            res = self.scheduler.submit(
                lambda: be.generate(prompt, max_tokens, temperature)
            )
            self.metrics.record(res.latency_ms,
                                res.prompt_tokens + res.completion_tokens)
            self._log({
                "ts": time.time(), "model": meta.alias, "quant": meta.quant,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                "tokens": res.prompt_tokens + res.completion_tokens,
                "latency_ms": round(res.latency_ms, 2),
                "offline": self.cfg.is_offline,
            })
            return {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": meta.alias,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": res.finish_reason,
                }],
                "usage": res.usage(),
                "local": {"quant": meta.quant, "est_ram_mb": meta.est_ram_mb,
                          "backend": be.name, "threads": self.cfg.threads},
            }
        except Exception:
            self.metrics.record((time.time() - t0) * 1000.0, 0, error=True)
            raise

    def _log(self, event: dict) -> None:
        try:
            with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 日志失败不应影响推理


class Handler(BaseHTTPRequestHandler):
    server_version = "CPUEdgeInference/0.1"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        try:
            if path == "/health":
                self._send(200, {"status": "ok", "backend": e.cfg.backend,
                                 "offline": e.cfg.is_offline, "threads": e.cfg.threads,
                                 "memory": e.mem.snapshot()})
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": m["alias"], "object": "model", "owned_by": "local", "meta": m}
                    for m in e.list_models()]})
            elif path == "/metrics":
                self._send(200, {"metrics": e.metrics.snapshot(), "memory": e.mem.snapshot()})
            else:
                self._send(404, {"error": {"message": f"not found: {path}"}})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": {"message": str(exc)}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        try:
            req = self._body()
            if path == "/v1/chat/completions":
                self._send(200, e.chat(req.get("model", "auto"), req.get("messages", []),
                                      req.get("temperature", 0.7), req.get("max_tokens")))
            elif path == "/v1/completions":
                r = e.chat(req.get("model", "auto"),
                           [{"role": "user", "content": req.get("prompt", "")}],
                           req.get("temperature", 0.7), req.get("max_tokens"))
                self._send(200, {
                    "id": r["id"], "object": "text_completion", "created": r["created"],
                    "model": r["model"],
                    "choices": [{"index": 0, "text": r["choices"][0]["message"]["content"],
                                 "finish_reason": r["choices"][0]["finish_reason"]}],
                    "usage": r["usage"], "local": r["local"],
                })
            elif path == "/admin/unload":
                e.unload(req.get("model", ""))
                self._send(200, {"ok": True, "memory": e.mem.snapshot()})
            else:
                self._send(404, {"error": {"message": f"not found: {path}"}})
        except (KeyError, RuntimeError, TimeoutError, ValueError) as exc:
            self._send(400, {"error": {"message": str(exc), "type": type(exc).__name__}})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": {"message": str(exc), "type": type(exc).__name__}})

    def log_message(self, fmt, *args) -> None:
        pass  # 安静模式：不把每次请求打到 stderr


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, engine: Engine):
        self.engine = engine
        super().__init__(addr, Handler)


def create_engine(cfg: Config | None = None) -> Engine:
    return Engine(cfg)


def serve(cfg: Config | None = None):
    cfg = cfg or Config()
    engine = Engine(cfg)
    srv = Server((cfg.host, cfg.port), engine)

    stop = threading.Event()

    def reaper():
        while not stop.wait(30):
            try:
                engine.reap_idle()
            except Exception:
                pass

    threading.Thread(target=reaper, daemon=True).start()
    print(f"CPUEdgeInference 监听 http://{cfg.host}:{cfg.port} "
          f"(backend={cfg.backend}, threads={cfg.threads}, "
          f"budget={cfg.memory_budget_mb}MB, offline={cfg.is_offline})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return srv
