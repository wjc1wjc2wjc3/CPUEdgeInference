"""OpenAI 兼容的本地推理服务（**标准库 http.server，零第三方依赖**）。

端点：
    GET  /health                  健康检查（含版本、后端、离线状态、内存）
    GET  /v1/models               列出本地模型（含量化与内存估算）
    POST /v1/chat/completions     OpenAI 兼容对话接口（支持 stream: true / SSE）
    POST /v1/completions          OpenAI 兼容续写接口
    POST /v1/embeddings           OpenAI 兼容向量接口
    GET  /metrics                 指标（?format=prom 输出 Prometheus 文本）
    POST /admin/unload            手动卸载模型释放内存
    OPTIONS *                     CORS 预检（浏览器直连必需）

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
from urllib.parse import parse_qs, urlparse

from .backends import get_backend
from .config import Config
from .memory import MemoryBudget, pick_variant
from .metrics import Metrics
from .registry import Registry, quality_of
from .scheduler import Scheduler

VERSION = "0.2.0"


# ---------------- chat 模板 ----------------
def render_messages(messages, template: str = "auto") -> str:
    """把 OpenAI 的 messages 渲染成纯文本 prompt。

    真实模型对 prompt 格式很敏感：用错模板会明显掉质量，
    因此这里提供几种常见模板，而不是一律拼 `role: content`。
    """
    if template == "auto":
        template = "chatml"  # 当前主流小模型（Qwen / Mistral-instruct 等）多为 ChatML

    parts = []
    system = ""
    turns = []
    for m in messages or []:
        role = m.get("role") or "user"
        content = m.get("content") or ""
        if isinstance(content, list):  # 多模态内容块
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        if role == "system":
            system += (system + "\n" if system else "") + content
        else:
            turns.append((role, content))

    if template == "chatml":
        out = ""
        if system:
            out += f"<|im_start|>system\n{system}<|im_end|>\n"
        for role, content in turns:
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return out + "<|im_start|>assistant\n"

    if template == "llama2":
        out = ""
        if system:
            out += f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
        for i, (role, content) in enumerate(turns):
            if role == "user":
                nxt = turns[i + 1][1] if i + 1 < len(turns) and turns[i + 1][0] == "assistant" else ""
                out += (f"[INST] {content} [/INST]" + (f" {nxt}\n" if nxt else "\n"))
            elif role == "assistant" and (i == 0 or turns[i - 1][0] != "user"):
                out += f"{content}\n"
        return out

    if template == "gemma":
        out = ""
        if system:
            out += f"<start_of_turn>system\n{system}<end_of_turn>\n"
        for role, content in turns:
            tag = "user" if role == "user" else "model"
            out += f"<start_of_turn>{tag}\n{content}<end_of_turn>\n"
        return out + "<start_of_turn>model\n"

    # plain
    if system:
        parts.append(f"system: {system}")
    for role, content in turns:
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
        self.version = VERSION
        self._lock = threading.RLock()
        self._reaper_stop = None

    # ---------------- 后台任务 ----------------
    def start_background_tasks(self, interval_s: float = 30.0) -> None:
        """启动空闲模型回收线程（边缘设备长期运行必备）。

        `serve()` 会自动调用；进程内直接使用 Engine 时可手动调用。
        """
        if self._reaper_stop is not None:
            return
        self._reaper_stop = threading.Event()

        def loop():
            while not self._reaper_stop.wait(interval_s):
                try:
                    self.reap_idle()
                except Exception:
                    pass

        threading.Thread(target=loop, daemon=True).start()

    def stop_background_tasks(self) -> None:
        if self._reaper_stop is not None:
            self._reaper_stop.set()
            self._reaper_stop = None

    # ---------------- 模型 ----------------
    def list_models(self) -> list:
        return [m.to_dict() for m in self.registry.scan()]

    def verify_models(self) -> list:
        """逐个计算 sha256，校验本地模型完整性（离线部署前自检）。"""
        out = []
        for m in self.registry.scan():
            digest = self.registry.sha256(m)
            m.sha256 = digest
            out.append({"alias": m.alias, "sha256": digest,
                        "size_mb": m.size_mb, "family": m.family,
                        "quant": m.quant, "est_ram_mb": m.est_ram_mb})
        return out

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
    def _sampling_params(self, top_p=None, top_k=None, seed=None) -> dict:
        p = {}
        if top_p is not None:
            p["top_p"] = top_p
        if top_k is not None:
            p["top_k"] = top_k
        if seed is not None:
            p["seed"] = seed
        return p

    def chat(self, model: str | None, messages, temperature: float = 0.7,
             max_tokens: int | None = None, stream: bool = False,
             stop=None, top_p: float | None = None,
             top_k: int | None = None, seed: int | None = None):
        """非流式返回 dict；流式返回 chunk 生成器（OpenAI SSE 结构）。"""
        meta = self.resolve(model)
        be = self.ensure_loaded(meta)
        prompt = render_messages(messages, self.cfg.chat_template)
        params = self._sampling_params(top_p, top_k, seed)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]

        if not stream:
            t0 = time.time()
            try:
                res = self.scheduler.submit(
                    lambda: be.generate(prompt, max_tokens, temperature, stop, **params))
            except Exception:
                self.metrics.record((time.time() - t0) * 1000.0, 0, error=True)
                raise
            self._after(meta, be, prompt, res.prompt_tokens + res.completion_tokens,
                        res.latency_ms)
            return {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": meta.alias,
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": res.text},
                             "finish_reason": res.finish_reason}],
                "usage": res.usage(),
                "local": {"quant": meta.quant, "est_ram_mb": meta.est_ram_mb,
                          "backend": be.name, "threads": self.cfg.threads,
                          "chat_template": self.cfg.chat_template},
            }

        # ---- 流式 ----
        def gen():
            t0 = time.time()
            total = 0
            try:
                for delta in be.stream_generate(prompt, max_tokens, temperature,
                                                stop, **params):
                    total += len(delta) // 2 or 1
                    yield {"id": cid, "object": "chat.completion.chunk",
                           "created": int(time.time()), "model": meta.alias,
                           "choices": [{"index": 0, "delta": {"content": delta},
                                        "finish_reason": None}]}
                yield {"id": cid, "object": "chat.completion.chunk",
                       "created": int(time.time()), "model": meta.alias,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self._after(meta, be, prompt, total, (time.time() - t0) * 1000.0)
            except Exception:
                self.metrics.record((time.time() - t0) * 1000.0, 0, error=True)
                raise

        return gen()

    def embeddings(self, model: str | None, inputs) -> dict:
        """OpenAI 兼容 /v1/embeddings。"""
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            raise ValueError("input 不能为空")
        meta = self.resolve(model)
        be = self.ensure_loaded(meta)
        try:
            vecs = self.scheduler.submit(lambda: be.embed(inputs))
        except NotImplementedError:
            raise RuntimeError(
                f"当前后端 {be.name} 不支持 embeddings；"
                "真实环境请用 --backend llama-cpp"
            )
        used = sum(len(v) for v in vecs)
        self.metrics.record(0.0, used)
        return {
            "object": "list",
            "model": meta.alias,
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": sum(len(str(x)) // 2 for x in inputs),
                      "total_tokens": sum(len(str(x)) // 2 for x in inputs)},
        }

    def _after(self, meta, be, prompt: str, tokens: int, latency_ms: float) -> None:
        self.metrics.record(latency_ms, tokens)
        self._log({
            "ts": time.time(), "model": meta.alias, "quant": meta.quant,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "tokens": tokens, "latency_ms": round(latency_ms, 2),
            "offline": self.cfg.is_offline,
        })

    def _log(self, event: dict) -> None:
        try:
            with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 日志失败不应影响推理


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"CPUEdgeInference/{VERSION}"

    # ---- 工具 ----
    def _cors(self) -> None:
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        if e.cfg.cors_enabled:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "86400")

    def _authorized(self) -> bool:
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        key = e.cfg.api_key
        if not key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {key}"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, text: str, ctype: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks) -> None:
        """SSE 流式响应（手工 chunked 编码，标准库实现）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

        def write(raw: bytes) -> None:
            self.wfile.write(f"{len(raw):X}\r\n".encode("ascii") + raw + b"\r\n")
            self.wfile.flush()

        try:
            for obj in chunks:
                write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            write(b"data: [DONE]\n\n")
        finally:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---- 路由 ----
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        try:
            if not self._authorized():
                return self._send(401, {"error": {"message": "unauthorized"}})
            if path == "/health":
                self._send(200, {"status": "ok", "version": e.version,
                                 "backend": e.cfg.backend,
                                 "chat_template": e.cfg.chat_template,
                                 "offline": e.cfg.is_offline, "threads": e.cfg.threads,
                                 "memory": e.mem.snapshot()})
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": m["alias"], "object": "model", "owned_by": "local", "meta": m}
                    for m in e.list_models()]})
            elif path == "/metrics":
                if query.get("format", [""])[0].lower() in ("prom", "prometheus"):
                    self._send_text(200, e.metrics.to_prometheus(e.mem.snapshot()),
                                    "text/plain; version=0.0.4; charset=utf-8")
                else:
                    self._send(200, {"metrics": e.metrics.snapshot(),
                                     "memory": e.mem.snapshot()})
            else:
                self._send(404, {"error": {"message": f"not found: {path}"}})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": {"message": str(exc)}})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        e: Engine = self.server.engine  # type: ignore[attr-defined]
        try:
            if not self._authorized():
                return self._send(401, {"error": {"message": "unauthorized"}})
            req = self._body()

            if path == "/v1/chat/completions":
                stream = bool(req.get("stream", False))
                out = e.chat(req.get("model", "auto"), req.get("messages", []),
                             req.get("temperature", 0.7), req.get("max_tokens"),
                             stream=stream, stop=req.get("stop"),
                             top_p=req.get("top_p"), top_k=req.get("top_k"),
                             seed=req.get("seed"))
                if stream:
                    self._send_sse(out)
                else:
                    self._send(200, out)

            elif path == "/v1/completions":
                r = e.chat(req.get("model", "auto"),
                           [{"role": "user", "content": req.get("prompt", "")}],
                           req.get("temperature", 0.7), req.get("max_tokens"),
                           stream=False, stop=req.get("stop"),
                           top_p=req.get("top_p"), top_k=req.get("top_k"),
                           seed=req.get("seed"))
                self._send(200, {
                    "id": r["id"], "object": "text_completion", "created": r["created"],
                    "model": r["model"],
                    "choices": [{"index": 0, "text": r["choices"][0]["message"]["content"],
                                 "finish_reason": r["choices"][0]["finish_reason"]}],
                    "usage": r["usage"], "local": r["local"],
                })

            elif path == "/v1/embeddings":
                self._send(200, e.embeddings(req.get("model", "auto"),
                                             req.get("input", [])))

            elif path == "/admin/unload":
                e.unload(req.get("model", ""))
                self._send(200, {"ok": True, "memory": e.mem.snapshot()})

            elif path == "/admin/verify":
                self._send(200, {"models": e.verify_models()})

            else:
                self._send(404, {"error": {"message": f"not found: {path}"}})
        except (KeyError, RuntimeError, TimeoutError, ValueError,
                NotImplementedError) as exc:
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
    engine.start_background_tasks()
    srv = Server((cfg.host, cfg.port), engine)
    print(f"CPUEdgeInference v{VERSION} 监听 http://{cfg.host}:{cfg.port} "
          f"(backend={cfg.backend}, threads={cfg.threads}, "
          f"budget={cfg.memory_budget_mb}MB, offline={cfg.is_offline}, "
          f"template={cfg.chat_template})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop_background_tasks()
        srv.server_close()
    return srv
