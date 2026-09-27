"""CPUEdgeInference 核心测试（标准库 unittest，无需 pytest）。

    py -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from edgeinfer.config import Config
from edgeinfer.memory import MemoryBudget, pick_variant
from edgeinfer.registry import ModelMeta, Registry, detect_quant, family_of
from edgeinfer.server import Engine, Server, render_messages

DUMMY_MODELS = [
    "tinyllama-1.1b-chat.Q4_K_M.gguf",
    "tinyllama-1.1b-chat.Q5_K_M.gguf",
    "qwen2-0.5b-instruct.Q8_0.gguf",
]


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in DUMMY_MODELS:
            with open(os.path.join(self.tmp.name, name), "wb") as f:
                f.write(b"\x00" * 1024)  # 占位文件，仅测解析逻辑

    def tearDown(self):
        self.tmp.cleanup()

    def test_quant_and_family_detection(self):
        self.assertEqual(detect_quant("x-1b.Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(detect_quant("x-1b.F16.gguf"), "F16")
        self.assertEqual(family_of("tinyllama-1.1b-chat.Q4_K_M.gguf", "Q4_K_M"),
                         "tinyllama-1.1b-chat")

    def test_families_group_variants(self):
        reg = Registry(self.tmp.name)
        fams = reg.families()
        self.assertIn("tinyllama-1.1b-chat", fams)
        self.assertEqual(len(fams["tinyllama-1.1b-chat"]), 2)  # Q4_K_M + Q5_K_M


def _meta(alias, quant, ram):
    return ModelMeta(alias=alias, path="/tmp/" + alias, family="f",
                     quant=quant, size_mb=ram, est_ram_mb=ram)


class TestMemory(unittest.TestCase):
    def test_pick_best_affordable_quant(self):
        variants = [_meta("a.F16", "F16", 2000), _meta("a.Q4_K_M", "Q4_K_M", 600)]
        self.assertEqual(pick_variant(variants, 800).quant, "Q4_K_M")  # 预算内选最好的
        self.assertEqual(pick_variant(variants, 3000).quant, "F16")

    def test_pick_returns_none_when_nothing_fits(self):
        self.assertIsNone(pick_variant([_meta("a.F16", "F16", 9000)], 500))

    def test_lru_eviction_within_budget(self):
        mem = MemoryBudget(1000)
        mem.reserve("x", 600)
        mem.reserve("y", 500)  # 超限 → 应卸载最早使用的 x
        self.assertNotIn("x", mem.loaded)
        self.assertIn("y", mem.loaded)
        self.assertLessEqual(mem.used_mb, 1000)
        self.assertEqual(mem.evictions, 1)


class TestServer(unittest.TestCase):
    """端到端：起真实 HTTP 服务（mock 后端，无需模型权重）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        for name in DUMMY_MODELS:
            with open(os.path.join(cls.tmp.name, name), "wb") as f:
                f.write(b"\x00" * 512)
        cls.cfg = Config(model_dir=cls.tmp.name, backend="mock",
                         memory_budget_mb=2048, threads=1, max_concurrency=2,
                         log_path=os.path.join(cls.tmp.name, "serve.jsonl"))
        cls.engine = Engine(cls.cfg)
        cls.srv = Server(("127.0.0.1", 0), cls.engine)  # port 0 → 随机端口
        cls.port = cls.srv.server_address[1]
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_health_and_models(self):
        self.assertEqual(self._get("/health")["status"], "ok")
        models = self._get("/v1/models")["data"]
        self.assertEqual(len(models), 3)
        self.assertTrue(any(m["id"].endswith(".gguf") for m in models))

    def test_chat_completions_openai_shape(self):
        status, r = self._post("/v1/chat/completions", {
            "model": "auto",
            "messages": [{"role": "user", "content": "你好，做个自我介绍"}],
            "max_tokens": 64,
        })
        self.assertEqual(status, 200)
        self.assertEqual(r["object"], "chat.completion")
        self.assertEqual(r["choices"][0]["message"]["role"], "assistant")
        self.assertIn("content", r["choices"][0]["message"])
        self.assertIn("total_tokens", r["usage"])
        # auto 应在预算内自动选中一个量化变体
        self.assertIn(r["local"]["quant"], {"Q4_K_M", "Q5_K_M", "Q8_0"})

    def test_metrics_and_unload(self):
        self._post("/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        m = self._get("/metrics")
        self.assertGreaterEqual(m["metrics"]["requests"], 1)
        self.assertIn("p50", m["metrics"]["latency_ms"])

        status, r = self._post("/admin/unload", {"model": "tinyllama-1.1b-chat.Q4_K_M.gguf"})
        self.assertEqual(status, 200)

    def test_unknown_model_returns_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/chat/completions", {"model": "not-exist", "messages": []})
        self.assertEqual(ctx.exception.code, 400)


def _mk_engine(tmp, **kw):
    cfg = Config(model_dir=tmp, backend="mock", memory_budget_mb=2048, threads=1,
                 log_path=os.path.join(tmp, "s.jsonl"), **kw)
    return Engine(cfg)


class TestChatTemplates(unittest.TestCase):
    """真实模型对 prompt 格式敏感，模板必须正确。"""

    def test_chatml(self):
        out = render_messages([{"role": "system", "content": "be nice"},
                               {"role": "user", "content": "hi"}], "chatml")
        self.assertIn("<|im_start|>system", out)
        self.assertIn("<|im_end|>", out)
        self.assertTrue(out.endswith("<|im_start|>assistant\n"))

    def test_auto_defaults_to_chatml(self):
        self.assertEqual(
            render_messages([{"role": "user", "content": "hi"}], "auto"),
            render_messages([{"role": "user", "content": "hi"}], "chatml"))

    def test_plain(self):
        out = render_messages([{"role": "user", "content": "hi"}], "plain")
        self.assertIn("user: hi", out)
        self.assertTrue(out.endswith("assistant:"))

    def test_gemma(self):
        out = render_messages([{"role": "user", "content": "hi"}], "gemma")
        self.assertIn("<start_of_turn>user", out)
        self.assertTrue(out.endswith("<start_of_turn>model\n"))

    def test_llama2(self):
        out = render_messages([{"role": "user", "content": "hi"}], "llama2")
        self.assertIn("[INST]", out)


class TestVerify(unittest.TestCase):
    def test_sha256_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "m.Q4_K_M.gguf"), "wb") as f:
                f.write(b"abc")
            res = _mk_engine(tmp).verify_models()
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["sha256"], hashlib.sha256(b"abc").hexdigest())


class TestStopAndSampling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in DUMMY_MODELS:
            with open(os.path.join(self.tmp.name, name), "wb") as f:
                f.write(b"\x00" * 512)
        self.engine = _mk_engine(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stop_truncates_output(self):
        r = self.engine.chat("auto", [{"role": "user", "content": "hi"}], stop=["]"])
        self.assertNotIn("]", r["choices"][0]["message"]["content"])

    def test_sampling_params_accepted(self):
        r = self.engine.chat("auto", [{"role": "user", "content": "hi"}],
                             top_p=0.9, top_k=40, seed=42)
        self.assertEqual(r["object"], "chat.completion")


class TestNewEndpoints(unittest.TestCase):
    """流式 / CORS / embeddings / Prometheus / verify 的 HTTP 端到端。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        for name in DUMMY_MODELS:
            with open(os.path.join(cls.tmp.name, name), "wb") as f:
                f.write(b"\x00" * 512)
        cls.engine = _mk_engine(cls.tmp.name)
        cls.srv = Server(("127.0.0.1", 0), cls.engine)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.headers, r.read().decode("utf-8")

    def test_stream_sse(self):
        headers, body = self._post("/v1/chat/completions", {
            "model": "auto", "stream": True,
            "messages": [{"role": "user", "content": "讲个故事"}],
        })
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        self.assertIn("data: [DONE]", body)
        chunks = []
        for line in body.splitlines():
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                chunks.append(json.loads(line[6:])["choices"][0]["delta"].get("content", ""))
        self.assertTrue("".join(chunks).strip())
        self.assertGreater(len(chunks), 1)  # 确实是多块增量，而非一大块

    def test_cors_preflight(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/models", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 204)
            self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "*")

    def test_embeddings(self):
        _, body = self._post("/v1/embeddings", {
            "model": "auto", "input": ["hello", "世界"]})
        r = json.loads(body)
        self.assertEqual(r["object"], "list")
        self.assertEqual(len(r["data"]), 2)
        self.assertGreater(len(r["data"][0]["embedding"]), 0)

    def test_metrics_prometheus(self):
        self._post("/v1/chat/completions", {"model": "auto",
                                            "messages": [{"role": "user", "content": "hi"}]})
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/metrics?format=prom", timeout=10) as r:
            text = r.read().decode("utf-8")
        self.assertIn("edgeinfer_requests_total", text)
        self.assertIn("edgeinfer_latency_ms", text)
        self.assertIn("edgeinfer_memory_budget_mb", text)

    def test_admin_verify(self):
        _, body = self._post("/admin/verify", {})
        models = json.loads(body)["models"]
        self.assertEqual(len(models), 3)
        self.assertEqual(len(models[0]["sha256"]), 64)

    def test_health_has_version(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/health", timeout=10) as r:
            h = json.loads(r.read().decode("utf-8"))
        self.assertIn("version", h)
        self.assertIn("chat_template", h)


class TestApiKey(unittest.TestCase):
    """可选的 Bearer 校验。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        for name in DUMMY_MODELS:
            with open(os.path.join(cls.tmp.name, name), "wb") as f:
                f.write(b"\x00" * 512)
        cls.engine = _mk_engine(cls.tmp.name, api_key="secret")
        cls.srv = Server(("127.0.0.1", 0), cls.engine)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def _get(self, path, auth=None):
        headers = {}
        if auth:
            headers["Authorization"] = auth
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_reject_without_key(self):
        self.assertEqual(self._get("/v1/models"), 401)

    def test_accept_with_key(self):
        self.assertEqual(self._get("/v1/models", "Bearer secret"), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
