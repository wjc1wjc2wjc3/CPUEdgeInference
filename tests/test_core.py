"""CPUEdgeInference 核心测试（标准库 unittest，无需 pytest）。

    py -m unittest discover -s tests -v
"""
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
from edgeinfer.server import Engine, Server

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
