"""零依赖快速上手：起本地服务 → OpenAI 兼容调用 → 看 metrics

    py examples/quickstart.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgeinfer.config import Config  # noqa: E402
from edgeinfer.server import Engine, Server  # noqa: E402

# 演示用的「模型文件」占位（真实使用请放入真正的 GGUF 权重）
DEMO_MODELS = [
    "tinyllama-1.1b-chat.Q4_K_M.gguf",
    "tinyllama-1.1b-chat.Q5_K_M.gguf",
    "qwen2-0.5b-instruct.Q8_0.gguf",
]


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="edgeinfer-demo-")
    for name in DEMO_MODELS:
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(b"\x00" * 1024)

    cfg = Config(model_dir=tmp, backend="mock", memory_budget_mb=2048,
                 threads=1, max_concurrency=2,
                 log_path=os.path.join(tmp, "serve.jsonl"))
    engine = Engine(cfg)
    srv = Server(("127.0.0.1", 0), engine)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"服务已启动：http://127.0.0.1:{port}\n")

    print("== 本地模型（含量化与内存估算）==")
    for m in _get(port, "/v1/models")["data"]:
        print(f"  {m['id']:<40} quant={m['meta']['quant']:<8} "
              f"est_ram={m['meta']['est_ram_mb']}MB")

    print("\n== OpenAI 兼容调用（model=auto，按内存预算自动选量化）==")
    _, r = _post(port, "/v1/chat/completions", {
        "model": "auto",
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 64,
    })
    print("  选中模型：", r["model"], "| quant:", r["local"]["quant"])
    print("  回复：", r["choices"][0]["message"]["content"][:120], "...")
    print("  usage：", r["usage"])

    print("\n== 降级演示：把预算压到 80MB 后重新解析 ==")
    engine.mem.total_mb = 80.0
    try:
        meta = engine.resolve("tinyllama-1.1b-chat")
        print(f"  预算 80MB 下选中：{meta.alias}（quant={meta.quant}）")
    except RuntimeError as exc:
        print("  无法加载：", exc)

    print("\n== 指标 ==")
    print(" ", json.dumps(_get(port, "/metrics")["metrics"], ensure_ascii=False))

    srv.shutdown()
    srv.server_close()
    print("\n完成。（真实推理请把 backend 设为 llama-cpp 并放入 GGUF 权重）")


if __name__ == "__main__":
    main()
