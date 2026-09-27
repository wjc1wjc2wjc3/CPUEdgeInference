"""CPUEdgeInference 命令行（标准库 argparse，零依赖）。

    py -m edgeinfer.cli models                       # 查看本地模型与内存估算
    py -m edgeinfer.cli serve --port 8080            # 启动 OpenAI 兼容服务
    py -m edgeinfer.cli chat "你好" --model auto     # 免启动服务直接对话
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .server import Engine, serve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edgeinfer",
        description="CPUEdgeInference：CPU / 边缘优先的轻量本地推理服务",
    )
    p.add_argument("--model-dir", default="./models", help="本地模型目录（不联网下载）")
    p.add_argument("--backend", default="mock", choices=["mock", "llama-cpp"])
    p.add_argument("--memory-budget-mb", type=int, default=0, help="0=自动取总内存 60%")
    p.add_argument("--threads", type=int, default=0, help="0=auto（min(4, cpu)）")
    p.add_argument("--context-size", type=int, default=2048)
    p.add_argument("--prefer-quant", default="auto", help="auto 或指定如 Q4_K_M")
    p.add_argument("--max-concurrency", type=int, default=2)
    p.add_argument("--allow-network", action="store_true", help="显式允许联网（默认全离线）")

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="启动服务")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)

    pm = sub.add_parser("models", help="列出本地模型")
    pm.add_argument("--json", action="store_true")

    pc = sub.add_parser("chat", help="免启动服务直接对话")
    pc.add_argument("message")
    pc.add_argument("--model", default="auto")
    pc.add_argument("--max-tokens", type=int, default=None)
    return p


def _cfg(args) -> Config:
    return Config(
        model_dir=args.model_dir, backend=args.backend,
        memory_budget_mb=args.memory_budget_mb, threads=args.threads,
        context_size=args.context_size, prefer_quant=args.prefer_quant,
        max_concurrency=args.max_concurrency, allow_network=args.allow_network,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "serve":
        cfg = _cfg(args)
        cfg.host, cfg.port = args.host, args.port
        serve(cfg)
        return 0

    if args.cmd == "models":
        cfg = _cfg(args)
        engine = Engine(cfg)
        models = engine.list_models()
        if args.json:
            print(json.dumps(models, ensure_ascii=False, indent=2))
            return 0
        if not models:
            print(f"模型目录 {cfg.model_dir} 为空：请放入 *.gguf / *.bin（本项目不联网下载）")
            return 0
        budget = engine.mem.total_mb
        print(f"内存预算 {budget:.0f}MB（总内存 {cfg.memory_budget_mb or 'auto'}）\n")
        print(f"{'模型':<44}{'量化':<10}{'体积MB':>10}{'估算内存MB':>12}{'是否放得下':>10}")
        for m in models:
            fits = "是" if m["est_ram_mb"] <= budget else "否"
            print(f"{m['alias']:<44}{m['quant']:<10}{m['size_mb']:>10.1f}"
                  f"{m['est_ram_mb']:>12.1f}{fits:>10}")
        return 0

    if args.cmd == "chat":
        cfg = _cfg(args)
        engine = Engine(cfg)
        r = engine.chat(args.model, [{"role": "user", "content": args.message}],
                        max_tokens=args.max_tokens)
        print(r["choices"][0]["message"]["content"])
        print(f"\n[model={r['model']} quant={r['local']['quant']} "
              f"threads={r['local']['threads']} usage={r['usage']}]")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
