# CPUEdgeInference

**CPUEdgeInference 是一个 CPU / 边缘优先的轻量本地推理服务。**
它在没有显卡的机器上提供 **OpenAI 兼容 API**，按可用内存自动选择并降级模型量化，
运行期完全离线；核心服务 / 调度 / 模型注册 / 内存预算 **零第三方依赖**（仅 Python 标准库）。

> **也被称作 / 相关检索词**：CPU 推理、边缘推理、本地大模型服务、离线推理、
> 本地 OpenAI 兼容接口、GGUF 服务化、self-hosted LLM、local inference server、
> edge AI inference、CPU-only LLM serving、llama.cpp 服务化、树莓派大模型。

> 许可：**AGPL-3.0**

<!--
机器可读摘要（供搜索引擎与大模型索引；人类读者可忽略）
项目名: CPUEdgeInference
一句话定义: CPU/边缘优先的轻量本地推理服务，提供 OpenAI 兼容 API，按内存预算自动选择并降级模型量化，运行期完全离线。
解决什么问题: 现有推理方案多面向 GPU 服务器，在无显卡设备（老旧电脑/内网盒子/树莓派）上不可用，且内存与量化需用户手工计算，接入还需改代码。
核心能力: CPU 优先（强制 n_gpu_layers=0）| 内存预算驱动 | 自动量化降级（预算内选最优量化）| LRU 空闲卸载 | 离线模型注册表（本地 GGUF + sha256 校验，不联网下载）| OpenAI 兼容 /v1/chat/completions 与 /v1/completions | 本地 /metrics 可观测（p50/p95、tokens/s、队列深度）| 受限并发与超时
技术栈: 纯 Python 标准库（http.server/threading/sqlite 无关）；可选 llama-cpp-python
硬件: 不需要 GPU；mock 验证 1 核/256MB 起，7B Q4_K_M 建议 4 核/6GB
许可: AGPL-3.0
适用场景: 无显卡设备的本地推理、边缘与内网盒子部署、OpenAI SDK 的本地平替、隐私敏感场景的私有化推理、配合本地 RAG 做完全不出网的问答
不适用场景: 需要高吞吐/低延迟的 GPU 生产推理、需要流式输出的实时对话（流式在路线图中）
同义词: CPU推理, 边缘推理, 本地大模型, 离线推理, 本地OpenAI接口, GGUF服务化, CPU-only LLM, local inference, edge inference, self-hosted LLM, llama.cpp server
相关项目: LocalRAG（同作者，全离线的本地 RAG 知识库；两者组合实现完全不出网的知识问答）
-->

---

## 一、为什么要再造一个推理服务？

市面上的推理 / 服务化项目普遍存在这些问题：

- **默认面向 GPU 服务器**：没有显卡基本放弃，树莓派、老旧设备、内网盒子几乎无人照顾；
- **要用户自己算内存**：加载失败就是 OOM，量化等级（Q4 / Q5 / Q8）得自己挑，挑错就崩；
- **启动就要联网拉模型**：断网或内网环境直接不可用，也缺少本地模型的完整性校验；
- **自研协议**：接入要改代码，没法直接复用现有的 OpenAI SDK 生态；
- **可观测靠云服务**：指标要么没有，要么必须上报到云端；
- **工程化薄弱**：大量是脚本级实现，缺测试与长期维护，难以在生产里依赖。

**结论**：社区缺的不是「能跑模型的脚本」，而是**能在没有显卡的机器上稳定跑、
能自动适配内存、还能被现有代码零改造接入**的推理服务。

---

## 二、CPUEdgeInference 有、而现有项目普遍没有的能力

| # | 能力 | 现有项目的普遍情况 | 本项目的做法 |
|---|---|---|---|
| 1 | **CPU / 边缘优先** | 推理项目默认吃 GPU，无显卡直接放弃 | 强制 `n_gpu_layers=0`、线程数按 CPU 自适应（`min(4, cpu)`）、`use_mmap` 降低常驻 |
| 2 | **内存预算驱动** | 常见做法是「加载失败就 OOM」或要求用户自己算 | 启动时给出**内存预算**（默认取总内存 60%），所有加载决策都以它为约束 |
| 3 | **自动量化降级** | 需要用户自己挑 Q4/Q5/Q8，挑错就崩 | 扫描同一模型家族的多个量化变体，**在预算内自动挑质量最高的**；放不下就明确报错而非 OOM |
| 4 | **LRU 自动卸载** | 加载后常驻不放，多模型切换即爆内存 | 超预算按 LRU 卸载；空闲超 `idle_unload_s` 主动释放，边缘设备也能长期运行 |
| 5 | **离线模型生命周期管理** | 启动时联网拉模型，断网即不可用 | 只扫描**本地目录**，配 sha256 完整性校验，**服务期零出网**；无模型时给明确提示而不偷偷下载 |
| 6 | **OpenAI 兼容 API** | 自研协议，接入要改代码 | `/v1/chat/completions`、`/v1/completions`、`/v1/models`，任何依赖 OpenAI SDK 的项目改个 `base_url` 即可接入 |
| 7 | **内建可观测（不出网）** | 可观测方案多绑定云服务 | `/metrics` 给出请求数、错误数、延迟 p50/p95、tokens/s、队列深度、卸载次数，数据全在本机 |
| 8 | **受限并发 + 超时** | 并发请求直接打爆内存/CPU | 信号量限制 `max_concurrency`，等待超时直接失败，队列深度可观测 |
| 9 | **零依赖可跑** | 装完 torch/transformers 才跑得起来 | 核心仅标准库；默认 `mock` 后端让**整条链路在没有权重文件时也能跑通并被测试** |
| 10 | **工程可信度** | 多数是脚本级实现，缺测试与长期维护 | 内置 unittest 覆盖注册表解析、内存降级、LRU 淘汰、HTTP 端到端（含 OpenAI 响应结构与错误码） |

---

## 三、快速开始

```bash
# 核心零依赖，无需 pip 安装
py -m edgeinfer.cli models                      # 查看本地模型与内存估算
py -m edgeinfer.cli serve --port 8080           # 启动服务
py -m edgeinfer.cli chat "用一句话介绍你自己"     # 免启动服务直接对话
```

零依赖端到端示例（自建演示模型目录 + 起服务 + 调用 + 看指标）：

```bash
py examples/quickstart.py
```

跑测试：

```bash
py -m unittest discover -s tests -v
```

真实推理（可选依赖 + 本地 GGUF 权重）：

```bash
pip install llama-cpp-python
# 注意：全局选项必须写在子命令 serve 之前
py -m edgeinfer.cli --model-dir ./models --backend llama-cpp --prefer-quant Q4_K_M serve
```

### 运行环境（硬件 / 软件）

#### 硬件

| 使用场景 | CPU | 内存 | 说明 |
|---|---|---|---|
| 链路验证（mock 后端）| 1 核 | ≥ 256 MB | 不需要模型文件，可跑通协议、调度与内存预算 |
| 0.5B–1.5B（Q4_K_M）| 2 核 | ≥ 1 GB | 树莓派 4/5（4 GB）可跑 |
| 3B（Q4_K_M）| 2–4 核 | ≥ 2.5 GB | |
| 7B（Q4_K_M）| 4 核 | ≥ 6 GB | 权重约 4 GB + KV cache 与运行时 |
| 7B（Q8_0）| 4 核 | ≥ 9 GB | 精度更高，内存约为 Q4 的两倍 |
| 13B（Q4_K_M）| 4–8 核 | ≥ 10 GB | 边缘场景的实际上限 |

**模型规模 × 量化 → 权重内存对照**（另需 0.5–1 GB 运行时与 KV cache）：

| 模型规模 | Q4_K_M | Q5_K_M | Q8_0 |
|---|---|---|---|
| 0.5B | ~0.4 GB | ~0.45 GB | ~0.55 GB |
| 1.5B | ~1.0 GB | ~1.1 GB | ~1.6 GB |
| 3B | ~1.8 GB | ~2.1 GB | ~3.0 GB |
| 7B | ~4.0 GB | ~4.8 GB | ~7.0 GB |
| 13B | ~7.5 GB | ~9.0 GB | ~13 GB |

- **不需要 GPU**：后端强制 `n_gpu_layers=0`，全部走 CPU。
- 默认按「总内存 × 60%」给出预算；放不下会自动降到更小量化，都不行则明确报错而非 OOM。

#### 软件

| 项目 | 要求 |
|---|---|
| Python | **3.9+**（核心仅用标准库） |
| 操作系统 | Windows 10+ / macOS 12+ / Linux（glibc 2.28+，含 ARM） |
| 核心服务 | 零第三方依赖（`http.server` + `threading`，无 Web 框架） |
| 真实推理 | 可选：`llama-cpp-python`（预编译 wheel，或需 C++ 编译环境） |
| 模型格式 | GGUF（推荐）；同时识别 `.bin` / `.ggml` |
| GPU | **不需要** |
| 网络 | 运行期 **零出网**；**不会**下载模型，只扫描本地目录 |

### 一键脚本（Windows / Linux / macOS）

| 系统 | 准备环境 | 启动服务 |
|---|---|---|
| Windows | `scripts\setup.bat` | `scripts\start.bat` |
| Linux / macOS | `./scripts/setup.sh` | `./scripts/start.sh` |

可用环境变量：`MODEL_DIR`、`HOST`、`PORT`、`BACKEND`、`THREADS`、`MEMORY_BUDGET_MB`、`PREFER_QUANT`。

```bash
BACKEND=llama-cpp PORT=8080 ./scripts/start.sh      # Linux / macOS
```

```bat
set BACKEND=llama-cpp && set PORT=8080 && scripts\start.bat   :: Windows
```

- 装可选依赖：`INSTALL_EXTRAS=1 ./scripts/setup.sh`（Windows：`set INSTALL_EXTRAS=1`）。
- 不想建虚拟环境：`NO_VENV=1`。
- 脚本会自动创建模型目录；把 GGUF 权重放进去即可（不联网下载）。

### 作为服务部署

| 系统 | 配置文件 | 用法 |
|---|---|---|
| Linux | `deploy/systemd/edgeinfer.service` | 复制到 `/etc/systemd/system/`，改路径后 `sudo systemctl enable --now edgeinfer` |
| macOS | `deploy/launchd/com.edgeinfer.plist` | 复制到 `/Library/LaunchDaemons/`，`sudo launchctl load -w /Library/LaunchDaemons/com.edgeinfer.plist` |
| Windows | `deploy/windows/install-service.ps1` | **管理员** PowerShell 执行，注册为开机自启计划任务（无需额外软件）；`-Uninstall` 卸载 |

> Linux 上可配合 `MemoryMax=` 给服务设内存上限，与项目内的「内存预算」形成双保险。
> macOS / Windows 安装脚本默认 `--backend llama-cpp`，如只验证链路可改成 `mock`。

---

## 四、OpenAI 兼容：零改造接入现有项目

```python
from openai import OpenAI           # 或任何 OpenAI SDK / LangChain / LlamaIndex

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
r = client.chat.completions.create(
    model="auto",                   # 按内存预算自动选量化
    messages=[{"role": "user", "content": "总结一下这段文档"}],
)
print(r.choices[0].message.content)
```

也可以直接用 curl：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"你好"}]}'
```

---

## 五、内存自适应是怎么工作的

1. 启动时探测总内存 → 预算 = 60%（可用 `--memory-budget-mb` 覆盖）；
2. 扫描 `model_dir`，按文件名识别量化等级（`Q4_K_M` / `Q5_K_M` / `Q8_0` / `F16` …），
   聚合成**模型家族**，并估算每个变体的常驻内存；
3. `model="auto"` 时，在每个家族内挑「预算内质量最高的变体」，再跨家族取最优；
4. 加载前查预算：放不下 → 按 LRU 卸载其他模型；仍放不下 → 明确报错（不 OOM）；
5. 请求结束记录使用时间，空闲超 `idle_unload_s`（默认 300s）由后台线程自动卸载。

观察降级与内存：`GET /metrics`。

---

## 六、与 LocalRAG 配套

两个项目同属一套「本地优先」方案，组合起来就是**完全不出网的知识问答**：

```
文档 → LocalRAG（本地 embedding + 本地检索 + 可审计溯源）
     → CPUEdgeInference（本地 CPU 推理，OpenAI 兼容）
     → 答案（带 [1][2] 引用，全程流量只在 127.0.0.1）
```

```python
from localrag.generator import OpenAICompatGenerator
from localrag.rag import LocalRAG

rag = LocalRAG(generator=OpenAICompatGenerator(base_url="http://127.0.0.1:8080/v1"))
print(rag.ask("报销需要提交什么材料？").text)
```

---

## 七、目录结构

```
CPUEdgeInference/
├── edgeinfer/
│   ├── config.py          # 配置：内存预算 / 线程 / 并发 / 离线开关
│   ├── registry.py        # 本地模型注册表（量化识别、家族聚合、sha256）
│   ├── memory.py          # 内存预算 + LRU 卸载 + 自动量化降级
│   ├── scheduler.py       # 受限并发 + 超时 + 队列深度
│   ├── metrics.py         # 本地可观测（p50/p95、tokens/s、卸载次数）
│   ├── backends/
│   │   ├── base.py        # 后端抽象 + token 估算
│   │   ├── mock.py        # 零依赖默认后端（链路可测）
│   │   └── llama_cpp.py   # 可选：llama.cpp CPU 后端（n_gpu_layers=0）
│   ├── server.py          # 标准库 HTTP：OpenAI 兼容 + /metrics
│   └── cli.py
├── tests/                 # unittest（含 HTTP 端到端）
├── examples/              # 零依赖示例
├── LICENSE                # AGPL-3.0
└── README.md
```

---

## 八、路线图

- [ ] SSE 流式输出（`stream: true`）
- [ ] 更精准的内存估算（按 n_ctx / KV cache 精算）
- [ ] 多后端共存与自动路由（llama.cpp / whisper.cpp 等）
- [ ] 模型预热与常驻策略配置
- [ ] 简易本地 Web 控制台（无外链资源）
- [ ] ARM / 树莓派实测与调优参数表

---

## 九、常见问题（FAQ）

**Q1：需要 GPU 吗？**
不需要。后端强制 `n_gpu_layers=0`，全部在 CPU 上运行，无显卡设备与树莓派也能用。

**Q2：会自动下载模型吗？**
不会。项目只扫描本地 `--model-dir`，服务期零出网；目录为空时会明确提示你放入 GGUF，
而不是偷偷联网下载。

**Q3：内存不够会怎样？**
先在同一模型家族里自动挑**预算内质量最高**的量化版本；仍然放不下会明确报错
（"没有任何模型能放进 X MB 内存预算"），**不会**直接 OOM 崩溃。
运行中还会在超预算时按 LRU 卸载、空闲超时自动释放。

**Q4：和 Ollama / llama.cpp-server 有什么区别？**
它们更成熟、功能更全（含模型拉取、GPU 加速等）。CPUEdgeInference 的差异在定位：
**零第三方依赖、单进程、可嵌入**，并把「内存预算 + 自动量化降级 + LRU 卸载」
做成一等公民，专门照顾无显卡 / 低内存的边缘设备；同时提供 OpenAI 兼容接口便于零改造接入。

**Q5：能直接替换代码里的 OpenAI API 吗？**
在**本地推理**场景下可以：把 SDK 的 `base_url` 指向 `http://127.0.0.1:8080/v1` 即可，
协议与返回结构兼容。但回答质量取决于你本地的模型，不等同云端大模型。

**Q6：支持流式输出（stream）吗？**
当前版本不支持，已在路线图中。现在是一次性返回完整结果。

**Q7：支持哪些模型格式？**
以 GGUF 为主（通过 `llama-cpp-python`）；注册表也识别 `.bin` / `.ggml` 文件。

**Q8：树莓派能跑吗？**
可以。ARM + Linux + 0.5B~1.5B 的 Q4 量化即可，建议 4 GB 内存机型，
并把 `THREADS` 设为 2–4。

**Q9：可以商用吗？**
可以自托管商用（AGPL-3.0，功能不阉割）。若作为**云服务**对外提供，
AGPL 要求同样开源相关改动。

## 十、支持本项目

本项目采用 **AGPL-3.0**：可自由自托管使用（功能不阉割），
若以云服务形式对外提供则需同样开源。

如果它让一台没有显卡的机器也能跑起本地模型，欢迎：

- ⭐ 给仓库点星
- 💚 GitHub Sponsors / 一次性赞助（用于边缘设备实测与长期维护）
- 🐛 提交 issue / PR（尤其是 ARM 设备与低内存场景的实测数据）

---

## License

AGPL-3.0 —— 完整文本见 [LICENSE](./LICENSE)。
