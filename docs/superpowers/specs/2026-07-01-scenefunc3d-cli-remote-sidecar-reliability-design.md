# SceneFunc3D CLI 与远端 Sidecar 可靠性设计

日期：2026-07-01

## 目标

把 SceneFunc3D 当前的 agent runner 推进到可稳定运行到 Molmo/SAM/result
边界，同时保持 NR3D、OpenEQA、SceneFunc3D 三个 benchmark 的启动方式一致：

```bash
PYTHONPATH=src python -m codex_agent.cli.run_nr3d ...
PYTHONPATH=src python -m codex_agent.cli.run_openeqa ...
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d ...
```

这次修复只解决工程可运行性和失败可诊断性，不做 SceneFunc3D 质量优化、不调
prompt 策略、不引入额外可选启动分支。所有改动必须符合
`docs/python_code_agent_quality_guide.md`：强类型、边界校验、业务逻辑不放在
CLI `main`、不隐藏失败、不提交真实凭证。

## 当前问题

实测暴露了五个问题：

1. `python src/codex_agent/cli/run_scenefunc3d.py --help` 会因为相对 import
   失败；这个入口不应被支持，三个 benchmark 统一只支持 `python -m`。
2. `python -m codex_agent.cli.run_scenefunc3d --help` 可以运行，但 usage
   显示为 `codex_agent.scenefunc3d.runner`，说明 CLI wrapper 没有真正接管
   parser 显示名。
3. Codex provider adapter 未启动时，runner 会进入 Codex turn 后才报
   `stream disconnected before completion`；这类运行前置条件应在 benchmark
   CLI 启动阶段 fail fast。
4. SceneFunc3D 还暴露 `--skip-sidecar-health-check`，这和 benchmark 启动方式
   统一、失败显式化相冲突。
5. Agent 需要手工复制很长的 `out_dir`，一次运行中把
   `6afc3b66-ea3e-4b8e-b630-ca6174da502a` 写成
   `6afc3b66-ea3e-b630-ca6174da502a`，导致最终结果路径不在 sample output
   directory 内。

此外，远端 workspace-proxy sidecar 无法读取本机文件路径，也不能把远端 SAM
生成的 mask 文件路径作为本机结果返回；Molmo/SAM 输入输出协议必须支持远端
按值传输。

## 非目标

- 不支持直接运行 `python src/codex_agent/cli/run_*.py`。
- 不增加 `--preflight`、`--no-preflight`、`--skip-*` 这类让 benchmark 行为分叉
  的开关。
- 不提交真实 workspace-proxy header、Cookie、AK、内部 URL 实例值。
- 不改变 NR3D/OpenEQA 的任务语义、评分逻辑或默认 prompt。
- 不用远端 sidecar 的临时文件路径作为本地 durable artifact。

## 设计决策

### 1. 统一 benchmark 启动入口

三个 benchmark 的人类和脚本入口都统一为：

```text
codex_agent.cli.run_nr3d
codex_agent.cli.run_openeqa
codex_agent.cli.run_scenefunc3d
```

实现要求：

- CLI wrapper 的 `build_arg_parser()` 必须返回显示自身 `prog` 的 parser。
- `main()` 必须使用 wrapper 的 parser，不能直接调用下游 runner 的 `main()` 后
  丢失 `prog`。
- 直接文件路径执行不是支持场景，不为它新增 sys.path hack。
- benchmark 文档和脚本只展示 `python -m codex_agent.cli.run_*`。

SceneFunc3D runner 可以继续保留业务编排函数，但 parser 构造需要能接收调用方
传入的 `prog`，或把“解析 CLI 参数”和“执行 runner 配置”拆开。

### 2. 固定前置检查，不暴露跳过开关

SceneFunc3D 启动时必须检查：

- backend config 可读、可校验。
- Molmo `/health` 可访问且 `model_loaded=true`。
- SAM `/health` 可访问且 `model_loaded=true`。
- 如果 URL 为远端 HTTPS，健康检查必须携带私有 header 配置。

删除 SceneFunc3D 的 `--skip-sidecar-health-check` 参数。失败时直接退出，不进入
Codex turn，不写假成功。

NR3D、OpenEQA、SceneFunc3D 都通过一个共享的 Codex runtime preflight 路径构建
`CodexAgentRuntime`。这个 preflight 的职责是捕获本地 provider adapter 未启动
这类启动条件错误；例如配置指向 `http://127.0.0.1:8787/v1` 时，在发起正式
turn 前检查 adapter health 或至少检查 base URL 可连接。它不是通用鉴权探测，
不打印 token、AK 或完整 private header。

### 3. 私有远端配置仍放 gitignored 文件

保留两个本地实例配置：

```text
configs/scenefunc3d_backends.toml
configs/scenefunc3d_sidecar_headers.toml
```

它们由 `.gitignore` 排除。tracked 文件只保留 example：

```text
configs/scenefunc3d_backends.example.toml
configs/scenefunc3d_sidecar_headers.example.toml
```

`allowed_image_roots` 是本地允许读取并上传给 sidecar 的图片根目录；
`allowed_output_roots` 是本地 tool 允许写 artifact 的根目录。二者都由本地
client/tool 校验，不是远端 server 的访问控制。默认输出根目录应位于 repo
`tmp/` 下，避免和 `data/SceneFun3D` 混在一起。

### 4. 远端 Molmo/SAM 按值传输，不传远端文件路径

Sidecar client 根据 backend URL scheme 选择传输形态：

- `http://127.0.0.1` 或 `http://localhost`：允许继续传本机 `image_path`。
- `https://...`：必须读取本机图片，在 request 中传递图片内容。

远端图片输入使用强类型 payload：

```json
{
  "image": {
    "filename": "000050.jpg",
    "mime_type": "image/jpeg",
    "sha256": "<hex>",
    "data_base64": "<base64>"
  }
}
```

Client 在读取图片前先用 `allowed_image_roots` 校验路径；server 解码后校验
`sha256`，不信任 filename 或 MIME。

Molmo response 仍只返回 raw text、latency、model name，本地 tool 负责解析
point 和生成 overlay。

SAM response 不再依赖 sidecar 写出的 mask 文件路径。远端返回每个 candidate
的紧凑 mask 数据，本地 tool 统一落盘：

```json
{
  "candidate_id": "candidate_0",
  "score": 0.94,
  "height": 1080,
  "width": 1920,
  "mask_rle": {
    "encoding": "counts",
    "counts": [320, 12, 818, 44]
  }
}
```

RLE 只表达二值 mask，通常比 raw bool array 或 base64 NPZ 小，也避免远端临时
路径不可访问的问题。本地 `sam_mask` tool 将 RLE 还原为 numpy mask，写入
sample `out_dir` 下的 `.npz`、overlay 和 contact sheet。对于本地 sidecar，
也优先走同一 response contract；如果保留旧 `mask_npz_path`，只能作为本地
兼容输入在 client 边界立即转换成同一内部 candidate 类型。

### 5. Tool context 代替 agent 手工复制长路径

Runner 为每个 sample 创建一个不可变 tool context 文件：

```text
<sample_output_dir>/tool_context.json
```

内容由 Pydantic v2 模型校验：

```json
{
  "sample_id": "421254::6afc3b66-ea3e-4b8e-b630-ca6174da502a",
  "scene_root": "/abs/path/data/SceneFun3D/421254",
  "backend_config_path": "/abs/path/configs/scenefunc3d_backends.toml",
  "out_dir": "/abs/path/tmp/scenefunc3d/artifacts/.../6afc3b66-ea3e-4b8e-b630-ca6174da502a"
}
```

Agent prompt 中只出现一条 tool 调用格式：

```bash
python -m codex_agent.scenefunc3d.tools <tool> \
  --context <sample_output_dir>/tool_context.json \
  --args '<json>'
```

Tool CLI 从 context 读取 `scene_root`、`backend_config_path`、`out_dir`，并在每
次写文件前确认目标路径位于 context `out_dir` 内。Agent 不再手工复制
`scene_root`、`backend_config`、`out_dir` 三个长路径，从根上消除 UUID 漏段
导致的 artifact 分叉。

如果 context 文件缺失、被篡改、路径不在允许根目录内，tool 直接失败。不得
fallback 到默认 scratch 目录。

### 6. 失败也要有结构化 artifact

`run_single_sample()` 不把 Codex turn 失败伪装成成功。它需要在 sample output
dir 写入 `failure.json`，然后把异常继续抛给 CLI 或 batch 汇总层。

`failure.json` 至少包含：

```json
{
  "task_name": "scenefunc3d_mask_generation",
  "sample_id": "...",
  "status": "failed",
  "failure_stage": "codex_turn",
  "error_type": "CodexResponseError",
  "error_message": "...",
  "events_path": ".../events.jsonl",
  "tool_context_path": ".../tool_context.json",
  "turn": {
    "turn_id": null,
    "status": null,
    "duration_ms": null,
    "usage": null,
    "input_tokens": null,
    "cached_input_tokens": null,
    "reasoning_summary": null,
    "run_home": null,
    "attempts": []
  }
}
```

单 sample CLI 遇到失败返回非零；batch mode 在 `evaluation_summary.json` 中记
录该样本 `status=failed` 和 `failure_stage`。两种模式都保留 `events.jsonl`、
`tool_context.json` 和 `failure.json`，方便复盘。

### 7. 分层和类型约束

实现时按能力拆模块，不把逻辑堆在 CLI `main`：

```text
src/codex_agent/cli/runtime_preflight.py
src/codex_agent/scenefunc3d/tool_context.py
src/codex_agent/scenefunc3d/backends/image_payload.py
src/codex_agent/scenefunc3d/backends/mask_codec.py
src/codex_agent/scenefunc3d/failure_artifacts.py
```

约束：

- 配置文件、CLI 参数、HTTP payload、Codex final JSON 都在边界用 Pydantic 或
  argparse 校验。
- 不在业务代码散落读取环境变量；Codex runtime 继续通过集中 config 构建。
- 不使用裸 `dict` 在多层传递真实对象；使用 `dataclass(frozen=True)`、
  `TypedDict` 或 Pydantic model。
- 不使用 `Any`。反序列化边界如果必须接收 `object`，立即转换成强类型。
- 不吞异常；只能补充上下文后继续抛出，或在 CLI 边界转换成非零退出。

## 测试计划

新增或调整测试应覆盖：

1. `run_scenefunc3d --help` 在 `python -m` 下显示
   `codex_agent.cli.run_scenefunc3d`。
2. SceneFunc3D parser 不再包含 `--skip-sidecar-health-check`。
3. 三个 benchmark CLI 共享 runtime preflight；本地 adapter 未启动时，在进入
   Codex turn 前给出可定位错误。
4. 远端图片 payload 会校验本地 allowed roots、MIME、sha256；server 端 sha256
   不匹配时失败。
5. SAM RLE response 能被 client 还原，并由本地 tool 写入 sample `out_dir`。
6. Tool context 缺失、路径越权、`out_dir` 不在 allowed output roots 时失败。
7. Agent prompt 只包含 `--context` tool 调用格式，不再暴露三段长路径命令。
8. `run_single_sample()` 在 `CodexResponseError` 或 `CodexTurnError` 时写入
   `failure.json`，单样本 CLI 返回非零，batch summary 记录失败样本。

最小验证命令：

```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
PYTHONPATH=src pytest src/codex_agent/scenefunc3d -q
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d --help
```

远端 sidecar 冒烟命令使用 gitignored config，不在文档中写真实 header：

```bash
PYTHONPATH=src python -m codex_agent.cli.run_scenefunc3d \
  --dataset-root data/SceneFun3D \
  --sample-id '<sample_id>' \
  --backend-config configs/scenefunc3d_backends.toml \
  --output-dir tmp/scenefunc3d/artifacts/remote_smoke \
  --score
```

## 验收标准

- 三个 benchmark 都通过 `python -m codex_agent.cli.run_*` 启动，脚本和文档没有
  其它启动分支。
- SceneFunc3D 健康检查和 Codex provider preflight 都是固定前置条件，不能通过
  CLI 开关跳过。
- 远端 Molmo/SAM 不需要访问本机路径，也不把远端路径返回给本机作为 durable
  artifact。
- Agent tool 调用不再要求手抄 `out_dir`，所有 tool artifacts 都落在当前
  sample output dir。
- 失败样本有 `failure.json` 和 `events.jsonl`，CLI/batch 不报告假成功。
- 全量质量 gate 通过，或明确列出无法运行的命令及风险。
