# Codex Agent SDK 从零搭建与验证指南

本文档用于指导下一个 agent 从空目录开始，搭建一个可以通过 OpenAI Codex Python SDK 调用内部 AIDP ModelHub 模型的项目级配置。它同时记录本仓库已经验证过的行为、失败边界和后续排查入口。

本文档中的 AK 一律使用占位符，不要把真实 AK 写入仓库文件。真实 AK 只通过当前 shell 环境变量临时注入。

## 0. 已验证结论

本仓库当前路径：

```text
/Users/bytedance/aispace/codex_modelhub_adapter
```

已经验证通过的能力：

1. `uv` 项目环境可用，Python SDK 包为 `openai-codex`。
2. Codex 配置使用项目内 `.codex-home/config.toml`，启动时通过 `CODEX_HOME=$(pwd)/.codex-home` 指定，不依赖用户级 `~/.codex/config.toml`。
3. 本地 adapter 暴露 OpenAI-compatible Responses endpoint：

   ```text
   POST /v1/responses
   POST /v1/responses/compact
   GET  /health
   ```

4. 办公网使用内部 ModelHub 域名：

   ```text
   https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online
   ```

5. `gpt-5.4-2026-03-05` / `gpt-5.5-2026-04-24` 当前默认走 ModelHub Responses API；adapter 负责把本地 `/v1/responses` 请求转发到 ModelHub `/responses`，并默认保持 request body 不变。
6. Codex SDK 能启动 thread 并完成文件创建任务。
7. repo skill 可通过 `SkillInput` 显式注入。
8. 项目级 MCP 配置能被 `codex mcp list/get` 识别。
9. 当前 Python SDK 签名没有 LangChain 风格的 `tools=[python_func]` 注册入口。
10. 本次 trace 没有观察到自定义 MCP tool 被 SDK 直接作为 `server=heart_template tool=heart_template` 调用；Codex 最终通过 shell 直接调用了 MCP server 的 JSON-RPC。结论是：MCP 配置可识别，但 SDK 这条链路的自定义 MCP 直连调用还需要继续验证或升级 runtime。

已完成的端到端验证任务：

```text
Codex SDK -> ModelHub adapter -> AIDP office endpoint -> Codex runtime tool actions
```

任务结果包括：

```text
codex_sdk_smoke.txt
print_heart.py
skill_mcp_heart.py
```

`skill_mcp_heart.py` 输出已独立校验：

```text
  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *
```

## 1. 目标架构

目标不是直接让 Codex SDK 访问内部 ModelHub，而是在中间放一个本地 adapter：

```text
openai-codex Python SDK
  -> Codex app-server/runtime
  -> http://127.0.0.1:8787/v1/responses
  -> adapter.app FastAPI
  -> AIDP ModelHub office endpoint /api/modelhub/online/responses
  -> Responses response/SSE passthrough
  -> Codex runtime 继续执行 shell/apply_patch/MCP 等动作
```

为什么需要 adapter：

1. Codex Python SDK 期望访问 OpenAI-compatible Responses API。
2. 内部 AIDP ModelHub 当前支持 Responses API：

   ```bash
   curl --location --request POST \
     'https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses?ak=replace-with-ak' \
     --header 'Content-Type: application/json' \
     --header 'X-TT-LOGID: replace-with-logid' \
     --data '{
       "model": "gpt-5.4-2026-03-05",
       "input": [
         {
           "role": "user",
           "content": [
             {
               "type": "input_text",
               "text": "What is the result of 1+1?"
             }
           ]
         }
       ]
     }'
   ```

3. Codex runtime 会传入 Responses-style request、tools、stream event、state、compact request 等，adapter 不能把它降级成简单文本转发，也不能默认改写 body。
4. adapter 要处理：

   - `/v1/responses` -> ModelHub `/responses`
   - `/v1/responses/compact` -> ModelHub `/responses/compact`
   - Responses request/response/SSE passthrough
   - legacy Chat Completions fallback，仅在显式配置时启用
   - encrypted state fallback
   - 429 retry
   - AK pool sticky routing

## 2. 从空目录初始化项目

推荐目录名：

```bash
mkdir -p /Users/bytedance/aispace/codex_modelhub_adapter
cd /Users/bytedance/aispace/codex_modelhub_adapter
```

初始化 uv 环境：

```bash
uv init --bare
```

写入 `pyproject.toml`：

```toml
[project]
name = "codex-modelhub-adapter"
version = "0.1.0"
description = "Project-local Codex SDK adapter for the internal ModelHub Responses endpoint."
requires-python = ">=3.10,<3.13"
dependencies = [
    "fastapi>=0.115",
    "httpx>=0.27",
    "openai-codex",
    "tomli>=2; python_version < '3.11'",
    "uvicorn[standard]>=0.30",
]

[tool.uv]
package = false
```

创建 Python 3.12 虚拟环境：

```bash
uv sync --python 3.12
```

验证 SDK 可导入：

```bash
uv run python - <<'PY'
import inspect
from openai_codex import Codex, Thread, SkillInput, TextInput

print("Codex.thread_start", inspect.signature(Codex.thread_start))
print("Thread.run", inspect.signature(Thread.run))
print("Thread.turn", inspect.signature(Thread.turn))
print("SkillInput", inspect.signature(SkillInput))
print("TextInput", inspect.signature(TextInput))
PY
```

本仓库当前实测签名：

```text
Codex.thread_start(..., config=None, cwd=None, model=None, model_provider=None, sandbox=None, ...)
Thread.run(input, ..., cwd=None, model=None, output_schema=None, sandbox=None, ...)
Thread.turn(input, ..., cwd=None, model=None, output_schema=None, sandbox=None, ...)
SkillInput(name: str, path: str)
TextInput(text: str)
```

注意：没有 `tools=` 参数。

## 3. 项目级 Codex Home

用户明确要求不能使用本地用户级 `~/.codex/config.toml`。因此使用项目内目录：

```text
.codex-home/
```

关键点：

1. 所有运行 Codex SDK 的命令都要带：

   ```bash
   CODEX_HOME=$(pwd)/.codex-home
   ```

2. Python 脚本也可以兜底设置：

   ```python
   os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / ".codex-home"))
   ```

3. 不要把真实 AK 写进 `.codex-home/config.toml`。
4. 不要假设 repo 内 `.codex/config.toml` 一定能覆盖 provider。虽然官方 Codex 配置支持 project config，但本项目实测 provider 相关配置使用项目级 `CODEX_HOME` 更稳定。

创建 `.codex-home/config.toml`：

```toml
model = "gpt-5.4-2026-03-05"
model_provider = "modelhub_adapter"

[model_providers.modelhub_adapter]
name = "ModelHub local adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"

[projects."/Users/bytedance/aispace/codex_modelhub_adapter"]
trust_level = "trusted"
```

如果要启用本仓库的示例 MCP server，再加：

```toml
[mcp_servers.heart_template]
command = "python3"
args = ["/Users/bytedance/aispace/codex_modelhub_adapter/tools/heart_mcp_server.py"]
startup_timeout_sec = 10
tool_timeout_sec = 30
enabled = true
```

验证配置是否被 CLI 读取：

```bash
CODEX_HOME=$(pwd)/.codex-home codex mcp list
CODEX_HOME=$(pwd)/.codex-home codex mcp get heart_template
```

本仓库实测可看到：

```text
heart_template ... enabled
```

## 4. 环境变量配置

创建 `.env.example`：

```bash
AIDP_GPT_AK=replace-with-modelhub-ak

# Office network default. Set to online only outside the office network if needed.
AIDP_CODEX_PROXY_UPSTREAM_ENV=office

# Default path is native ModelHub Responses API passthrough.
AIDP_CODEX_PROXY_UPSTREAM_API=responses
AIDP_CODEX_PROXY_CHAT_COMPLETIONS_PATH=/v2/crawl
AIDP_CODEX_PROXY_RESPONSES_PATH=/responses

AIDP_CODEX_PROXY_MAX_OUTPUT_TOKENS=65536
AIDP_CODEX_PROXY_CHAT_CONTEXT_TOKEN_LIMIT=820000
AIDP_CODEX_PROXY_CHAT_CONTEXT_RETRY_TOKEN_LIMIT=700000
AIDP_CODEX_PROXY_CHAT_CONTEXT_CHARS_PER_TOKEN=2.8
AIDP_CODEX_PROXY_RESPONSES_BODY_MUTATION_ENABLED=false
AIDP_CODEX_PROXY_ENCRYPTED_STATE_FALLBACK_ENABLED=true
AIDP_CODEX_PROXY_TIMEOUT_SECONDS=300
AIDP_CODEX_PROXY_MAX_429_RETRIES=3
AIDP_CODEX_PROXY_SESSION_ID=case-reviewer-codex

# Optional TOML upstream pool with url/model_name/ak/weight entries.
# AIDP_MODELHUB_UPSTREAMS_TOML=.modelhub_upstreams.toml

# Optional sticky AK pool, selected by extra.session_id with rendezvous hash.
# AIDP_MODELHUB_AK_POOL=[{"alias":"k1","ak":"replace-ak-1"},{"alias":"k2","ak":"replace-ak-2"}]

# Backward-compatible legacy names still work:
# MODELHUB_AK=replace-with-modelhub-ak
# MODELHUB_URL=https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses

CODEX_PROMPT=Explain this repository in three bullets.
```

真实运行时复制并注入 AK：

```bash
cp .env.example .env
```

然后编辑 `.env`，写入真实 AK。`.env` 不应提交。

办公网必须使用：

```text
https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online
```

非办公网或线上环境可以使用：

```text
https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online
```

当前 adapter 默认 `AIDP_CODEX_PROXY_UPSTREAM_ENV=office`，也就是办公网域名。

## 5. Adapter 文件结构

建议从零创建如下结构：

```text
adapter/
  __init__.py
  app.py
  mapping.py
  proxy.py
examples/
  run_codex_sdk.py
  run_skill_tool_trace.py
tests/
  test_app.py
  test_mapping.py
tools/
  heart_mcp_server.py
.agents/
  skills/
    heart-task/
      SKILL.md
.codex-home/
  config.toml
pyproject.toml
.env.example
README.md
```

各文件职责：

```text
adapter/proxy.py
  读取环境变量，解析 office/online base URL，选择 upstream API，
  生成 ModelHub 请求 URL、headers、body，处理 AK pool。

adapter/mapping.py
  负责 Responses API 和 Chat Completions/crawl API 之间的结构转换。

adapter/app.py
  FastAPI 入口，暴露 /health、/v1/responses、/v1/responses/compact，
  并实现 retry、stream proxy、错误透传。

examples/run_codex_sdk.py
  最小 Codex SDK smoke 脚本。

examples/run_skill_tool_trace.py
  带 SkillInput 和 trace 输出的验证脚本。

tools/heart_mcp_server.py
  极简 stdio MCP server，用于验证项目级 MCP 配置。

.agents/skills/heart-task/SKILL.md
  repo skill，用于验证 skill 注入。
```

## 6. proxy.py 设计要点

`AdapterSettings.from_env()` 要覆盖这些配置：

```text
AIDP_CODEX_PROXY_ONLINE_BASE_URL
AIDP_CODEX_PROXY_OFFICE_BASE_URL
AIDP_BASE_URL
AIDP_CODEX_PROXY_UPSTREAM_ENV
AIDP_CODEX_PROXY_UPSTREAM_API
AIDP_CODEX_PROXY_RESPONSES_PATH
AIDP_CODEX_PROXY_CHAT_COMPLETIONS_PATH
AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS
AIDP_CODEX_PROXY_MAX_OUTPUT_TOKENS
AIDP_CODEX_PROXY_CHAT_CONTEXT_TOKEN_LIMIT
AIDP_CODEX_PROXY_CHAT_CONTEXT_RETRY_TOKEN_LIMIT
AIDP_CODEX_PROXY_CHAT_CONTEXT_CHARS_PER_TOKEN
AIDP_CODEX_PROXY_COMPACT_SUMMARY_MAX_CHARS
AIDP_CODEX_PROXY_RESPONSES_BODY_MUTATION_ENABLED
AIDP_CODEX_PROXY_ENCRYPTED_STATE_FALLBACK_ENABLED
AIDP_CODEX_PROXY_TIMEOUT_SECONDS
AIDP_CODEX_PROXY_MAX_429_RETRIES
AIDP_CODEX_PROXY_SESSION_ID
AIDP_GPT_AK
AIDP_MODELHUB_AK
MODELHUB_AK
CASE_REVIEW_LLM_AK
AIDP_MODELHUB_UPSTREAMS_TOML
AIDP_MODELHUB_AK_POOL
```

核心默认值：

```python
online_base_url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"
office_base_url = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
upstream_env = "office"
upstream_api = "auto"
responses_path = "/responses"
chat_completions_path = "/v2/crawl"
chat_completions_models = ("gpt-5.4*", "gpt-5.5*")  # legacy chat mode only
max_output_tokens = 65536
responses_body_mutation_enabled = False
```

`resolve_upstream_api()` 逻辑：

```text
if AIDP_CODEX_PROXY_UPSTREAM_API is "responses":
  use /responses
elif it is "chat_completions":
  use /v2/crawl
else:
  use /responses  # auto is responses-first
```

`build_upstream_request()` 需要生成：

```text
url = {base_url}{path}?ak={urlencoded_ak}
headers = {
  "content-type": "application/json",
  "X-TT-LOGID": generated-or-forwarded-logid,
  "extra": json.dumps({"session_id": ...})
}
body = original Responses request body by default
```

AK 选择规则：

1. 如果配置了 `AIDP_MODELHUB_UPSTREAMS_TOML`，读取 `[[upstreams]]` 数组，每个元素包含 `url`、`model_name`、`ak`、`weight`。
2. TOML pool 先按请求 `model` 匹配 `model_name`，再用 `extra.session_id` 做 deterministic weighted hash，保证同一个 session 粘到同一个 target。
3. 如果 TOML 存在但没有匹配模型，必须 fail closed，不能回退到其他 AK。
4. 如果没有 TOML pool，但配置了 `AIDP_MODELHUB_AK_POOL`，用 `extra.session_id` 做 rendezvous hash，保证同一个 session 粘到同一个 AK。
5. 如果没有 pool，回退到单个 `AIDP_GPT_AK`。
6. 如果没有 AK，`/v1/responses` 应返回 503。

## 7. mapping.py 设计要点

### 7.1 Responses input -> messages

Codex SDK 可能传入：

```json
{
  "model": "gpt-5.4-2026-03-05",
  "input": "What is 1+1?",
  "stream": false
}
```

需要转换成：

```json
{
  "model": "gpt-5.4-2026-03-05",
  "messages": [
    {"role": "user", "content": "What is 1+1?"}
  ],
  "stream": false,
  "max_tokens": 65536
}
```

Responses `instructions` 应作为 system message 注入。

Responses content item：

```json
{"type": "input_text", "text": "..."}
```

要转换为 Chat Completions 可接受的 text content。

### 7.2 tools 转换

Responses tool：

```json
{
  "type": "function",
  "name": "shell",
  "description": "Run a shell command.",
  "parameters": {
    "type": "object",
    "properties": {"cmd": {"type": "string"}},
    "required": ["cmd"]
  }
}
```

要转为 Chat Completions tool：

```json
{
  "type": "function",
  "function": {
    "name": "shell",
    "description": "Run a shell command.",
    "parameters": {
      "type": "object",
      "properties": {"cmd": {"type": "string"}},
      "required": ["cmd"]
    }
  }
}
```

### 7.3 function call 配对

Codex runtime 会把历史工具调用放进 Responses input：

```json
{"type": "function_call", "call_id": "call_keep", "name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}
{"type": "function_call_output", "call_id": "call_keep", "output": "/tmp/project"}
```

Chat Completions 需要配对消息：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_keep",
      "type": "function",
      "function": {"name": "shell", "arguments": "{\"cmd\":\"pwd\"}"}
    }
  ]
}
{"role": "tool", "tool_call_id": "call_keep", "content": "/tmp/project"}
```

如果存在未配对的 function call，当前实现会丢弃未配对项，避免 upstream Chat Completions 报错。

### 7.4 Chat Completions response -> Responses response

输入：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "2"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 1,
    "total_tokens": 11
  }
}
```

输出：

```json
{
  "object": "response",
  "status": "completed",
  "output_text": "2",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 1,
    "total_tokens": 11
  }
}
```

### 7.5 SSE stream 转换

Codex SDK streaming 需要完整 Responses lifecycle。Chat Completions delta：

```text
data: {"choices":[{"delta":{"content":"he"}}]}
data: {"choices":[{"delta":{"content":"llo"}}]}
data: [DONE]
```

要转换为 Responses events，至少包括：

```text
response.created
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
data: [DONE]
```

否则 SDK 的 `turn.stream()` 可能无法形成完整 turn。

### 7.6 compact endpoint

Codex runtime 会调用 compaction。adapter 要支持：

```text
POST /v1/responses/compact
```

本仓库实现为本地 lossy checkpoint summary，返回：

```json
{
  "object": "response.compaction",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "CONTEXT CHECKPOINT SUMMARY ..."}
      ]
    }
  ]
}
```

## 8. app.py 设计要点

FastAPI app 需要有：

```python
app = FastAPI(title="Codex ModelHub Adapter")
```

Health endpoint：

```text
GET /health
```

返回内容要能看出：

```text
status
service
config.upstream_base_url
config.upstream_env
config.upstream_api
config.has_upstream_ak
```

Responses endpoint：

```text
POST /v1/responses
```

处理流程：

1. 解析 request JSON。
2. 从环境变量生成 `AdapterSettings`。
3. 解析客户端 `extra` header，生成 upstream `extra`。
4. 调用 `build_upstream_request()`。
5. 通过 `httpx.AsyncClient.stream()` 调 ModelHub。
6. 对 429 做指数退避 retry。
7. 对 context length exceeded 做更强裁剪后 retry。
8. 对 invalid encrypted content 做 sanitize 后 retry。
9. 根据 upstream API 类型决定返回：

   - chat_completions + stream -> `iter_chat_sse_as_responses`
   - chat_completions + non-stream -> `chat_completion_to_response`
   - responses + stream -> 透传
   - responses + non-stream -> 透传

Compact endpoint：

```text
POST /v1/responses/compact
```

默认代理到 ModelHub `/responses/compact`。只有显式
`AIDP_CODEX_PROXY_UPSTREAM_API=chat_completions` 的 legacy 模式才返回本地
`build_compaction_response()`。

## 9. 最小 Codex SDK 脚本

创建 `examples/run_codex_sdk.py`：

```python
from __future__ import annotations

import os
from pathlib import Path

from openai_codex import Codex, Sandbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / ".codex-home"))


def main() -> None:
    prompt = os.environ.get(
        "CODEX_PROMPT",
        "Explain this repository in three bullets.",
    )

    with Codex() as codex:
        thread = codex.thread_start(
            model="gpt-5.4-2026-03-05",
            sandbox=Sandbox.workspace_write,
        )
        result = thread.run(prompt)
        print(result.final_response)


if __name__ == "__main__":
    main()
```

运行前先启动 adapter，然后在另一个 terminal 执行：

```bash
CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_codex_sdk.py
```

覆盖 prompt：

```bash
CODEX_HOME=$(pwd)/.codex-home \
CODEX_PROMPT="创建 codex_sdk_smoke.txt，内容必须正好是 CODEX_MODELHUB_ADAPTER_OK" \
uv run python examples/run_codex_sdk.py
```

验证：

```bash
cat codex_sdk_smoke.txt
```

期望：

```text
CODEX_MODELHUB_ADAPTER_OK
```

## 10. Skill 配置

Codex skill 是一个包含 `SKILL.md` 的目录。repo scope 位置：

```text
.agents/skills/<skill-name>/SKILL.md
```

创建示例 skill：

```text
.agents/skills/heart-task/SKILL.md
```

内容：

```md
---
name: heart-task
description: Use when validating Codex SDK skill and MCP tool integration with a Python heart-printing task.
---

For validation tasks:

1. Use the MCP tool `heart_template` to get the exact heart output.
2. Create the requested Python file so running it prints exactly that output.
3. Run the file with `python3`.
4. Report whether the MCP tool was used and include exact stdout.
```

Python SDK 显式注入 skill：

```python
from openai_codex import SkillInput, TextInput

turn = thread.turn(
    [
        SkillInput(name="heart-task", path=str(SKILL_PATH)),
        TextInput("Use the heart-task skill. Create a file and run it."),
    ],
    cwd=str(PROJECT_ROOT),
)
```

注意：

1. Skill 不是 tool。
2. Skill 是工作流说明和上下文包。
3. Skill 可以指导 Codex 使用 shell、MCP、文件编辑等能力。
4. Skill 本身不等价于 Python function。

## 11. MCP 配置

MCP 是 Codex 推荐的工具扩展方式。它适合模型在回合中自主调用外部工具。

本仓库创建了一个极简 stdio MCP server：

```text
tools/heart_mcp_server.py
```

它支持：

```text
initialize
notifications/initialized
tools/list
tools/call
```

暴露工具：

```text
heart_template
```

返回内容格式：

```text
TRACE_LABEL=codex-sdk-skill-tool-test
HEART_OUTPUT_START
  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *
HEART_OUTPUT_END
```

`.codex-home/config.toml` 中配置：

```toml
[mcp_servers.heart_template]
command = "python3"
args = ["/Users/bytedance/aispace/codex_modelhub_adapter/tools/heart_mcp_server.py"]
startup_timeout_sec = 10
tool_timeout_sec = 30
enabled = true
```

验证 MCP server 进程协议最小可用：

```bash
python3 - <<'PY' | python3 tools/heart_mcp_server.py
import json, sys
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05"},
}
data = json.dumps(payload, separators=(",", ":"))
sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n{data}")
PY
```

期望返回类似：

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"heart-template-mcp","version":"0.1.0"}}}
```

验证工具调用：

```bash
python3 - <<'PY' | python3 tools/heart_mcp_server.py
import json, sys
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "heart_template",
        "arguments": {"label": "manual-smoke"},
    },
}
data = json.dumps(payload, separators=(",", ":"))
sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n{data}")
PY
```

验证 Codex 配置识别：

```bash
CODEX_HOME=$(pwd)/.codex-home codex mcp list
CODEX_HOME=$(pwd)/.codex-home codex mcp get heart_template
```

当前实测结论：

1. `codex mcp list/get` 能识别 `heart_template` server。
2. SDK trace 中没有看到自定义 MCP tool 以 `server=heart_template tool=heart_template` 被直接调用。
3. trace 中只看到 Codex 内置 MCP 资源接口：

   ```text
   mcpToolCall server=codex tool=list_mcp_resource_templates status=completed
   mcpToolCall server=codex tool=list_mcp_resources status=completed
   ```

4. Codex 最终通过 shell 手动调用了 `tools/heart_mcp_server.py`。

因此后续 agent 不应宣称“自定义 MCP tool 直连已完全跑通”。准确说法是：

```text
项目级 MCP 配置可被 Codex 识别；本次 SDK run 没有直接暴露自定义 MCP tool call，仍需继续排查 SDK/runtime 对自定义 MCP tools 的暴露条件。
```

## 12. Python function tool 是否可替代 MCP

当前 `openai-codex` Python SDK 不支持 LangChain 风格：

```python
Agent(tools=[python_func])
```

本仓库实测 SDK 签名没有 `tools=` 参数：

```text
Codex.thread_start(...): no tools
Thread.run(input, ...): no tools
Thread.turn(input, ...): no tools
```

Python function 可以作为应用层编排：

```python
def heart_template() -> str:
    return """  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *"""

with Codex() as codex:
    thread = codex.thread_start(cwd=".")
    result = thread.run(
        "Create print_heart.py that prints exactly this:\n"
        + heart_template()
    )
```

这种方式可用，但不是 agent tool call。模型不会在 turn 中自主调用 Python 函数。

如果需要模型在回合中自主选择工具，优先方案仍是：

```text
Python function -> MCP stdio server -> config.toml -> Codex runtime
```

## 13. Trace 脚本

`examples/run_skill_tool_trace.py` 用于观察 SDK turn 轨迹。

关键点：

1. 使用 `thread.turn()` 而不是 `thread.run()`，因为 `turn.stream()` 可以看到 item completed events。
2. 输入包含：

   ```python
   SkillInput(name="heart-task", path=str(SKILL_PATH))
   TextInput(prompt)
   ```

3. 监听事件类型：

   ```python
   ItemCompletedNotification
   ThreadTokenUsageUpdatedNotification
   TurnCompletedNotification
   ```

4. 对 item 做分类打印：

   ```text
   mcpToolCall
   commandExecution
   agentMessage
   fileChange
   userMessage
   ```

运行：

```bash
CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_skill_tool_trace.py
```

本仓库实测最终状态：

```text
TRACE final_status=completed
```

实测 item 类型序列：

```json
[
  "userMessage",
  "agentMessage",
  "mcpToolCall",
  "mcpToolCall",
  "commandExecution",
  "agentMessage",
  "commandExecution",
  "commandExecution",
  "agentMessage",
  "commandExecution",
  "commandExecution",
  "commandExecution",
  "agentMessage",
  "commandExecution",
  "agentMessage",
  "commandExecution",
  "agentMessage",
  "fileChange",
  "agentMessage",
  "commandExecution",
  "agentMessage"
]
```

关键观察：

```text
mcpToolCall server=codex tool=list_mcp_resource_templates status=completed
mcpToolCall server=codex tool=list_mcp_resources status=completed
commandExecution ... read .agents/skills/heart-task/SKILL.md
commandExecution ... read tools/heart_mcp_server.py
commandExecution ... manual JSON-RPC call to tools/heart_mcp_server.py
fileChange ... created skill_mcp_heart.py
commandExecution ... python3 skill_mcp_heart.py exit_code=0
```

## 14. 启动和验证顺序

### 14.1 编译检查

```bash
uv run python -m compileall adapter tools examples .agents
```

### 14.2 单元测试

```bash
uv run python -m unittest discover -s tests
```

本仓库测试覆盖：

```text
tests/test_mapping.py
  Responses input -> messages
  Responses tools -> Chat tools
  Chat response -> Responses response
  SSE lifecycle
  function call pair preservation
  encrypted state sanitize
  compact response
  AK pool sticky selection

tests/test_app.py
  /v1/responses route
  office endpoint routing
  /v1/responses/compact route

tests/test_print_heart.py
  print_heart.py exact stdout
```

### 14.3 启动 adapter

不要把真实 AK 写进命令历史的共享文档。实际运行时可以在当前 shell 临时 export：

```bash
export AIDP_GPT_AK='replace-with-real-ak'
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
export AIDP_CODEX_PROXY_UPSTREAM_API=responses

uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787
```

多 AK/多 target 推荐用私有 TOML：

```toml
[[upstreams]]
alias = "gpt54_a"
url = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-real-ak-1"
weight = 5

[[upstreams]]
alias = "gpt54_b"
url = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-real-ak-2"
weight = 1

[[upstreams]]
alias = "gpt54_c"
url = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-real-ak-3"
weight = 5
```

启动前设置：

```bash
export AIDP_MODELHUB_UPSTREAMS_TOML=.modelhub_upstreams.toml
```

如果使用 `.env`：

```bash
set -a
source .env
set +a
uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787
```

### 14.4 health check

另开 terminal：

```bash
curl -sS --max-time 10 http://127.0.0.1:8787/health
```

期望：

```json
{
  "status": "healthy",
  "service": "codex-modelhub-adapter",
  "config": {
    "upstream_env": "office",
    "upstream_base_url": "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online",
    "has_upstream_ak": true
  }
}
```

### 14.5 直接 Responses smoke

```bash
curl --request POST 'http://127.0.0.1:8787/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-5.4-2026-03-05",
    "input": "What is the result of 1+1?",
    "stream": false
  }'
```

期望 `output_text` 包含：

```text
2
```

### 14.6 Codex SDK smoke

```bash
CODEX_HOME=$(pwd)/.codex-home \
CODEX_PROMPT="Create codex_sdk_smoke.txt containing exactly CODEX_MODELHUB_ADAPTER_OK. Then report what you wrote." \
uv run python examples/run_codex_sdk.py
```

验证：

```bash
cat codex_sdk_smoke.txt
```

期望：

```text
CODEX_MODELHUB_ADAPTER_OK
```

### 14.7 爱心打印任务

```bash
CODEX_HOME=$(pwd)/.codex-home \
CODEX_PROMPT="Create print_heart.py. It must print an ASCII heart. Run python3 print_heart.py and report the exact output." \
uv run python examples/run_codex_sdk.py
```

验证：

```bash
python3 print_heart.py
```

期望：

```text
  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *
```

### 14.8 Skill + MCP trace

```bash
CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_skill_tool_trace.py
```

然后独立验证输出：

```bash
python3 - <<'PY'
import subprocess

expected = """  **   **
 **** ****
***********
 *********
  *******
   *****
    ***
     *
"""

proc = subprocess.run(
    ["python3", "skill_mcp_heart.py"],
    text=True,
    capture_output=True,
)

print({
    "stdout_matches": proc.stdout == expected,
    "returncode": proc.returncode,
    "stderr": proc.stderr,
})
print(proc.stdout)
PY
```

本仓库实测：

```text
{'stdout_matches': True, 'returncode': 0, 'stderr': ''}
```

### 14.9 停止 adapter

如果 adapter 是前台启动，直接 `Ctrl-C`。

确认端口没有残留：

```bash
lsof -nP -iTCP:8787 -sTCP:LISTEN
```

无输出表示没有监听。

## 15. 常见问题和排查

### 15.1 `/health` 是 degraded

原因通常是没有 AK。

检查：

```bash
env | rg 'AIDP_MODELHUB_UPSTREAMS_TOML|AIDP_GPT_AK|AIDP_MODELHUB_AK|MODELHUB_AK|CASE_REVIEW_LLM_AK'
```

修复：

```bash
export AIDP_GPT_AK='replace-with-real-ak'
```

不要把真实 AK 写入文档、README 或 git tracked 文件。

### 15.2 请求打到了 byteintl 而不是 tiktok-row

办公网必须使用：

```text
aidp-i18ntt-sg.tiktok-row.net
```

检查：

```bash
curl -sS http://127.0.0.1:8787/health | python3 -m json.tool
```

确认：

```text
upstream_env = office
upstream_base_url = https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online
```

修复：

```bash
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
```

或者显式覆盖：

```bash
export AIDP_BASE_URL=https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online
```

### 15.3 Codex SDK 没有走本地 adapter

检查是否设置了项目级 `CODEX_HOME`：

```bash
echo "$CODEX_HOME"
```

运行必须类似：

```bash
CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_codex_sdk.py
```

检查 `.codex-home/config.toml`：

```toml
model_provider = "modelhub_adapter"

[model_providers.modelhub_adapter]
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
```

### 15.4 adapter 收到请求但上游报 context length

adapter 已有 retry：

```text
AIDP_CODEX_PROXY_CHAT_CONTEXT_TOKEN_LIMIT=820000
AIDP_CODEX_PROXY_CHAT_CONTEXT_RETRY_TOKEN_LIMIT=700000
AIDP_CODEX_PROXY_CHAT_CONTEXT_CHARS_PER_TOKEN=2.8
```

如果仍然失败：

1. 降低 retry token limit。
2. 检查是否有大文件内容或历史 tool output 被塞进 input。
3. 检查 `/v1/responses/compact` 是否被正常调用。

### 15.5 invalid encrypted content

Codex runtime 可能带上 opaque encrypted state。内部 upstream 不一定接受。

adapter 当前支持：

```text
AIDP_CODEX_PROXY_ENCRYPTED_STATE_FALLBACK_ENABLED=true
```

它会移除：

```text
previous_response_id
encrypted_content
空 reasoning item
```

然后 retry。

### 15.6 模型只返回文本，不执行工具

Codex coding-agent 行为依赖 upstream 模型能否按兼容格式返回 tool calls。

如果 upstream 只返回文本：

1. 简单问答可以成功。
2. 文件编辑、shell、apply_patch 等 agent 行为可能失败或退化。
3. 需要检查 adapter 是否正确转换了 tools。
4. 需要观察 trace 中是否出现 `commandExecution`、`fileChange`、`mcpToolCall`。

### 15.7 Python function tool 不能注册

这是当前 SDK 能力边界，不是 adapter bug。

已验证签名：

```text
thread_start: no tools
run: no tools
turn: no tools
```

替代方案：

1. 前置 Python function，把结果拼进 prompt。
2. 后置 Python function，处理 Codex 输出。
3. 把 Python function 包成 MCP server。

### 15.8 MCP server 被 CLI 识别但 SDK trace 不直接调用

当前实测就是这个状态。

排查方向：

1. 使用 `CODEX_HOME=$(pwd)/.codex-home codex mcp list` 确认配置层。
2. 使用 `CODEX_HOME=$(pwd)/.codex-home codex mcp get heart_template` 确认 server 启用。
3. 在 Codex CLI TUI 中用 `/mcp` 看工具是否出现在 runtime。
4. 检查 SDK app-server 是否使用同一个 `CODEX_HOME`。
5. 检查 SDK/runtime 版本是否支持自定义 MCP tools 注入到 Python SDK thread。
6. 检查模型是否收到 MCP tools schema。
7. 如果 trace 只有 `server=codex tool=list_mcp_resources`，说明至少本次回合没有直接暴露自定义 MCP tool。

## 16. 下一个 agent 的建议执行清单

从零构建时按这个顺序做：

1. 建目录并 `uv sync --python 3.12`。
2. 写 `.codex-home/config.toml`，确认 `CODEX_HOME` 指向项目目录。
3. 写 `.env.example`，真实 AK 只放 `.env` 或 shell export。
4. 实现 `adapter/proxy.py`，先保证 URL、AK、office endpoint 和 body route 正确。
5. 实现 `adapter/mapping.py`，先做非 streaming text，再补 tools、function call pair、SSE、compact、sanitize。
6. 实现 `adapter/app.py`。
7. 写 `tests/test_mapping.py` 和 `tests/test_app.py`。
8. `uv run python -m unittest discover -s tests`。
9. 启动 adapter。
10. `/health`。
11. 直接 `/v1/responses` 问 `1+1`。
12. 跑 `examples/run_codex_sdk.py` 创建 `codex_sdk_smoke.txt`。
13. 跑爱心打印任务。
14. 创建 skill 和 MCP server。
15. 跑 `examples/run_skill_tool_trace.py`。
16. 如实记录 trace，不要把“CLI 识别 MCP server”等同于“SDK 原生调用自定义 MCP tool”。

## 17. 当前仓库关键文件索引

```text
README.md
  项目简介、安装、运行、基础验证。

pyproject.toml
  uv Python 项目和依赖。

.env.example
  环境变量模板。

.codex-home/config.toml
  项目级 Codex home 配置。

adapter/app.py
  FastAPI endpoint 和 upstream proxy。

adapter/proxy.py
  配置解析、AK、URL、upstream request 构建。

adapter/mapping.py
  Responses <-> Chat Completions/crawl 转换。

examples/run_codex_sdk.py
  最小 SDK smoke。

examples/run_skill_tool_trace.py
  Skill + MCP trace。

.agents/skills/heart-task/SKILL.md
  示例 repo skill。

tools/heart_mcp_server.py
  示例 MCP stdio server。

tests/test_app.py
tests/test_mapping.py
tests/test_print_heart.py
  回归测试。
```

## 18. 准确表述模板

后续汇报时建议使用下面这些准确说法：

```text
已跑通 Codex Python SDK 通过本地 Responses adapter 调用内部 ModelHub，并完成文件创建和执行任务。
```

```text
项目级 CODEX_HOME 生效，避免使用用户级 ~/.codex/config.toml。
```

```text
办公网 endpoint 使用 https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online。
```

```text
SkillInput 可用，repo skill 已被注入并遵循。
```

```text
MCP server 配置可被 codex mcp list/get 识别；但本次 SDK trace 未观察到自定义 MCP tool 直接作为 mcpToolCall 执行。
```

```text
当前 openai-codex Python SDK 没有 LangChain 式 Python function tool 注册入口；Python function 只能作为应用层前后置编排，若要 agent 自主调用，应包装成 MCP tool。
```

## 19. 不要做的事

1. 不要提交真实 AK。
2. 不要修改用户级 `~/.codex/config.toml` 来验证本项目。
3. 不要把 project `.codex/config.toml` 当成唯一 provider 配置来源；本项目使用 `.codex-home/config.toml`。
4. 不要只测 direct curl 就宣称 Codex SDK 跑通；必须跑 SDK 脚本。
5. 不要只看到 `codex mcp list` 就宣称 MCP tool 被模型直接调用；必须看 trace 中的 `mcpToolCall server=<custom-server> tool=<custom-tool>`。
6. 不要把 Skill 当成 Tool。Skill 是说明和资源包，Tool 是 runtime 可调用动作。
7. 不要把 Python function 当成 SDK tool。当前 SDK 没有这种注册入口。

## 20. 最小成功标准

一个从零搭建的版本，至少要满足：

```text
uv run python -m unittest discover -s tests
  pass

curl http://127.0.0.1:8787/health
  status healthy
  upstream_base_url is tiktok-row office URL
  has_upstream_ak true

curl http://127.0.0.1:8787/v1/responses
  output_text answers 1+1

CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_codex_sdk.py
  creates or edits a file as requested

python3 generated_file.py
  produces expected stdout
```

扩展成功标准：

```text
SkillInput visible in trace
custom MCP server visible in codex mcp list/get
custom MCP tool directly appears in SDK trace as mcpToolCall
```

其中最后一条在当前实测中尚未达成。
