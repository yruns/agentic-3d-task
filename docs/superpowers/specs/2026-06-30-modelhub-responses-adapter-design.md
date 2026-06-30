# ModelHub Responses Adapter 简化设计

日期：2026-06-30

## 目标

将当前 `codex_modelhub_adapter` 从“Responses 到 Chat Completions 的协议转换器”
收敛为一个更窄的网关：

```text
Codex / OpenAI SDK
  -> localhost /v1/responses
  -> ModelHub /api/modelhub/online/responses
```

ModelHub 当前已经支持 Responses API，因此 adapter 的默认职责不再是重写请求
语义，而是把本地 OpenAI-compatible Responses 请求转发到 ModelHub 的真实
Responses endpoint，同时保留 ModelHub 认证、办公网域名切换、AK 池、sticky
routing、重试、日志脱敏和健康检查。

这次改动的核心验收目标不是“接口能通”，而是“NR3D 效果不能退回到历史 chat
转换路径”。历史记录显示，chat path 会结构性丢失 Responses API 中的 reasoning
state，导致 cache ratio 和 NR3D 准确率明显下降。因此新实现必须保持 v5
`/responses` 基线的结果特征。

## 背景

当前 adapter 最初承担了两个不同职责：

1. 让 Codex SDK 的 OpenAI-compatible 请求可以访问 ModelHub。
2. 在 ModelHub 尚未完整支持 Responses API 时，把部分 Responses 请求转换为
   Chat Completions `/v2/crawl` 请求。

第二个职责现在应该退出默认路径。Responses API 与 Chat Completions API 不等价：

- Responses `input` 中可以携带 `type: reasoning` item。
- Responses 可以通过 `encrypted_content`、`previous_response_id` 和 `store`
  维持跨轮 reasoning carryover。
- Chat Completions `messages` 没有等价槽位承载这些 reasoning item。
- 历史 NR3D 记录显示，chat path 虽然仍可能有 prompt cache 命中，但 cache
  ratio 与最终准确率都明显低于修复后的 `/responses` path。

因此 adapter 应该从“协议转换器”降级为“Responses API 反向代理 + ModelHub
认证/路由层”。

## 决策

采用 response-first adapter 方案：

- 默认所有 Codex `/v1/responses` 请求都转发到 ModelHub `/responses` endpoint。
- 默认不再按模型名称把 `gpt-5.4*` 或其他模型自动分流到 `/v2/crawl`。
- 默认不再把 Responses request body 转换为 Chat Completions body。
- 请求体默认原样透传。adapter 只允许修改网关层信息：URL、query 参数、HTTP
  headers、超时、重试、上游选择和日志。
- legacy chat path 可以短期保留，但只能通过显式配置启用，用于回滚或对照实验。
- `/v1/responses/compact` 在 responses 模式下优先代理到 ModelHub
  `/responses/compact`，不能默认使用本地 lossy compact。

## 非目标

本次设计不包含：

- 删除所有历史 chat mapping 代码。删除可以作为 NR3D 完整验收通过后的后续清理。
- 改 Codex SDK 或 OpenAI provider 配置模型。
- 把真实 AK 写入仓库、文档、日志或测试 fixture。
- 修改 NR3D 任务逻辑、prompt、proposal pool、数据集或 scoring 规则。
- 用 smoke test 替代完整 NR3D strat600 验收。

## 请求体透传不变量

adapter 的默认 responses 路径必须满足：

```text
request body in localhost /v1/responses
  == semantic request body sent to ModelHub /responses
```

默认情况下不得删除、重排、转换或解释这些字段：

- `model`
- `input`
- `input[].content`
- `type: reasoning`
- `encrypted_content`
- `previous_response_id`
- `store`
- `tools`
- tool call 和 tool result item
- `text.format`
- `json_schema`
- `stream`
- `max_output_tokens`
- `reasoning`
- `metadata`

如果后续确实需要兼容补丁，例如为旧调用方补 `store=true` 或
`max_output_tokens`，必须满足三个条件：

1. 默认关闭，放在显式 legacy compatibility 开关后面。
2. 单元测试覆盖补丁打开和关闭两种路径。
3. NR3D 验收使用默认透传路径，而不是 legacy body mutation 路径。

这条规则的原因很直接：既然 Codex/OpenAI SDK 和 ModelHub 当前都使用 Responses
API，adapter 不应再承担协议语义解释。语义字段由调用方和 ModelHub 共同定义，
adapter 只负责传递。

## 网关职责

保留 adapter，是因为它仍然承担 ModelHub 接入层职责：

- 将本地 `/v1/responses` 映射到 ModelHub `/api/modelhub/online/responses`。
- 将本地 `/v1/responses/compact` 映射到 ModelHub `/api/modelhub/online/responses/compact`。
- 根据环境选择办公网域名或线上域名。
- 从 gitignored 私有配置或环境变量读取 AK，并以 query 参数注入上游请求。
- 支持多 AK weighted pool。
- 按 `chat_run_id` 或等价 request chain id 做 sticky routing，避免一个样本链路
  中途换 AK。
- 注入 `X-TT-LOGID` 和必要的 extra header。
- 对 429 做有限重试和 failover。
- 提供 `/health`，暴露上游模式、路径、环境和配置状态。
- 对日志、health、错误响应做 secret redaction，禁止输出 AK、完整上游 URL 或
  私有 header 值。

这些职责属于 transport/auth/routing 层，不应修改 Responses body 的语义。

## 目标架构

```text
Codex SDK / Codex CLI
  |
  | POST http://127.0.0.1:<port>/v1/responses
  | body: OpenAI Responses request
  v
codex_modelhub_adapter
  - parse headers and request metadata
  - choose upstream environment
  - choose AK with sticky routing
  - build ModelHub URL with ak query
  - forward original body
  - stream or return upstream response
  v
ModelHub
  POST https://<domain>/api/modelhub/online/responses?ak=<redacted>
```

推荐拆分边界：

- `adapter.app`：FastAPI routing、request/response streaming、health endpoint。
- `adapter.proxy`：上游选择、URL 构建、重试/failover、secret redaction。
- `adapter.settings` 或现有 settings model：环境变量和 TOML 配置解析。
- `adapter.mapping`：只保留 legacy chat mapping 或删除默认调用；responses 默认路径
  不应依赖 mapping。

如果现有代码文件过大，可以做小幅拆分，但本次不做无关重构。优先让默认数据流
变简单、可读、可测。

## 路由策略

`upstream_api` 的语义调整为：

| 配置值 | 行为 |
| --- | --- |
| `responses` | 强制走 ModelHub Responses API。推荐生产默认值。 |
| `auto` | responses-first；除非显式 legacy 条件命中，否则等价于 `responses`。 |
| `chat_completions` | 显式 legacy fallback，只用于回滚或对照实验。 |

`chat_completions_models` 不再能在默认配置下把 `gpt-5.4*` 自动分流到
`/v2/crawl`。如果保留该配置，只能在 `upstream_api=chat_completions` 或明确的
legacy mode 下生效。

## Compact 策略

`/v1/responses/compact` 不能默认走本地 lossy compaction。新的策略：

1. responses 模式下代理到 ModelHub `/responses/compact`。
2. 如果上游 compact 不可用，返回明确错误，或者在显式 legacy fallback 开关打开
   时才使用本地 compact。
3. fallback 发生时记录不含敏感信息的结构化日志，例如：

```text
compact_mode=local_fallback upstream_api=responses reason=<redacted_error_class>
```

NR3D 最终验收时，不能出现大量 local compact fallback。否则即使 accuracy 暂时
达标，也不能说明 reasoning carryover 风险已经消除。

## 错误处理

错误处理遵循 fail-loud 原则：

- 上游返回 4xx/5xx 时，adapter 保留状态码和可公开错误摘要，不伪造成成功。
- 429 可以按现有策略做有限重试和 AK failover，但达到上限后必须返回失败。
- 网络超时返回清晰的 gateway error，并记录 redacted upstream alias、环境、
  retry count。
- legacy fallback 必须显式配置；不能在 responses 失败时静默降级到 chat path。
- 所有错误日志必须脱敏，禁止包含 `ak=`、完整私有 URL query、secret header 值、
  原始大段 prompt 或模型私有输出。

## 观测与调试

`/health` 和结构化日志需要能回答这些问题：

- 当前默认 upstream API 是否为 `responses`。
- 当前 ModelHub path 是否为 `/responses`。
- 当前是否启用了 legacy chat fallback。
- 当前是否启用了 legacy body mutation。
- 当前上游环境是办公网域名还是线上域名。
- AK pool 是否加载成功；只显示 alias/count/weight，不显示 AK。
- 最近请求是否发生 429 retry、failover、compact fallback。

请求级日志建议只记录计数和类型，不记录内容：

- `body_passthrough=true`
- `has_previous_response_id=true|false`
- `reasoning_item_count=<n>`
- `encrypted_content_count=<n>`
- `stream=true|false`
- `text_format=json_schema|text|none`

这些字段用于验证 reasoning carryover 没有被 adapter 丢掉，同时避免泄露内容。

## 测试设计

### 单元测试

覆盖默认 behavior：

- `upstream_api=auto` 对 `gpt-5.4-2026-03-05` 解析为 responses。
- `upstream_api=responses` 使用 `/responses` path。
- 默认 responses path 不调用 chat body builder。
- 默认 responses path 不新增、删除或改写 request body 字段。
- 包含 `type: reasoning` 和 `encrypted_content` 的 fixture 被原样转发。
- 包含 `previous_response_id` 的 fixture 被原样转发。
- 包含 `text.format.json_schema` 的 fixture 被原样转发。
- `/responses/compact` 默认构造 ModelHub compact URL。
- health 和日志输出不包含 `ak=`。

覆盖 legacy behavior：

- 只有显式 `upstream_api=chat_completions` 时才允许 chat mapping。
- legacy body mutation 开关关闭时不补 `store` 或 `max_output_tokens`。
- legacy body mutation 开关打开时补丁行为可预测、可测试、可观测。

### Live smoke

用真实 ModelHub 配置做三条 smoke：

1. direct ModelHub `/responses`：简单 `1+1`，确认 HTTP 200。
2. adapter `/v1/responses`：同样请求，确认返回等价 response。
3. Codex SDK `thread.run(output_schema=...)`：确认 `TurnStatus.completed`，
   schema JSON 可解析。

这些 smoke 只能证明链路可用，不能证明 NR3D 修复合格。

### NR3D canary

先跑小样本 canary，用于提前发现 wiring 问题：

- 确认日志显示主路径是 `/responses`。
- 确认没有系统性 chat fallback。
- 确认没有系统性 local compact fallback。
- 确认 cache ratio 不呈现 v6/chat 的低 ratio 特征。
- 确认 sample failure 不集中在 schema、tool loop 或 gateway error。

canary 不作为最终验收依据。

### NR3D final gate

最终必须跑完整 strat600。以历史 v5 `/responses` 结果为合格基线，要求：

- Overall Acc@0.25 `>=83.0%`
- View-Dep `>=76.5%`
- mean cache ratio `>=0.85`
- cache hit rate `>=0.98`
- hard failures `<=10`
- 主路径必须是 `/responses`
- 不能大量 fallback 到 chat path
- 不能大量 fallback 到 local compact

如果结果接近 v6/chat 特征，例如 Overall 约 `80%`、mean cache ratio 约 `0.57`，
即使接口 smoke 全部通过，也判定为不合格。

## 迁移步骤

1. 增加或调整单元测试，先锁定 responses passthrough 行为。
2. 调整默认 route resolution：`auto` 和生产默认都指向 responses。
3. 调整 responses upstream builder：默认直接转发 body，不做 semantic mutation。
4. 将 legacy body mutation 放到显式配置后面。
5. 将 chat mapping 限制到显式 legacy mode。
6. 将 `/responses/compact` 改为默认代理 ModelHub compact endpoint。
7. 增加 health/log 字段，证明当前运行模式和 fallback 状态。
8. 跑 live smoke。
9. 跑 NR3D canary。
10. 跑 NR3D full strat600，并把结果写入 `docs/benchmark/nr3d/`。

## 回滚策略

短期保留 legacy chat path，作为明确回滚开关：

```text
AIDP_CODEX_PROXY_UPSTREAM_API=chat_completions
```

回滚只用于恢复服务或做对照实验。任何回滚运行都必须在日志和 benchmark 文档中
标明 `upstream_api=chat_completions`，不能和 responses-first 验收结果混淆。

如果 responses-first 改动导致 live smoke 失败，优先回滚 adapter route 配置。
如果 smoke 通过但 NR3D final gate 失败，不能简单宣布升级成功，需要继续排查
body passthrough、compact、reasoning carryover、AK sticky routing 和 ModelHub
上游行为。

## 文档要求

实现完成后需要更新：

- `codex_modelhub_adapter` 的本地 README 或运行说明，明确默认路径是 responses。
- `docs/codex_agent/reasoning_carryover_caching_ak_routing_20260611.md`，补充这次
  adapter 简化对 reasoning carryover 的保护边界。
- `docs/benchmark/nr3d/README.md`，登记新一轮结果和它与 v5/v6 的关系。
- 新的 `docs/benchmark/nr3d/v*_*.md`，保存完整 strat600 命令、环境、结果、
  cache 指标、failure 分析和是否满足 gate。

如果只新增 `docs/superpowers/specs/` 下的设计或计划文档，不需要更新 MkDocs
导航，因为当前 `mkdocs.yml` 明确排除了 `superpowers/**`。

## 成功标准

这次设计成功的标准是：

- adapter 默认路径足够简单：本地 Responses 请求到 ModelHub Responses 请求。
- request body 默认原样透传，reasoning state 不被 adapter 删除或降级。
- legacy chat path 无法被默认模型 pattern 意外触发。
- compact 默认不再本地 lossy 处理。
- smoke、unit、canary、full NR3D strat600 形成分层证据。
- NR3D full strat600 满足 v5-level parity gate。

只有接口可用但 NR3D 指标退化，不算合格修复。
