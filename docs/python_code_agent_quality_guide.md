# Python Code Agent 项目质量规范

本文档用于约束 code agent 在 Python 项目中的编码、设计、验证和交付行为。目标不是生成“能跑”的代码，而是生成可维护、可测试、可演进、可审查的项目代码。

## 1. 总体原则

- 优先遵守项目已有风格；没有明确项目风格时，遵守 PEP 8、PEP 257、现代 Python typing、pytest、ruff、mypy 的通用实践。
- 先理解现有目录、依赖、测试和调用链，再写代码。
- 保持改动范围小而清晰，避免顺手重构无关模块。
- 业务逻辑、输入校验、IO 副作用、展示/API 层要有明确边界。
- 每次完成改动后，必须运行可用的验证命令；无法运行时说明原因和风险。

## 2. 命名规范

### 2.1 通用命名

| 对象 | 推荐形式 | 示例 |
| --- | --- | --- |
| 包名 | 小写，短名 | `billing` |
| 模块名 | `snake_case` | `order_service.py` |
| 函数 | `snake_case` | `calculate_total()` |
| 方法 | `snake_case` | `load_config()` |
| 变量 | `snake_case` | `user_id` |
| 类 | `CapWords` | `OrderRepository` |
| 异常类 | `CapWords` + `Error` | `InvalidOrderError` |
| 常量 | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT_SECONDS` |
| 类型别名 | `CapWords` | `UserId` |
| 私有成员 | 单下划线前缀 | `_parse_token()` |

### 2.2 命名质量要求

- 名字要表达业务意图，避免 `data`、`info`、`obj`、`tmp`、`handle`、`process` 这类空泛命名，除非作用域极小且语义明显。
- 不用缩写污染可读性；常见领域缩写可以保留，例如 `id`、`url`、`db`、`api`。
- 布尔变量使用判断式命名，例如 `is_active`、`has_permission`、`should_retry`。
- 集合变量使用复数或表达集合含义，例如 `users`、`order_ids`、`records_by_id`。
- 函数名使用动词或动宾结构，例如 `create_order()`、`fetch_user()`、`validate_payload()`。
- 类名使用名词或领域概念，例如 `Order`, `UserRepository`, `PaymentService`。

## 3. 项目结构

新 Python 项目优先使用 `src/` layout：

```text
project/
  pyproject.toml
  README.md
  src/
    package_name/
      __init__.py
      py.typed
      config.py
      domain/
      application/
      infrastructure/
      interfaces/
  tests/
```

### 3.1 目录职责

- `domain/`：业务实体、值对象、领域规则、领域异常；不依赖框架、数据库、HTTP。
- `application/`：用例编排和服务层，组合领域逻辑与外部端口。
- `infrastructure/`：数据库、HTTP 客户端、文件系统、消息队列、第三方 SDK 等具体实现。
- `interfaces/`：CLI、Web API handler、worker entrypoint、RPC adapter 等入口层。
- `tests/`：测试代码，通常放在应用包外部，避免混淆生产代码和测试代码。

### 3.2 分层约束

- 依赖方向应向内：`interfaces -> application -> domain`，`infrastructure` 实现 application/domain 定义的端口。
- 不把业务逻辑写在 API handler、CLI main、ORM model、serializer 里。
- 不创建万能 `utils.py`；应按领域或能力命名，例如 `time_window.py`、`retry_policy.py`、`json_codec.py`。
- 小项目可以减少层级，但仍要分清核心逻辑和 IO 边界。

## 4. 函数与类设计

### 4.1 函数结构

- 一个函数只做一件清晰的事。
- 函数名、参数名、返回类型应基本说明函数用途。
- 公开函数必须有类型标注。
- 复杂公开函数必须有 docstring，说明行为、参数约束、返回值和异常。
- 函数内部推荐顺序：
  1. 参数和前置条件校验。
  2. 核心计算或业务判断。
  3. 外部 IO 调用。
  4. 返回结构化结果。

### 4.2 参数设计

- 参数超过 4-5 个时，优先抽成配置对象、请求对象或领域对象。
- 布尔开关、策略参数、可选参数应使用 keyword-only 参数。
- 禁止使用可变默认参数。
- 避免把大型 `dict` 在多层函数间裸传；应抽成 `dataclass`、`TypedDict` 或 Pydantic model。

推荐：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    timeout_seconds: float


def fetch_user(user_id: str, *, retry_policy: RetryPolicy) -> User:
    ...
```

不推荐：

```python
def fetch_user(user_id, retry=True, timeout=10, headers={}, options=None):
    ...
```

### 4.3 类设计

- 类应代表稳定概念：实体、服务、仓储、客户端、策略、配置。
- 不要为了“面向对象”创建只有一个方法且无状态的类。
- 有状态类应明确状态生命周期和线程/并发安全假设。
- 数据容器优先使用 `dataclass(frozen=True)` 或 Pydantic model。
- 服务类依赖外部资源时，通过构造函数注入依赖，避免函数内部隐式创建全局客户端。

## 5. 类型标注

- 所有公开函数、方法、类属性必须有类型标注。
- 内部复杂函数也应补充类型标注。
- 禁止无理由使用 `Any`；必须使用时，要把范围限制在边界层，并尽快转换成强类型对象。
- 优先使用标准集合泛型：`list[str]`、`dict[str, int]`、`set[str]`。
- 对只读序列参数，优先使用抽象类型：`Sequence[str]`、`Mapping[str, str]`。
- 可空值使用 `T | None`，并在函数开头显式处理。
- 对固定枚举值，使用 `Literal` 或 `Enum`。

示例：

```python
from collections.abc import Mapping, Sequence
from typing import Literal


def summarize_scores(
    scores: Sequence[float],
    *,
    mode: Literal["mean", "median"] = "mean",
    labels: Mapping[str, str] | None = None,
) -> float:
    ...
```

## 6. 参数约束与输入校验

类型标注不等于运行时校验。所有外部输入必须显式校验。

### 6.1 外部输入范围

以下输入都视为不可信：

- HTTP request body、query、headers。
- CLI 参数。
- 环境变量。
- 配置文件。
- 数据库读取结果。
- 消息队列事件。
- 第三方 API 响应。
- LLM/tool 返回内容。

### 6.2 校验方式

- 简单 CLI 参数：使用 `argparse` 的 `type`、`choices`、`required`。
- JSON/API/配置：优先使用 Pydantic v2。
- 内部不可变配置：可使用 `dataclass(frozen=True)` 并在 `__post_init__` 校验。
- 领域约束：放在领域对象或领域服务中，不散落在调用层。

Pydantic 示例：

```python
from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)
    mode: Literal["fast", "accurate"] = "fast"
```

### 6.3 错误信息

参数错误必须包含：

- 字段名。
- 当前非法值。
- 允许范围、格式或枚举值。
- 必要时包含请求 id、任务 id、用户 id 等定位上下文，但不能泄露密钥或隐私。

## 7. 异常处理

- 只捕获能处理的异常，或捕获后补充上下文再抛出。
- 禁止裸 `except:`。
- 禁止吞异常后返回假成功。
- 不用异常处理正常业务分支，除非底层 API 本身以异常表达状态。
- 领域错误定义领域异常，基础设施错误定义基础设施异常。
- 跨边界返回给用户/API 的错误要转换成稳定错误码或结构化错误。

推荐：

```python
try:
    payload = client.fetch(order_id)
except HttpClientError as exc:
    raise OrderFetchError(f"failed to fetch order: order_id={order_id}") from exc
```

不推荐：

```python
try:
    payload = client.fetch(order_id)
except Exception:
    return None
```

## 8. 日志规范

- 使用 `logging.getLogger(__name__)`。
- 库代码不调用 `logging.basicConfig()`。
- CLI 或应用入口可以统一初始化日志。
- 普通用户输出使用 `print()`；诊断信息使用 logging。
- 日志必须带关键上下文，例如任务 id、请求 id、用户 id、资源 id。
- 不记录 secret、token、密码、完整隐私数据。
- 高频日志要控制级别和采样，避免污染生产日志。

示例：

```python
import logging

logger = logging.getLogger(__name__)


def run_job(job_id: str) -> None:
    logger.info("job started", extra={"job_id": job_id})
```

## 9. 配置与依赖

- 使用 `pyproject.toml` 管理项目元数据、构建系统、工具配置。
- 应用项目应有明确依赖版本策略，必要时使用 lock 文件。
- 不把密钥、token、生产账号、私有证书写入仓库。
- 环境相关配置通过环境变量或部署平台注入。
- 配置读取集中在 `config.py` 或专门配置模块，避免业务代码到处读取环境变量。
- 启动时应尽早校验必需配置，失败时给出清晰错误。

## 10. 测试规范

### 10.1 测试分层

- 单元测试：测试纯函数、领域逻辑、参数校验、错误分支。
- 集成测试：测试数据库、HTTP、文件系统、消息队列等真实边界。
- 回归测试：每个 bugfix 都应覆盖曾经失败的输入或场景。

### 10.2 测试要求

- 测试命名清楚表达场景和预期，例如 `test_rejects_empty_query()`。
- 测试不依赖执行顺序。
- 测试不依赖真实生产服务，除非明确标记为集成测试。
- mock 只 mock 外部边界，不 mock 被测核心逻辑。
- 断言业务结果，不只断言函数被调用。

推荐测试结构：

```text
tests/
  unit/
    test_search_request.py
    test_order_policy.py
  integration/
    test_order_repository.py
```

## 11. 工具链与质量门禁

推荐在 `pyproject.toml` 中配置：

- `ruff format`：格式化。
- `ruff check`：lint、import 排序、常见错误检查。
- `mypy` 或 `pyright`：静态类型检查。
- `pytest`：测试。

交付前优先运行：

```bash
ruff format --check .
ruff check .
mypy src tests
pytest
```

如果项目没有启用某个工具，code agent 应根据现有项目脚本运行等价命令，例如：

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```

## 12. Code Agent 工作流程

### 12.1 开始编码前

code agent 必须先完成：

- 识别项目结构、包管理工具、测试命令、lint/type check 命令。
- 阅读相关模块和相邻测试。
- 找到已有命名、错误处理、日志和分层风格。
- 明确改动范围和验证方式。

### 12.2 编码时

code agent 必须遵守：

- 先改测试或补充最小回归测试，再改实现；如果不适合 TDD，应说明原因。
- 保持改动集中，不做无关重构。
- 使用项目已有工具、框架、helper 和抽象。
- 不复制大段相似逻辑；必要时抽小函数或领域对象。
- 不引入新依赖，除非收益明确且已有依赖无法满足。
- 不隐藏失败，不制造假成功路径。

### 12.3 完成后

code agent 必须输出：

- 改了哪些文件和核心行为。
- 运行了哪些验证命令。
- 哪些验证没有运行及原因。
- 剩余风险或建议后续工作。

## 13. 禁止事项清单

- 禁止可变默认参数。
- 禁止裸 `except:`。
- 禁止无理由 `Any`。
- 禁止业务逻辑写在 API handler 或 CLI main 中。
- 禁止全局隐式初始化重型客户端。
- 禁止在业务代码中散落读取环境变量。
- 禁止万能 `utils.py`。
- 禁止把外部输入当作可信数据。
- 禁止提交未解释的魔法数字、魔法字符串。
- 禁止用日志泄露密钥、token、密码、完整隐私数据。
- 禁止只靠手动观察，不运行可用验证命令就声称完成。

## 14. 推荐给 Code Agent 的短指令

可以将下面这段直接放进项目级 agent instruction：

```text
编写 Python 项目代码时，遵守 PEP 8、PEP 257、现代类型标注和本项目已有风格。优先使用 src/ layout、pyproject.toml、ruff、mypy/pyright、pytest。公开函数必须有类型标注；外部输入必须 runtime validation；业务逻辑放在 domain/application 层，不放在 CLI/API handler。参数过多时抽模型；禁止可变默认参数、裸 except、隐式全局状态、万能 utils、未验证的外部输入。改动前阅读相关代码和测试；改动后运行格式检查、lint、类型检查和测试，并说明验证结果。
```

## 15. 审查清单

提交前逐项检查：

- [ ] 命名是否清楚表达业务含义？
- [ ] 函数是否职责单一？
- [ ] 公开 API 是否有类型标注？
- [ ] 外部输入是否有运行时校验？
- [ ] 错误信息是否包含足够定位上下文？
- [ ] 业务逻辑是否远离 API/CLI/ORM 边界层？
- [ ] 是否避免了万能 `utils.py` 和无关抽象？
- [ ] 是否没有泄露 secret 或隐私数据？
- [ ] 是否有对应测试或回归测试？
- [ ] 是否运行了 format、lint、type check、tests？

## 16. 参考资料

- PEP 8 - Style Guide for Python Code: https://peps.python.org/pep-0008/
- PEP 257 - Docstring Conventions: https://peps.python.org/pep-0257/
- Python typing documentation: https://docs.python.org/3/library/typing.html
- Python Packaging User Guide - src layout vs flat layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Python Packaging User Guide - Writing your pyproject.toml: https://packaging.python.org/guides/writing-pyproject-toml/
- pytest Good Integration Practices: https://docs.pytest.org/en/latest/goodpractices.html
- Ruff documentation: https://docs.astral.sh/ruff/
- Pydantic documentation: https://docs.pydantic.dev/latest/
- Python logging HOWTO: https://docs.python.org/3/howto/logging.html
- The Twelve-Factor App - Config and Dependencies: https://12factor.net/
