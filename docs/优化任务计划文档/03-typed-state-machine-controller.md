# Typed State Machine Agent Controller 优化计划

## 定位

这是 Agent 架构治理任务。当前 `agent.py` 是项目最核心的文件，负责配置 Deep Agent runtime、意图识别、历史拼接、群聊上下文、检索工具、沉淀工具、streaming、错误处理和 request summary。它功能完整，但职责过重。

本任务目标是把 Agent 编排从“prompt 驱动的一大段入口逻辑”逐步升级为“typed state machine controller”。这不是换框架，而是在现有 Deep Agents 能力之上，把关键控制流显式建模。

## 核心痛点

- `agent.py` 过大，修改任何 Agent 行为都容易影响其它路径。
- `knowledge_deposit`、`kb_search`、`follow_up` 等路径散落在不同 helper 中。
- 工具调用输入输出主要依赖 prompt 和字符串约定，边界不够硬。
- 每个阶段的 timeout、fallback、错误分类还不够清晰。
- 后续做 eval、trace、retrieval 升级时，需要更稳定的 controller 数据结构。

## 目标

- 显式定义 controller state 和阶段。
- 将意图、上下文构建、路由、工具执行、答案生成拆成可测试单元。
- 保留现有 `invoke_agent` 和 `invoke_agent_stream` 对外接口。
- 降低 `agent.py` 复杂度。
- 为后续 Agent Trace 和 Eval 提供结构化数据。

## 建议目录与文件

建议新增：

- `multimodal_rag_agent/agent_controller/__init__.py`
- `multimodal_rag_agent/agent_controller/models.py`
- `multimodal_rag_agent/agent_controller/context_builder.py`
- `multimodal_rag_agent/agent_controller/router.py`
- `multimodal_rag_agent/agent_controller/history.py`
- `multimodal_rag_agent/agent_controller/tool_gateway.py`
- `multimodal_rag_agent/agent_controller/runtime_gateway.py`
- `multimodal_rag_agent/agent_controller/streaming.py`
- `tests/test_agent_controller_router.py`
- `tests/test_agent_controller_context_builder.py`

建议修改：

- `agent.py`
- `protocols/controller_models.py`
- `protocols/tool_models.py`
- `multimodal_rag_agent/rag_query_pipeline/controller_input_prompts.py`
- `multimodal_rag_agent/rag_query_pipeline/intent_prompts.py`

## 状态模型建议

核心模型建议包括：

```python
ControllerStage = Literal[
    "received",
    "classified",
    "context_built",
    "routed",
    "tool_completed",
    "answered",
    "failed",
]
```

建议定义：

- `AgentTurnInput`：原始用户问题、thread_id、images、message_context、language。
- `IntentDecision`：intent、allow_retrieval、rewrite_query、raw_output。
- `ControllerContext`：历史、群聊上下文、reply_context、image metadata。
- `RouteDecision`：route 类型，可能是 `direct_answer`、`retrieve_then_answer`、`deposit_fast_path`、`agent_runtime`。
- `ToolCallPlan`：工具名、输入 schema、timeout。
- `ControllerResult`：最终答案、sources、trace fields、error。

## 路由规则

建议第一阶段明确以下 deterministic routing：

- `knowledge_deposit`：优先走 deposit fast path，避免 LLM 在写入前改写原文。
- `greeting/chitchat`：不允许检索，直接进入 runtime 或轻量直接回答。
- `follow_up`：默认不检索，除非上下文判断必须查知识库。
- `kb_search`：必须走 retrieval path。
- 图片输入：
  - 如果用户问知识库，则检索。
  - 如果用户只要求看图分析，则不检索。

这些规则应进入 `router.py`，不要散在 prompt 文本里。

## 实施阶段

### 阶段 1：模型抽取

- 新增 controller models。
- 保持原 `ControllerInputContext` 兼容，逐步迁移。
- 为每个模型写 `to_dict()`，方便日志和 eval。

### 阶段 2：历史与上下文抽取

- 将 `_load_history_from_runtime`、`_group_memory_history_turns`、`_merge_history_turns` 移入 `history.py`。
- 将 `_build_controller_context` 拆到 `context_builder.py`。
- 保留原函数作为兼容 wrapper。

### 阶段 3：Router 抽取

- 新增 `route_turn(intent, images, message_context)`。
- 将图片跳过检索、沉淀 fast path、非检索 intent 的判断集中管理。
- 对 router 做纯单元测试。

### 阶段 4：Tool Gateway

- 将 `search_knowledge_tool_text` 和 `deposit_knowledge_tool_text` 包成 typed gateway。
- 工具输入输出都转成协议模型。
- 工具异常转成明确错误类型。

### 阶段 5：Runtime Gateway

- Deep Agent runtime 的构建、缓存、stream 调用集中到 `runtime_gateway.py`。
- `agent.py` 只保留对外入口和少量兼容代码。

## 测试策略

- Router 测试覆盖每个 intent。
- Context builder 测试覆盖 history、reply_context、group_recent_turns。
- Tool gateway 用 fake pipeline 测试错误和成功。
- `invoke_agent` 兼容测试：原有 streaming 和 history 测试必须通过。

## 验收标准

- `agent.py` 行数明显下降，核心职责变为入口适配。
- controller 每个阶段都有结构化模型。
- router 不依赖 LLM 即可单测。
- 现有测试全部通过。
- Eval 和 Trace 可以直接消费 controller 结构化状态。

## 简历表达

可以写成：

> 将 Deep Agents 编排层重构为 typed state-machine controller，显式建模意图、路由、工具调用、历史上下文和错误恢复，提升 Agent 行为稳定性与可测试性。

## 推荐优先级

第三。建议在 Eval 基础框架之后开始，否则重构收益难以量化。
