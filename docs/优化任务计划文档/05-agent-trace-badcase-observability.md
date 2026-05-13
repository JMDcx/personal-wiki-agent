# Agent Trace 与 Badcase 诊断系统优化计划

## 定位

项目已经有较好的结构化日志基础：`observability/context.py`、`observability/events.py`、`observability/logging.py` 能记录 request summary、阶段耗时和关键事件。下一步要解决的问题是：当 Agent 答错、检索 miss、沉淀失败或回复超时时，开发者能不能快速知道错在哪里。

本任务目标是把日志升级成面向 Agent 系统的 trace 和 badcase 诊断能力。

## 核心痛点

- 普通后端日志回答不了“模型为什么这么答”。
- 检索、rerank、工具调用、LLM 输出、最终回答之间的关系不够直观。
- badcase 需要手动从日志里捞，难以复现。
- token、成本、延迟等指标没有形成统一视图。
- Agent 调优需要知道每次请求的完整决策路径。

## 目标

- 为每个请求生成结构化 timeline。
- 自动保存 badcase 所需上下文。
- 记录模型、token、成本、延迟、工具调用摘要。
- 支持从 trace 生成 eval case 草稿。
- 不引入重型可观测性平台，先用本地 JSONL 和 Markdown 报告。

## 建议目录与文件

建议新增：

- `observability/trace.py`
- `observability/badcase.py`
- `observability/costs.py`
- `scripts/export_badcases.py`
- `docs/agent-trace.md`
- `tests/test_agent_trace.py`
- `tests/test_badcase_export.py`

建议修改：

- `agent.py`
- `observability/context.py`
- `observability/events.py`
- `observability/logging.py`
- `multimodal_rag_agent/rag_query_pipeline/pipeline.py`
- `multimodal_rag_agent/deposit_pipeline/pipeline.py`
- `channel/feishu/feishu_channel.py`
- `channel/weixin/weixin_channel.py`

## Trace 结构设计

建议每个请求生成一个 trace 文件：

```text
data/traces/<date>/<request_id>.json
```

建议结构：

```json
{
  "request_id": "feishu:om_xxx",
  "thread_id": "feishu:oc_xxx",
  "channel": "feishu",
  "question_preview": "用户问题",
  "events": [
    {
      "stage": "intent",
      "event": "intent_model_classified",
      "timestamp": "2026-05-13T20:00:00",
      "duration_ms": 123,
      "fields": {}
    }
  ],
  "summary": {
    "intent": "kb_search",
    "allow_retrieval": true,
    "retrieval_ms": 800,
    "llm_ms": 3000,
    "reply_ms": 200
  }
}
```

关键 stage：

- `channel_received`
- `message_normalized`
- `intent`
- `history`
- `routing`
- `retrieval`
- `rerank`
- `tool`
- `llm`
- `deposit`
- `reply`
- `error`

## Badcase 设计

badcase 目录建议：

```text
data/badcases/<date>/<request_id>.json
data/badcases/<date>/<request_id>.md
```

触发条件：

- request failed。
- retrieval no_match 但用户追问表示不满意。
- citation validation failed。
- LLM 或工具超时。
- 用户手动标记，后续可以通过命令补充。

badcase 内容：

- 原始问题。
- thread_id。
- message_context。
- intent 和 rewrite_query。
- retrieved sources。
- tool result。
- final answer。
- error 信息。
- 运行配置摘要。

## 成本与延迟统计

建议记录：

- `intent_ms`
- `retrieval_ms`
- `rerank_ms`
- `llm_ms`
- `reply_ms`
- `total_ms`
- `model`
- `prompt_tokens`
- `completion_tokens`
- `estimated_cost`

如果 provider 不返回 token usage，先允许为空，不阻塞 trace。

## 实施阶段

### 阶段 1：Trace Collector

- 在 request context 中增加 trace event buffer。
- `log_event` 时可选写入 trace。
- 请求结束时 flush 到 `data/traces/`。

### 阶段 2：关键链路接入

- Agent controller 阶段写 trace。
- RAG prepare_context 写 retrieved chunks 摘要。
- deposit pipeline 写每个阶段状态。
- channel 写 reply 结果。

### 阶段 3：Badcase Writer

- 实现 `record_badcase(reason, payload)`。
- 在 error、no_match、citation_failed 时写 badcase。
- 生成 Markdown 便于人工阅读。

### 阶段 4：导出 Eval Case

- `scripts/export_badcases.py` 支持将 badcase 转成 eval JSONL 草稿。
- 人工补充 expected fields 后进入 golden dataset。

## 测试策略

- fake request context 下写 trace，验证文件结构。
- 模拟 request_failed，验证 badcase JSON 和 Markdown。
- 模拟 citation_failed，验证 badcase reason。
- 不在测试里调用真实 Feishu、Qdrant 或 LLM。

## 验收标准

- 每个成功请求能生成 trace summary。
- 每个失败请求能生成 badcase 文件。
- 从 badcase 能复现当时的问题、检索结果和最终回答。
- trace 不记录原始密钥。
- trace preview 长度可控，避免日志爆炸。

## 简历表达

可以写成：

> 建设 Agent 全链路观测与 badcase 诊断系统，记录意图、检索、工具调用、LLM 输出、引用和延迟成本，显著提升线上问题定位效率。

## 推荐优先级

第五。它和 Eval 系统互相增强，适合在前几项开始落地后同步推进。
