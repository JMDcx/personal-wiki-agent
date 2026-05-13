# 可靠知识沉淀流水线优化计划

## 定位

知识沉淀是这个项目很有差异化的能力：用户可以把小红书、URL、文本、图片等内容沉淀到飞书知识库，并同步进入本地 Qdrant 索引。它不是普通 RAG 问答，而是“外部内容获取 + AI 结构化整理 + 飞书写回 + 向量入库”的完整业务闭环。

当前链路能跑，但涉及 MCP、docreader、VLM、lark-cli、Feishu API、Qdrant 等多个外部依赖。多外部依赖系统的最大风险是部分失败、重复写入和不可恢复。

## 核心痛点

- 同一个 URL 或内容可能重复沉淀。
- 飞书写成功但 Qdrant 入库失败时，用户看到失败，但飞书文档可能已经存在。
- 沉淀过程没有持久化 job 状态，进程重启后难以恢复。
- source、draft、Feishu doc、wiki node、chunk ids 的映射不完整。
- preview、write、retry、resume 的语义还没有统一。

## 目标

- 为知识沉淀建立幂等机制。
- 为每次沉淀建立可持久化 job 状态。
- 支持失败重试和补偿。
- 支持飞书写回和本地索引之间的恢复。
- 保持现有 agent 工具调用入口兼容。

## 建议目录与文件

建议新增：

- `multimodal_rag_agent/deposit_pipeline/idempotency.py`
- `multimodal_rag_agent/deposit_pipeline/jobs.py`
- `multimodal_rag_agent/deposit_pipeline/state_store.py`
- `multimodal_rag_agent/deposit_pipeline/recovery.py`
- `tests/test_deposit_idempotency.py`
- `tests/test_deposit_job_state.py`
- `tests/test_deposit_recovery.py`

建议修改：

- `multimodal_rag_agent/deposit_pipeline/pipeline.py`
- `multimodal_rag_agent/deposit_pipeline/models.py`
- `multimodal_rag_agent/deposit_pipeline/feishu_writer.py`
- `multimodal_rag_agent/api/routes_deposit.py`
- `agent.py`
- `protocols/tool_models.py`
- `protocols/renderers.py`

## 状态机设计

建议将一次沉淀建模为 `DepositJob`：

- `created`
- `source_fetched`
- `draft_built`
- `preview_ready`
- `feishu_written`
- `indexed`
- `completed`
- `failed`
- `retryable_failed`

关键字段：

- `job_id`
- `idempotency_key`
- `source_type`
- `source_uri`
- `source_hash`
- `request_text`
- `draft_title`
- `draft_markdown_path`
- `feishu_doc_token`
- `feishu_doc_url`
- `wiki_node_token`
- `local_document_id`
- `chunk_ids`
- `status`
- `error_type`
- `error_message`
- `created_at`
- `updated_at`

第一阶段存储可以使用 SQLite：`data/deposit/jobs.sqlite`。这比引入 Redis 或 Postgres 更适合当前项目。

## 幂等设计

建议按来源类型生成 idempotency key：

- 小红书：`xiaohongshu:<feed_id>:<xsec_token hash>`
- URL：`url:<normalized_url hash>`
- 文本：`text:<content hash>`
- 图片：`image:<file content hash>`
- provided content：`provided:<source_url hash>:<content hash>`

行为建议：

- 如果已有 `completed` job，直接返回已有飞书文档链接和本地 document id。
- 如果已有 `retryable_failed` job，允许 retry。
- 如果已有 `running` job，返回“正在处理中”。
- preview 不写入飞书和 Qdrant，但也可以记录 job 方便后续 write。

## 补偿与恢复

典型失败场景：

1. source fetch 失败：可重试，不产生外部副作用。
2. draft build 失败：可重试，不产生外部副作用。
3. Feishu write 失败：可重试，不入 Qdrant。
4. Feishu write 成功，Qdrant ingest 失败：必须记录 `feishu_doc_token`，后续只补本地索引，不重复创建飞书文档。
5. Qdrant ingest 成功，最终响应失败：job 仍应标记 completed。

恢复命令建议：

```powershell
uv run python scripts/recover_deposit_jobs.py --status retryable_failed
```

## 实施阶段

### 阶段 1：Job 模型和 SQLite store

- 定义 `DepositJob`。
- 实现 SQLite 初始化和 CRUD。
- 在 pipeline 开始时创建 job。

### 阶段 2：幂等键

- 实现 source normalization 和 hash。
- pipeline 执行前检查已有 job。
- API 和 agent 工具返回幂等结果。

### 阶段 3：状态落盘

- source fetch、draft build、Feishu write、Qdrant ingest 后更新状态。
- 错误时记录可重试类型。

### 阶段 4：恢复与补偿

- 增加 recovery service。
- 对 `feishu_written` 但未 `indexed` 的 job 补索引。
- 对 fetch/write 阶段失败的 job 支持 retry。

### 阶段 5：API 与用户反馈

- `/api/deposit` 返回 `job_id`、`status`、`feishu_doc_url`。
- 新增 `/api/deposit/jobs/{job_id}` 查询状态。
- Agent 侧回答清晰区分 preview、completed、retryable failed。

## 测试策略

- 单元测试幂等 key 生成。
- 使用 fake writer 和 fake ingest pipeline 模拟部分失败。
- 验证 Feishu 写成功后 Qdrant 失败不会重复写飞书。
- 验证 retry 后能补齐 indexed 状态。
- 验证 completed job 再次请求直接复用结果。

## 验收标准

- 同一 URL 重复沉淀不会重复创建飞书文档。
- Feishu 成功但 Qdrant 失败后可以恢复。
- 每次沉淀都有可查询状态。
- Agent 返回中能明确说明完成、预览或失败原因。
- 老的 `deposit_knowledge_tool_text` 调用方式仍可用。

## 简历表达

可以写成：

> 设计可恢复的知识沉淀流水线，支持多来源内容抽取、AI 结构化整理、飞书写回、向量索引、幂等去重和补偿重试，解决多外部依赖下的部分失败问题。

## 推荐优先级

第四。它适合在 Agent/RAG 基础质量稳定后做，能明显体现 AI 应用后端工程能力。
