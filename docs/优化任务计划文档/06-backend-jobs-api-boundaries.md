# 后端任务化与 API 服务边界优化计划

## 定位

AI 应用开发本质上仍然是后端工程。这个项目现在已经有 FastAPI、Feishu channel、Weixin channel、Qdrant、Deep Agents、沉淀流水线和结构化日志，但整体仍偏“脚本入口 + 模块调用”。如果要进一步体现生产级能力，需要把长任务、API 边界、健康检查、错误模型和外部依赖隔离补齐。

本任务目标是把项目从可运行的 Agent 项目，推进到更像生产后端服务的结构。

## 核心痛点

- 索引构建、知识沉淀、OCR、评测回放都是长任务，不适合一直同步阻塞。
- FastAPI 路由直接持有 pipeline 实例，service layer 不够清晰。
- channel、API、scripts 调用同一业务能力时缺少统一服务边界。
- 没有统一 job status API。
- 健康检查和依赖 readiness 不完整。
- 错误响应缺少统一结构。

## 目标

- 建立轻量后台任务系统。
- 建立清晰 service layer。
- 提供 job submit/status/retry API。
- 增加 health/readiness。
- 隔离外部依赖 client。
- 让 channel、API、scripts 复用同一服务能力。

## 建议目录与文件

建议新增：

- `multimodal_rag_agent/backend/__init__.py`
- `multimodal_rag_agent/backend/jobs.py`
- `multimodal_rag_agent/backend/job_store.py`
- `multimodal_rag_agent/backend/worker.py`
- `multimodal_rag_agent/backend/errors.py`
- `multimodal_rag_agent/backend/services.py`
- `multimodal_rag_agent/api/routes_jobs.py`
- `multimodal_rag_agent/api/routes_health.py`
- `scripts/run_worker.py`
- `tests/test_backend_jobs.py`
- `tests/test_api_health.py`

建议修改：

- `multimodal_rag_agent/api/app.py`
- `multimodal_rag_agent/api/routes_query.py`
- `multimodal_rag_agent/api/routes_ingest.py`
- `multimodal_rag_agent/api/routes_deposit.py`
- `config.py`
- `docker-compose.yml`
- `.env.example`

## 后端分层建议

建议分成四层：

### 1. API Layer

负责：

- 请求参数校验。
- 返回统一响应。
- 不直接写复杂业务逻辑。

### 2. Service Layer

负责：

- Query service。
- Ingest service。
- Deposit service。
- Eval service。
- Job service。

### 3. Infrastructure Layer

负责：

- Qdrant client。
- Feishu client。
- Weixin API。
- OpenAI-compatible client。
- lark-cli runner。
- SQLite job store。

### 4. Agent Layer

负责：

- Agent controller。
- Tool gateway。
- Runtime gateway。

## Job 系统设计

第一阶段建议用 SQLite + 线程 worker，不急着引入 Celery。

Job 类型：

- `deposit`
- `ingest_url`
- `ingest_file`
- `rebuild_index`
- `eval_run`
- `recover_deposit`

Job 状态：

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `retryable_failed`

Job 字段：

- `job_id`
- `job_type`
- `payload_json`
- `status`
- `result_json`
- `error_type`
- `error_message`
- `attempt_count`
- `max_attempts`
- `created_at`
- `started_at`
- `finished_at`

## API 设计建议

新增：

- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/retry`
- `GET /healthz`
- `GET /readyz`

改造：

- `/api/deposit` 可以支持同步和异步两种模式。
- `/api/ingest/url` 可以返回 job_id。
- `/api/query` 继续保持同步，因为用户问答通常需要实时返回。

统一响应格式：

```json
{
  "ok": true,
  "data": {},
  "error": null,
  "request_id": "api-query-xxx"
}
```

错误格式：

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "QDRANT_UNAVAILABLE",
    "message": "Qdrant is not ready",
    "retryable": true
  },
  "request_id": "api-query-xxx"
}
```

## Health 与 Readiness

`/healthz`：

- 只检查进程是否活着。
- 不访问外部依赖。

`/readyz`：

- 检查 Qdrant 是否可访问。
- 检查配置是否完整。
- 可选检查 Feishu app id/secret 是否存在。
- 不调用高成本 LLM。

## 实施阶段

### 阶段 1：统一错误模型

- 定义 `AppError`。
- 定义错误 code。
- API route 捕获后统一返回。

### 阶段 2：Service Layer

- 将 query、ingest、deposit route 中的 pipeline 调用移入 service。
- route 只负责参数和响应。

### 阶段 3：Job Store

- 实现 SQLite job store。
- 支持 create、get、update_status、list_recent。

### 阶段 4：Worker

- 实现单进程 worker loop。
- 支持按 job_type dispatch。
- 支持失败重试。

### 阶段 5：API 接入

- 新增 jobs route。
- deposit 和 ingest 支持 async submit。
- 增加 health route。

### 阶段 6：部署配置

- 更新 `docker-compose.yml`，保留 qdrant，并为 API/worker 预留服务定义。
- 更新 `.env.example`，增加 job store 和 worker 配置。

## 测试策略

- 单元测试 job store 状态流转。
- fake handler 测试 worker 成功和失败。
- API 测试统一响应格式。
- readiness 测试 Qdrant 不可用时返回非 ready。
- 不在单元测试里真实调用外部模型。

## 验收标准

- `/healthz` 和 `/readyz` 可用。
- 至少 deposit 或 ingest_url 能异步提交 job。
- 能查询 job 状态和结果。
- API 响应格式统一。
- channel、API、scripts 可以逐步复用 service layer。

## 简历表达

可以写成：

> 将脚本型 Agent 应用演进为任务驱动的后端服务，支持异步 job、状态查询、健康检查、统一错误模型和外部依赖隔离，为 RAG 评测、知识沉淀和索引构建提供稳定后端基础。

## 推荐优先级

第六。它不应抢在 Agent 质量优化之前，但应该作为可靠沉淀、Eval 回放和生产化部署的基础逐步补齐。
