# 01 Agent/RAG 评测与回归系统优化计划

## 最新状态快照（2026-05-14）

这份文档已经不只是计划文档，也记录当前实现状态。当前第一个优化已经完成评测系统的主体框架，下一阶段的主要阻塞不在代码，而在真实知识库规模和人工确认的 regression case。

### 已完成

- 新增 `multimodal_rag_agent/eval/` 评测包。
- 重构 `scripts/replay_eval_dataset.py`，支持 `--mode agent` 和 `--mode retrieval`。
- 新增 `scripts/export_eval_badcases.py`，支持从结构化日志导出 badcase 草稿。
- 新增可提交脱敏 smoke fixture：`tests/fixtures/evals/smoke_cases.jsonl`。
- 支持 Agent 级回放评测。
- 支持 retrieval-only 检索级评测。
- 支持失败原因分类。
- 支持 `summary.json`、`results.jsonl`、`report.md` 输出。
- 支持 `--baseline` summary 级对比。
- 支持 badcase draft export。
- 已用本地 Qdrant 和本地 Agent 跑过真实 smoke。
- 已生成待人工审核 regression 草稿：
  - `data/evals/regression_drafts/pending_review_20260514.jsonl`
  - `data/evals/regression_drafts/pending_review_20260514.md`

### 当前本地索引状态

当前 `data/index_manifest.json` 显示：

- root token 数：1
- document_count：1
- chunk_count：26

这说明当前知识库规模适合验证链路，不适合直接构建高质量正式 regression dataset。

### 当前 chunk 切分状态

当前切分配置：

- `MULTIMODAL_RAG_CHUNK_SIZE` 默认：512 字符
- `MULTIMODAL_RAG_CHUNK_OVERLAP` 默认：128 字符

当前 Qdrant 中 26 个 chunk 的长度分布曾统计为：

- min：286
- avg：约 741
- median：约 473
- max：2955

结论：

- 大多数 chunk 在 300-800 字符之间。
- 偶发 1000+ 字符可以接受。
- 超过 2000 字符的 chunk 后续需要重点检查，通常说明长段落、表格、代码块或受保护结构没有被切开。

### 知识库规模建议

当前 26 个 chunk 偏小。正式 regression dataset 建议等知识库扩充后再做。

推荐规模：

- 最低可用：1 万字左右，约 15-25 个有效 chunk。
- 推荐起步：3 万字以上，约 50-80 个有效 chunk。
- 比较理想：5 万到 8 万字，约 80-200 个有效 chunk。
- 10 万字以上：可以开始更系统地做混合检索、rerank、query rewrite A/B 和 source grounding 优化。

建议优先补充的知识库内容：

- 项目使用手册
- 知识沉淀流程
- 知识问答流程
- 飞书机器人配置、权限、事件订阅说明
- Qdrant 启动、索引、重建、排查说明
- 常见 badcase 和修复记录
- 真实 debug 记录
- 已完成优化记录
- 环境变量和配置说明

### 当前不建议继续深做的部分

在知识库仍然只有 26 个 chunk 时，不建议急着做：

- 20-50 条正式 regression case。
- 复杂 rerank A/B。
- 混合检索收益评估。
- LLM judge。
- 大规模 case-by-case baseline diff。

这些能力需要更大的知识库才有区分度。

### 近期最推荐下一步

1. 先扩充飞书知识库到至少 3 万字，理想 5 万字左右。
2. 重新跑 ingest，更新 Qdrant 和 `data/index_manifest.json`。
3. 用 retrieval-only eval 验证新索引能正常检索。
4. 人工审核 `pending_review_20260514.md`，挑出第一批 10 条正式 regression case。
5. 再生成 `data/evals/regression/local_regression.jsonl`。

### 需要人工介入的点

- 确认哪些问题真的符合飞书机器人聊天场景。
- 确认每条 case 的正确来源。
- 确认答案必须包含/不能包含的内容。
- 确认无答案问题的标准拒答话术。
- 确认知识沉淀类请求的真实 intent 名称和是否应禁止检索。

代码框架可以继续自动推进，但正式 regression dataset 需要人工确认后才有工程价值。

## 当前结论

本优化项已经从“计划阶段”推进到“可运行的评测闭环雏形”。目前项目已经具备：

- 可提交的脱敏 smoke fixture
- 可测试的 eval 包
- Agent 级回放评测
- retrieval-only 检索级评测
- Markdown/JSON/JSONL 报告
- 失败原因分类
- baseline 对比
- badcase 草稿导出
- 本地真实 Qdrant/Agent smoke 验证

这个方向仍然是第一优先级。它的价值不只是“写测试脚本”，而是给后续 prompt、RAG、Qdrant、rerank、controller、模型配置改动建立质量安全网。

## 已实现内容

### 1. Eval 包模块化

已新增 `multimodal_rag_agent/eval/`：

- `models.py`：定义 `EvalCase`、`EvalActual`、`EvalResult`、`RunArtifacts`
- `dataset.py`：读取 JSONL dataset，统一字段，兼容旧字段
- `metrics.py`：计算通过/失败、指标和失败原因
- `runner.py`：执行 Agent eval 和 retrieval-only eval
- `report.py`：输出 `summary.json`、`results.jsonl`、`report.md`
- `compare.py`：对比 baseline summary
- `badcase.py`：从结构化日志导出 badcase 草稿

### 2. Eval Case Schema

正式字段：

```json
{
  "id": "case_id",
  "user_query": "用户问题",
  "thread_id": "",
  "language": "中文",
  "history": [],
  "message_context": {},
  "images": [],
  "expected_intent": "kb_search",
  "expected_allow_retrieval": true,
  "expected_rewrite_query": "",
  "expected_answer_must_include": [],
  "expected_answer_must_not_include": [],
  "expected_source_titles": [],
  "expected_source_uris": [],
  "expected_match_status": "matched",
  "reference_answer": "",
  "tags": []
}
```

兼容旧字段：

- `question` -> `user_query`
- `should_retrieve` -> `expected_allow_retrieval`
- `expected_answer_points` -> `expected_answer_must_include`

### 3. Agent 级评测

入口：

```powershell
.venv\Scripts\python.exe scripts\replay_eval_dataset.py `
  --mode agent `
  --dataset tests\fixtures\evals\smoke_cases.jsonl
```

能力：

- 调用真实 `invoke_agent`
- 为每条 case 使用独立 thread
- 支持 history seed
- 读取结构化日志中的 intent、allow_retrieval、rewrite_query、timing
- 生成评测报告
- 单条 case 异常不会中断整批评测

### 4. Retrieval-only 检索级评测

入口：

```powershell
.venv\Scripts\python.exe scripts\replay_eval_dataset.py `
  --mode retrieval `
  --dataset data\evals\local_retrieval_smoke.jsonl
```

能力：

- 直接调用 `RAGQueryPipeline.prepare_context`
- 不走最终 LLM 生成
- 不要求最终回答中的 `来源：`
- 检查 source hit、match_status、retrieved context、retrieval latency
- 支撑后续混合检索、rerank、引用校验优化

### 5. 报告输出

每次 eval 输出到：

```text
data/evals/runs/<run_id>/summary.json
data/evals/runs/<run_id>/results.jsonl
data/evals/runs/<run_id>/report.md
```

报告包含：

- 总 case 数
- 通过/失败数
- pass rate
- intent accuracy
- allow_retrieval accuracy
- source hit rate
- match_status accuracy
- citation presence/source hit
- answer match rate
- latency avg/p50/p95/max
- failure reason 分布
- failed cases 明细

### 6. 失败原因分类

已支持：

- `runtime_error`
- `intent_mismatch`
- `allow_retrieval_mismatch`
- `rewrite_mismatch`
- `retrieval_not_called`
- `unexpected_retrieval_called`
- `retrieval_miss`
- `match_status_mismatch`
- `citation_missing`
- `citation_source_mismatch`
- `answer_missing_required_point`
- `forbidden_claim`
- `no_match_answer_incorrect`
- `latency_regression`

### 7. Baseline 对比

入口：

```powershell
.venv\Scripts\python.exe scripts\replay_eval_dataset.py `
  --mode retrieval `
  --dataset data\evals\local_retrieval_smoke.jsonl `
  --baseline data\evals\runs\<old_run_id>\summary.json
```

对比内容：

- `pass_rate_delta`
- `source_hit_rate_delta`
- `match_status_accuracy_delta`
- `answer_match_rate_delta`
- `p95_latency_delta_ms`
- `failed_cases_delta`
- `newly_failed_case_ids`
- `fixed_case_ids`
- `still_failed_case_ids`

### 8. Badcase 草稿导出

入口：

```powershell
.venv\Scripts\python.exe scripts\export_eval_badcases.py `
  --log data\logs\app.jsonl
```

也可以导出成功请求用于人工复盘：

```powershell
.venv\Scripts\python.exe scripts\export_eval_badcases.py `
  --log data\evals\runs\<run_id>\app.jsonl `
  --include-ok
```

输出：

```text
data/evals/badcase_drafts/<timestamp>.jsonl
```

设计原则：

- 默认只导出失败请求
- `--include-ok` 才导出成功请求
- 只生成草稿，不自动进入 golden regression set
- `expected_*` 字段留空，需要人工确认
- 保留 `actual_intent`、`actual_answer`、`actual_rewrite_query`、耗时和错误信息

### 9. 测试覆盖

已新增：

- `tests/test_eval_dataset.py`
- `tests/test_eval_metrics.py`
- `tests/test_eval_runner.py`
- `tests/test_eval_report.py`
- `tests/test_eval_replay_cli.py`
- `tests/test_eval_compare.py`
- `tests/test_eval_badcase.py`
- `tests/test_eval_badcase_cli.py`
- `tests/fixtures/evals/smoke_cases.jsonl`

当前验证：

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests -v
```

最新结果：`71 passed`

静态检查：

```powershell
.venv\Scripts\python.exe -m ruff check multimodal_rag_agent\eval scripts\replay_eval_dataset.py scripts\export_eval_badcases.py tests\test_eval_badcase.py tests\test_eval_badcase_cli.py tests\test_eval_compare.py tests\test_eval_dataset.py tests\test_eval_metrics.py tests\test_eval_report.py tests\test_eval_runner.py tests\test_eval_replay_cli.py
```

最新结果：`All checks passed`

## 本地真实 smoke 状态

已验证：

- Qdrant 可启动
- 本地索引存在
- `data/index_manifest.json` 显示：
  - `document_count`: 1
  - `chunk_count`: 26
- retrieval-only smoke 已通过
- Agent chat smoke 已通过
- baseline 对比 smoke 已通过
- badcase draft export smoke 已通过

注意：`data/` 目录被 `.gitignore` 忽略，本地真实知识库用例、运行报告和 badcase 草稿不提交。

## 还未实现

### 1. 正式 regression dataset

当前只有：

- 可提交的脱敏 smoke fixture
- 本地私有 smoke dataset
- 待人工确认的 regression draft

还没有 20-50 条正式高价值 regression case。

下一步需要人工确认：

- 问题是否重要
- 正确来源是什么
- 答案必须包含哪些信息
- 答案不能包含哪些错误断言
- 无答案时是否应该拒答

### 2. Case-by-case baseline diff

当前 baseline 对比是 summary 级别，已能看：

- 新增失败 case
- 已修复 case
- 仍失败 case
- 核心指标变化

还没有逐 case 展示每个字段具体怎么变化。

### 3. 多模态评测

还没有系统覆盖：

- 图片理解
- OCR
- 链接解析
- 文件问答
- 图文混合沉淀

### 4. LLM judge / 语义评分

当前主要使用确定性规则：

- 字符串包含
- source title/uri 命中
- match_status 匹配
- 禁止断言检查

还没有引入 LLM judge 或语义相似度评分。

### 5. CI / nightly 分层

还没有正式拆成：

- CI：只跑 fake eval 和脱敏 fixture
- local smoke：跑本地 Qdrant/LLM
- nightly：跑完整真实 regression set

### 6. Badcase 入库流程

当前能导出 badcase 草稿，但还没有：

- 人工审核清单
- 从草稿晋升到正式 regression dataset 的命令
- 重复 case 去重
- 敏感信息检查

## 推荐下一步

### Step 1：人工确认 regression draft

先从 10 条候选 case 开始，不追求一次性做满 50 条。

确认每条：

- `user_query`
- `expected_allow_retrieval`
- `expected_answer_must_include`
- `expected_answer_must_not_include`
- `expected_source_titles`
- `expected_source_uris`
- `expected_match_status`

### Step 2：形成本地正式 regression set

建议路径：

```text
data/evals/regression/local_regression.jsonl
```

这个文件不提交，用于真实知识库本地回归。

### Step 3：保留脱敏提交版 regression fixture

建议路径：

```text
tests/fixtures/evals/smoke_cases.jsonl
tests/fixtures/evals/regression_synthetic.jsonl
```

提交版只放脱敏数据，用于 CI 或开源展示。

### Step 4：再做 case-by-case diff

等正式 regression set 有一定规模后，再做逐 case diff 才有价值。

## 面试表达

可以表述为：

> 为企业知识库 Agent 构建离线评测与回归体系，将原有单文件回放脚本演进为可测试 eval 框架，覆盖 Agent 意图路由、检索召回、引用命中、无答案拒答、失败原因分类、真实 Qdrant smoke、baseline 对比和 badcase 草稿回流；支持后续 prompt、RAG、rerank、模型配置和 controller 改动的量化验证，避免质量退化。
