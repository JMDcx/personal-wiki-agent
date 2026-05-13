# Hybrid Retrieval、Rerank 与 Citation Validation 优化计划

## 定位

这是 RAG 质量提升的核心任务。当前项目的检索链路已经能工作：`RAGQueryPipeline` 会执行 query understand、dense retrieval、轻量 rerank、merge 和 context build。但现有 rerank 主要靠简单 lexical boost，chunk 之间也没有更强的结构化关系，引用校验还比较依赖 prompt 约束。

本任务目标是把检索链路从“可用”升级到“可调优、可解释、可评测”。

## 核心痛点

- Dense retrieval 对关键词、专有名词、短查询和数字类问题不够稳定。
- 当前 rerank 逻辑较轻，难以处理多个候选片段相关度接近的情况。
- 检索结果没有明确区分 text、image_ocr、image_caption 的优先级策略。
- 回答里的 `来源：` 可能依赖模型遵守提示，缺少程序侧校验。
- no-match 判断目前是固定阈值，后续需要和 eval 指标联动。

## 目标

- 引入 hybrid retrieval：dense + lexical/BM25。
- 引入更可靠的 rerank 层。
- 支持 chunk neighbor expansion 和 parent-child context。
- 增强 metadata filter。
- 在回答前后建立 citation validation。
- 与评测系统打通，能证明检索质量提升。

## 建议涉及文件

建议新增：

- `multimodal_rag_agent/rag_query_pipeline/hybrid_retrieval.py`
- `multimodal_rag_agent/rag_query_pipeline/lexical_index.py`
- `multimodal_rag_agent/rag_query_pipeline/citation.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_citation_validation.py`

建议修改：

- `multimodal_rag_agent/rag_query_pipeline/pipeline.py`
- `multimodal_rag_agent/rag_query_pipeline/retrieval.py`
- `multimodal_rag_agent/rag_query_pipeline/rerank.py`
- `multimodal_rag_agent/rag_query_pipeline/prompt_builder.py`
- `multimodal_rag_agent/ingest_pipeline/chunking.py`
- `multimodal_rag_agent/ingest_pipeline/qdrant_index.py`
- `multimodal_rag_agent/config.py`
- `protocols/tool_models.py`

## 方案设计

### 1. Hybrid Retrieval

第一阶段建议使用本地 sidecar lexical index，不急着引入复杂基础设施。

实现方式：

- 入库时为每个 chunk 生成 lexical payload。
- 使用 chunk content、title、headers、section_path、ocr_text、caption_text 建立可搜索文本。
- 本地 BM25 可以先用轻量实现，或者用简单 inverted index + term score。
- 查询时并行获取：
  - dense candidates from Qdrant
  - lexical candidates from sidecar index
- 合并候选时按 chunk_id 去重，并保留两个分数：
  - `dense_score`
  - `lexical_score`

第二阶段再考虑 Qdrant sparse vector 或专门搜索引擎。

### 2. Rerank

建议分两级：

- 第一层 deterministic rerank：分数归一化、title boost、exact phrase boost、source type boost。
- 第二层 optional model rerank：如果配置了 rerank provider，再调用 cross-encoder 或 LLM reranker。

配置建议：

- `MULTIMODAL_RAG_HYBRID_ENABLED=true`
- `MULTIMODAL_RAG_DENSE_WEIGHT=0.65`
- `MULTIMODAL_RAG_LEXICAL_WEIGHT=0.35`
- `MULTIMODAL_RAG_RERANK_PROVIDER=none`
- `MULTIMODAL_RAG_NEIGHBOR_EXPANSION=1`

### 3. Chunk Expansion

当前 chunk 是独立片段。建议增加：

- `previous_chunk_id`
- `next_chunk_id`
- `parent_document_id`
- `section_path`

检索 top chunks 后，按配置补充相邻 chunk，避免答案上下文断裂。

### 4. Citation Validation

程序侧校验分两层：

- 工具返回层：`render_retrieval_result_text` 只输出真实 retrieved sources。
- 最终回答层：解析答案中的 `来源：`，检查是否来自 retrieved source titles 或 uris。

如果校验失败，建议第一阶段只记录 warning 和 badcase，不自动重写答案。第二阶段再考虑自动修复。

## 实施阶段

### 阶段 1：检索结果结构升级

- 在 `RetrievedChunk.metadata` 中补充分数字段。
- 修改 `RAGQueryPipeline.prepare_context` 返回 dense、lexical、merged 的候选数量。
- 保持旧接口兼容。

### 阶段 2：Lexical Index

- 建立 chunk 文本标准化方法。
- 在 ingest 完成后写 sidecar lexical index。
- 提供 `LexicalRetriever.search(query, top_k)`。

### 阶段 3：Hybrid 合并

- 实现 score normalization。
- 合并 dense 和 lexical 候选。
- 按权重计算 `hybrid_score`。

### 阶段 4：Rerank 与 Expansion

- 改造 `Reranker`，支持 hybrid score、title boost、exact phrase boost。
- 增加 neighbor expansion。
- 更新 prompt context，明确标注来源、片段类型和 section。

### 阶段 5：Citation Validation

- 新增 citation 校验模块。
- 将校验结果写入 request summary。
- 失败 case 自动进入 eval/badcase 目录。

## 测试策略

- 构造 fake chunks 验证 score merge。
- 构造中文短查询验证 lexical 命中。
- 构造重复 chunk 验证去重。
- 构造 citation 正确和错误答案验证 citation validator。
- 用 eval dataset 对比优化前后 source hit rate 和 no-match accuracy。

## 验收标准

- hybrid retrieval 可以通过配置开关启用或关闭。
- 检索返回结果中能看到 dense、lexical、hybrid 分数。
- 同一 query 的 top-k source hit rate 在 eval dataset 上不低于旧方案。
- citation 校验能识别无效来源。
- 所有改动保持 `invoke_agent` 和 `/api/query` 兼容。

## 简历表达

可以写成：

> 优化多模态企业知识库 RAG 检索架构，引入 hybrid retrieval、reranking、chunk expansion 与引用校验，提高复杂问题召回质量并降低无依据回答。

## 推荐优先级

第二。建议在评测框架建立后启动，用评测结果驱动权重和策略调整。
