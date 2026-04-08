# Feishu Wiki RAG Agent

[English](./README.md) | 简体中文

一个轻量级的知识助手，基于 Deep Agents 构建，并由基于 Qdrant 的多模态 RAG 流水线提供支持。

这个项目会将飞书 Wiki 和 Docs 内容索引到 Qdrant 中，并通过 Deep Agent runtime 回答用户问题。当前同时支持通过 websocket 接入飞书，以及通过官方 iLink bot 通道接入个人微信。该 agent 可以直接回答简单的对话类请求，也可以将文档查询委派给一个专门的知识检索子 agent，而这个子 agent 由本地多模态 RAG 流水线驱动。

## 功能特性

- 基于 websocket 的飞书机器人集成
- 基于官方 iLink bot API 的个人微信接入
- 以 Deep Agents runtime 作为主编排层
- 面向文档类问题的专用检索子 agent
- 面向外部链接、文本、图片的知识沉淀链路
- 将飞书 Wiki 和 Docs 内容写入 Qdrant
- 包含检索、重排、合并和答案生成的多模态 RAG 流水线
- 聊天模型和 embedding 模型可分别配置不同 provider
- 可选的图片 OCR / caption 索引流水线，用于处理多模态内容
- 微信文本、链接、图片、文件输入适配
- 通过本地运行的 Xiaohongshu MCP 服务抓取小红书帖子详情
- 使用本地 `AGENTS.md` 记忆和 `SKILL.md` 规范回答行为
- 具备来源感知能力，回答中可以引用已索引文档的标题或链接

## 架构

主运行链路如下：

1. 飞书或微信将用户消息交给通道层
2. `channel/feishu/feishu_channel.py` 或 `channel/weixin/weixin_channel.py` 对消息进行标准化
3. 通道层把附件适配成文本提示和可选本地图片路径
4. `agent.py` 使用稳定的 `thread_id` 调用 Deep Agent runtime
5. 主 agent 判断当前请求是否需要知识检索
6. 检索型问题会被委派给 `knowledge_retriever` 子 agent
7. 检索子 agent 使用多模态 `RAGQueryPipeline` 从 Qdrant 中准备上下文
8. 主 agent 生成最终回复，再由通道发送回用户

索引链路如下：

1. 读取配置中的 Wiki 节点 token 或直接的 `doc` / `docx` token
2. 拉取可读的飞书文档内容
3. 将内容转换成 markdown chunk
4. 生成 embedding
5. 将索引持久化到 Qdrant
6. 将索引清单写入 `data/index_manifest.json`

知识沉淀链路如下：

1. 识别用户是否在明确要求“沉淀到知识库”
2. 从小红书、普通链接、纯文本或图片中提取原始材料
3. 规范化为统一的 markdown 知识稿
4. 按配置选择是否写回飞书 Doc / Wiki
5. 将最终 markdown 写入本地 Qdrant 索引

## 项目结构

```txt
feishu_wiki_rag_agent/
├── AGENTS.md
├── LICENSE
├── README.md
├── README.zh-CN.md
├── agent.py
├── config.py
├── channel/
│   ├── __init__.py
│   └── feishu/
│       ├── __init__.py
│       ├── feishu_channel.py
│       └── feishu_client.py
│   └── weixin/
│       ├── __init__.py
│       ├── weixin_api.py
│       ├── weixin_channel.py
│       └── weixin_message.py
├── indexer.py
├── multimodal_rag_agent/
│   ├── api/
│   ├── docreader_service/
│   ├── image_resolver/
│   ├── ingest_pipeline/
│   ├── multimodal_image_pipeline/
│   └── rag_query_pipeline/
├── pyproject.toml
├── retrieval.py
├── schemas.py
├── scripts/
│   └── verify_xhs_deposit.py
├── skills/
│   └── knowledge-qa/
│       └── SKILL.md
│   └── knowledge-deposit/
│       └── SKILL.md
├── tools/
│   └── xiaohongshu-bin/
│       └── README.md
├── tests/
│   ├── test_agent_controller_flow.py
│   ├── test_query_understand_service.py
│   ├── test_sqlite_checkpointer.py
│   └── test_weixin_channel.py
└── .env.example
```

## 运行要求

- Python 3.11+
- `uv`
- Qdrant
- 一个启用了机器人、消息以及 wiki/doc 读取权限的飞书自建应用
- 如果要使用微信通道，还需要一个 Weixin iLink bot 账号
- 一个兼容 OpenAI 的聊天模型接口
- 一个兼容 OpenAI 的 embedding 模型接口
- Deep Agents runtime 所需依赖

## 安装

```bash
uv sync
```

在建立索引或查询之前先启动 Qdrant。对于本地开发，建议使用持久化的 Docker 运行方式：

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

## 配置

将 `.env.example` 复制为 `.env`，然后填写你的配置。

必填变量：

```env
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=your_feishu_app_secret
FEISHU_EVENT_MODE=websocket
FEISHU_WIKI_ROOT_TOKENS=wiki_or_doc_token

FEISHU_RAG_MODEL=your_chat_model_name
FEISHU_RAG_CHAT_API_KEY=your_chat_api_key
FEISHU_RAG_CHAT_BASE_URL=https://your-chat-compatible-endpoint/v1

FEISHU_RAG_EMBEDDING_MODEL=your_embedding_model_name
FEISHU_RAG_EMBEDDING_API_KEY=your_embedding_api_key
FEISHU_RAG_EMBEDDING_BASE_URL=https://your-embedding-compatible-endpoint/v1
```

可选变量：

```env
FEISHU_RAG_TOP_K=4
FEISHU_RAG_COLLECTION=feishu_wiki_docs
FEISHU_RAG_DATA_DIR=./data
FEISHU_RAG_MANIFEST=index_manifest.json
FEISHU_API_BASE=https://open.feishu.cn
FEISHU_REQUEST_TIMEOUT=20
WEIXIN_BASE_URL=https://ilinkai.weixin.qq.com
WEIXIN_CDN_BASE_URL=https://novac2c.cdn.weixin.qq.com/c2c
WEIXIN_CREDENTIALS_PATH=./data/weixin/credentials.json
WEIXIN_TMP_DIR=./data/weixin/tmp
WEIXIN_REQUEST_TIMEOUT=15
WEIXIN_LONG_POLL_TIMEOUT=35

MULTIMODAL_RAG_QDRANT_URL=http://127.0.0.1:6333
MULTIMODAL_RAG_QDRANT_API_KEY=
MULTIMODAL_RAG_QDRANT_COLLECTION=feishu_wiki_docs
MULTIMODAL_RAG_VECTOR_SIZE=1536
MULTIMODAL_RAG_TOP_K=6
MULTIMODAL_RAG_RERANK_TOP_K=4
MULTIMODAL_RAG_CHUNK_SIZE=512
MULTIMODAL_RAG_CHUNK_OVERLAP=128

FEISHU_DEPOSIT_SPACE_ID=
FEISHU_DEPOSIT_PARENT_NODE_TOKEN=
DEPOSIT_ENABLE_AUTO_WRITE=true
XHS_MCP_URL=http://127.0.0.1:18060/mcp
```

如果你希望聊天和 embedding 共用同一个 provider，也可以使用：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

当示例专用变量存在时，它们仍然具有更高优先级。

如果你已经拿到了可复用的 Weixin iLink token，也可以直接设置：

- `WEIXIN_TOKEN`

## 小红书 MCP 部署

当前仓库推荐使用“本机 macOS 二进制 + 本地 MCP 服务”的方式接入小红书。

推荐目录结构：

1. 将上游二进制放到 [tools/xiaohongshu-bin/README.md](/Users/jmdcx/Documents/GitHub/feishu-wiki-rag-agent/tools/xiaohongshu-bin/README.md) 所在目录：
   - `xiaohongshu-mcp-darwin-arm64`
   - `xiaohongshu-login-darwin-arm64`
2. 赋予执行权限：

```bash
chmod +x tools/xiaohongshu-bin/xiaohongshu-*-darwin-arm64
```

3. 先运行登录二进制：

```bash
./tools/xiaohongshu-bin/xiaohongshu-login-darwin-arm64
```

4. 再启动本地 MCP 服务：

```bash
./tools/xiaohongshu-bin/xiaohongshu-mcp-darwin-arm64
```

5. 保持 `.env` 中的地址指向本地服务：

```env
XHS_MCP_URL=http://127.0.0.1:18060/mcp
```

当前仓库已经不再把之前的 Docker 试验方案作为推荐部署方式。

## 知识沉淀使用方式

当前 agent 支持类似下面的请求：

- `把这个小红书链接沉淀到知识库`
- `把这段文本沉淀到知识库`
- `把这张图片沉淀到知识库`

当来源是小红书时，运行时会调用本地 MCP 服务抓取帖子详情，再生成结构化知识稿。

如果你想验证 Xiaohongshu MCP 和本地沉淀链路是否通畅，可以运行：

```bash
python scripts/verify_xhs_deposit.py --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

默认只做 preview，不写回飞书。如果要走完整写入，请配置：

- `FEISHU_DEPOSIT_SPACE_ID`
- `FEISHU_DEPOSIT_PARENT_NODE_TOKEN`
- `DEPOSIT_ENABLE_AUTO_WRITE=true`

## 飞书配置

你的飞书应用需要：

- 是一个企业自建应用
- 配置为 websocket 事件投递模式
- 具备机器人消息和 wiki/doc 读取权限

至少请确认：

- 已启用 bot capability
- 已启用消息接收与发送权限
- 已启用 wiki/doc 读取权限
- 已订阅 `接收消息 v2.0` 事件
- 最新版本的应用已发布

## 构建索引

启动 Docker 容器：

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

在启动机器人之前，先运行手动入库步骤：

```bash
uv run python indexer.py
```

这个命令会：

1. 遍历配置中的根 token
2. 下载支持的飞书 Wiki 或 Doc 内容
3. 将文档切分为 chunk
4. 生成 embedding
5. 存入 Qdrant
6. 将索引清单写入 `data/index_manifest.json`

如果你更换了 embedding 模型，请在重建索引之前更新 `MULTIMODAL_RAG_VECTOR_SIZE`，使其与模型输出维度一致。

## 本地快速测试

构建完索引后，你可以在不接入飞书的情况下对 Deep Agent 做一次 smoke test：

```bash
uv run python - <<'PY'
from agent import invoke_agent
print(invoke_agent("这个知识库里主要讲了什么？", thread_id="local-smoke-test"))
PY
```

## 运行机器人

```bash
uv run python channel/feishu/feishu_channel.py
```

然后在飞书中进行测试：

- 给机器人发送私聊消息
- 或者在群聊里 @ 机器人

如果要运行个人微信通道：

```bash
uv run python channel/weixin/weixin_channel.py
```

然后在微信中测试：

- 给官方 iLink 接入生成的机器人助手发送私聊消息
- 文本消息会直接透传给 agent
- 链接会先经过本地 docreader 抓取和解析，再交给 agent
- 图片会通过现有 `images=[...]` 接口传给 agent
- 文件会先本地解析为 markdown，并附带解析出的图片上下文

## 说明

- 当前版本仅支持 `FEISHU_EVENT_MODE=websocket`
- 当前飞书主运行时使用 Deep Agents 作为编排层
- 文档检索被委派给专门的 `knowledge_retriever` 子 agent
- 当前项目即使索引了图片 OCR/caption chunk，也只返回文本回复
- 群聊中只有在 @ 机器人时才会触发回复
- 微信第一版只支持个人单聊和文本回复
- 微信语音、群聊以及图片/文件回传暂未实现
- 微信文件理解能力取决于本地 docreader 当前支持的格式
- 如果你更换 embedding 模型，请确保 `MULTIMODAL_RAG_VECTOR_SIZE` 与模型输出维度完全一致

## GitHub 上传检查清单

在将项目推送到 GitHub 之前：

- 删除或忽略你真实的 `.env` 文件，只保留 `.env.example`
- 不要提交真实的 API key、飞书应用密钥或 provider token
- 不要提交 `data/` 下的任何内容；其中可能包含 Qdrant 存储、manifest、提取出的图片和文档派生内容
- 不要提交 `.venv/`、`__pycache__/`、`.pytest_cache/` 或 `*.egg-info/`
- 保留 `uv.lock`，这样其他人就可以通过 `uv sync` 复现环境
- 重新检查 `README.md`，确认其中的配置和启动步骤仍与当前项目一致
- 如果仓库将公开，请确认你的飞书索引内容允许保留在本地机器之外

## License

MIT
