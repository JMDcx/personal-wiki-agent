# Feishu Wiki RAG Agent

English | [简体中文](./README.zh-CN.md)

A lightweight Feishu knowledge assistant built on Deep Agents and backed by a Qdrant-based multimodal RAG pipeline.

This project indexes Feishu Wiki and Docs content into Qdrant and answers user questions through a Deep Agent runtime. It currently supports Feishu over websocket and personal Weixin through the official iLink bot channel. The agent can answer simple conversational prompts directly, and it can delegate documentation lookup to a dedicated knowledge retrieval subagent powered by the local multimodal RAG pipeline.

## Features

- Feishu bot integration over websocket
- Personal Weixin integration over the official iLink bot API
- Deep Agents runtime as the main orchestration layer
- Dedicated retrieval subagent for documentation questions
- Dedicated knowledge-deposit flow for external links, text, and images
- Feishu Wiki and Docs ingestion into Qdrant
- Multimodal RAG pipeline for retrieval, rerank, merge, and answer generation
- Separate chat-model and embedding-model provider configuration
- Optional image OCR / caption indexing pipeline for multimodal content
- Weixin attachment adaptation for text, links, images, and files
- Xiaohongshu post retrieval through a locally running MCP service
- Local `AGENTS.md` memory and `SKILL.md` guidance for answer behavior
- Source-aware answers that can cite the indexed document title or link

## Architecture

The main runtime flow is:

1. Feishu or Weixin delivers an incoming user message to the channel layer
2. `channel/feishu/feishu_channel.py` or `channel/weixin/weixin_channel.py` normalizes the message
3. The channel adapts attachments into a text prompt plus optional local image paths
4. `agent.py` invokes the Deep Agent runtime with a stable `thread_id`
5. The main agent decides whether the request needs knowledge retrieval
6. Retrieval-heavy questions are delegated to the `knowledge_retriever` subagent
7. The retrieval subagent uses the multimodal `RAGQueryPipeline` to prepare context from Qdrant
8. The main agent produces the final reply and the channel sends it back to the user

The indexing flow is:

1. Read configured Wiki node tokens or direct `doc` / `docx` tokens
2. Fetch readable Feishu document content
3. Convert content into markdown chunks
4. Generate embeddings
5. Persist the index in Qdrant
6. Write an index manifest to `data/index_manifest.json`

The knowledge-deposit flow is:

1. Detect whether the user wants to save content into the knowledge base
2. Fetch source material from Xiaohongshu, generic URLs, plain text, or images
3. Normalize the source into a markdown draft
4. Optionally write the final draft into Feishu Docs / Wiki
5. Ingest the final markdown into the local Qdrant index

## Project Layout

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

## Requirements

- Python 3.11+
- `uv`
- Qdrant
- A Feishu self-built app with bot, message, and wiki/doc read permissions
- A Weixin iLink bot account if you want to use the Weixin channel
- One OpenAI-compatible chat model endpoint
- One OpenAI-compatible embedding model endpoint
- Dependencies required by Deep Agents runtime

## Installation

```bash
uv sync
```

Start Qdrant before indexing or querying. For local development, a persistent Docker run is recommended:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

You can also use the included Compose file:

```bash
docker compose up -d qdrant
```

## Configuration

Copy `.env.example` to `.env` and fill in your values.

Required variables:

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

Notes:
- If you use a Qwen embedding endpoint, keep the provider model id lowercase, for example `qwen3-embedding-4b`
- If you start from a Feishu wiki URL such as `https://.../wiki/HjPjwGtbrikafFkN6sdcSe9zn0d`, set `FEISHU_WIKI_ROOT_TOKENS` to the trailing node token: `HjPjwGtbrikafFkN6sdcSe9zn0d`

Optional variables:

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

If you prefer one provider for both chat and embeddings, you can use:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

The example-specific variables still take precedence when present.

If you already have a valid Weixin iLink token, you can also set:

- `WEIXIN_TOKEN`

## Xiaohongshu MCP Setup

This project expects Xiaohongshu post retrieval to come from a locally running MCP service.

Recommended layout:

1. Put the upstream macOS binaries in [tools/xiaohongshu-bin/README.md](/Users/jmdcx/Documents/GitHub/feishu-wiki-rag-agent/tools/xiaohongshu-bin/README.md):
   - `xiaohongshu-mcp-darwin-arm64`
   - `xiaohongshu-login-darwin-arm64`
2. Make them executable:

```bash
chmod +x tools/xiaohongshu-bin/xiaohongshu-*-darwin-arm64
```

3. Log in with the upstream login binary:

```bash
./tools/xiaohongshu-bin/xiaohongshu-login-darwin-arm64
```

4. Start the local MCP service:

```bash
./tools/xiaohongshu-bin/xiaohongshu-mcp-darwin-arm64
```

5. Keep `.env` pointed at the local service:

```env
XHS_MCP_URL=http://127.0.0.1:18060/mcp
```

The local binary approach is the current recommended setup in this repository. The earlier Docker experiment is no longer part of the documented workflow.

## Knowledge Deposit

The agent can now handle requests such as:

- `把这个小红书链接沉淀到知识库`
- `把这段文本沉淀到知识库`
- `把这张图片沉淀到知识库`

When the source is Xiaohongshu, the runtime will call the local MCP service, normalize the note details, and build a structured knowledge draft.

To verify the MCP connection and the local deposit pipeline:

```bash
python scripts/verify_xhs_deposit.py --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

By default this runs in preview mode and does not write to Feishu. To enable full write-back, configure:

- `FEISHU_DEPOSIT_SPACE_ID`
- `FEISHU_DEPOSIT_PARENT_NODE_TOKEN`
- `DEPOSIT_ENABLE_AUTO_WRITE=true`

## Feishu Setup

Your Feishu app should be:

- a self-built enterprise app
- configured to use websocket event delivery
- granted permissions for bot messaging and wiki/doc reading

At minimum, verify:

- bot capability is enabled
- message receive and send permissions are enabled
- wiki/doc read permissions are enabled
- the `接收消息 v2.0` event is subscribed
- the latest app version is published

## Build The Index
Start Docker container:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Run the manual ingestion step before starting the bot:

```bash
uv run python indexer.py
```

This command:

1. Traverses the configured root tokens
2. Downloads supported Feishu Wiki or Doc content
3. Splits documents into chunks
4. Generates embeddings
5. Stores them in Qdrant
6. Writes an index manifest to `data/index_manifest.json`

If you change the embedding model, update `MULTIMODAL_RAG_VECTOR_SIZE` to match the model output dimension before rebuilding the index.

## Quick Local Test

After building the index, you can smoke-test the Deep Agent locally without Feishu:

```bash
uv run python - <<'PY'
from agent import invoke_agent
print(invoke_agent("这个知识库里主要讲了什么？", thread_id="local-smoke-test"))
PY
```

## Run The Bot

```bash
uv run python channel/feishu/feishu_channel.py
```

Then test it in Feishu:

- send a direct message to the bot
- or mention the bot in a group chat

To run the personal Weixin channel instead:

```bash
uv run python channel/weixin/weixin_channel.py
```

Then test it in Weixin:

- send a direct message to the bot assistant created by the official iLink integration
- text messages are passed through directly
- links are fetched with the local docreader before being sent to the agent
- images are passed as local files through the existing `images=[...]` agent interface
- files are parsed locally and sent as extracted markdown plus any extracted images

## Notes

- Only `FEISHU_EVENT_MODE=websocket` is supported in this version
- The main Feishu runtime now uses Deep Agents as the orchestration layer
- Documentation retrieval is delegated to a dedicated `knowledge_retriever` subagent
- This project currently replies with text only, even if image-derived OCR/caption chunks are indexed
- Group chats only trigger a response when the bot is mentioned
- The first Weixin version only supports personal direct chat and text replies
- Weixin voice input, group chat, and media replies are not implemented yet
- Weixin file understanding depends on the local docreader's supported formats
- If you change the embedding model, make sure `MULTIMODAL_RAG_VECTOR_SIZE` matches the model output dimension exactly

## GitHub Upload Checklist

Before pushing this project to GitHub:

- Remove or ignore your real `.env` file and keep only `.env.example`
- Do not commit real API keys, Feishu app secrets, or provider tokens
- Do not commit anything under `data/`; it may contain Qdrant storage, manifests, extracted images, and document-derived content
- Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, or `*.egg-info/`
- Keep `uv.lock` checked in so others can reproduce the environment with `uv sync`
- Re-read `README.md` and make sure the setup steps still match your current provider configuration
- If the repository will be public, verify that your indexed Feishu content is allowed to leave your local machine

## License

MIT
