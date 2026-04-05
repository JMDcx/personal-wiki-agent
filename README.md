# Feishu Wiki RAG Agent

English | [简体中文](./README.zh-CN.md)

A lightweight Feishu knowledge assistant built on Deep Agents and backed by a Qdrant-based multimodal RAG pipeline.

This project connects to Feishu through websocket events, indexes Feishu Wiki and Docs content into Qdrant, and answers user questions through a Deep Agent runtime. The agent can answer simple conversational prompts directly, and it can delegate documentation lookup to a dedicated knowledge retrieval subagent powered by the local multimodal RAG pipeline.

## Features

- Feishu bot integration over websocket
- Deep Agents runtime as the main orchestration layer
- Dedicated retrieval subagent for documentation questions
- Feishu Wiki and Docs ingestion into Qdrant
- Multimodal RAG pipeline for retrieval, rerank, merge, and answer generation
- Separate chat-model and embedding-model provider configuration
- Optional image OCR / caption indexing pipeline for multimodal content
- Local `AGENTS.md` memory and `SKILL.md` guidance for answer behavior
- Source-aware answers that can cite the indexed document title or link

## Architecture

The main runtime flow is:

1. Feishu sends a websocket event to the bot
2. `channel/feishu/feishu_channel.py` normalizes the incoming message
3. `agent.py` invokes the Deep Agent runtime with a stable `thread_id`
4. The main agent decides whether the request needs knowledge retrieval
5. Retrieval-heavy questions are delegated to the `knowledge_retriever` subagent
6. The retrieval subagent uses the multimodal `RAGQueryPipeline` to prepare context from Qdrant
7. The main agent produces the final reply and sends it back to Feishu
6. The reply is sent back to Feishu

The indexing flow is:

1. Read configured Wiki node tokens or direct `doc` / `docx` tokens
2. Fetch readable Feishu document content
3. Convert content into markdown chunks
4. Generate embeddings
5. Persist the index in Qdrant
6. Write an index manifest to `data/index_manifest.json`

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
├── skills/
│   └── knowledge-qa/
│       └── SKILL.md
├── tests/
│   ├── test_feishu_wiki_rag_agent.py
│   └── test_multimodal_rag_agent.py
└── .env.example
```

## Requirements

- Python 3.11+
- `uv`
- Qdrant
- A Feishu self-built app with bot, message, and wiki/doc read permissions
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

Optional variables:

```env
FEISHU_RAG_TOP_K=4
FEISHU_RAG_COLLECTION=feishu_wiki_docs
FEISHU_RAG_DATA_DIR=./data
FEISHU_RAG_MANIFEST=index_manifest.json
FEISHU_API_BASE=https://open.feishu.cn
FEISHU_REQUEST_TIMEOUT=20

MULTIMODAL_RAG_QDRANT_URL=http://127.0.0.1:6333
MULTIMODAL_RAG_QDRANT_API_KEY=
MULTIMODAL_RAG_QDRANT_COLLECTION=feishu_wiki_docs
MULTIMODAL_RAG_VECTOR_SIZE=1536
MULTIMODAL_RAG_TOP_K=6
MULTIMODAL_RAG_RERANK_TOP_K=4
MULTIMODAL_RAG_CHUNK_SIZE=512
MULTIMODAL_RAG_CHUNK_OVERLAP=128
```

If you prefer one provider for both chat and embeddings, you can use:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

The example-specific variables still take precedence when present.

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

## Notes

- Only `FEISHU_EVENT_MODE=websocket` is supported in this version
- The main Feishu runtime now uses Deep Agents as the orchestration layer
- Documentation retrieval is delegated to a dedicated `knowledge_retriever` subagent
- This project currently replies with text only, even if image-derived OCR/caption chunks are indexed
- Group chats only trigger a response when the bot is mentioned
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
