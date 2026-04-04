# Feishu Wiki RAG Agent

A lightweight, open-source Feishu knowledge assistant built with Deep Agents.

This project connects to Feishu through websocket events, indexes Feishu Wiki and Docs content into a local Chroma vector store, and answers user questions with retrieval-augmented generation. It is designed as a practical starter project for teams that want a Feishu bot with local indexing, provider-flexible model access, and minimal operational setup.

## Features

- Feishu bot integration over websocket
- Feishu Wiki and Docs ingestion into a local vector store
- Deep Agents-based question answering with a dedicated RAG tool
- Separate chat-model and embedding-model provider configuration
- Local `AGENTS.md` memory and `SKILL.md` guidance for answer behavior
- Source-aware answers that can cite the indexed document title or link

## Architecture

The main runtime flow is:

1. Feishu sends a websocket event to the bot
2. `feishu_channel.py` normalizes the incoming message
3. The Deep Agent calls `search_feishu_knowledge`
4. The retriever queries local Chroma data built from Feishu docs
5. The chat model answers based on retrieved context
6. The reply is sent back to Feishu

The indexing flow is:

1. Read configured Wiki node tokens or direct `doc` / `docx` tokens
2. Fetch readable Feishu document content
3. Split content into chunks
4. Generate embeddings
5. Persist the index locally in Chroma

## Project Layout

```txt
feishu_wiki_rag_agent/
├── AGENTS.md
├── LICENSE
├── README.md
├── agent.py
├── config.py
├── feishu_channel.py
├── feishu_client.py
├── indexer.py
├── pyproject.toml
├── retrieval.py
├── schemas.py
├── skills/
│   └── knowledge-qa/
│       └── SKILL.md
├── tests/
│   └── test_feishu_wiki_rag_agent.py
└── .env.example
```

## Requirements

- Python 3.11+
- `uv`
- A Feishu self-built app with bot, message, and wiki/doc read permissions
- One OpenAI-compatible chat model endpoint
- One OpenAI-compatible embedding model endpoint

## Installation

```bash
uv sync
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
FEISHU_RAG_CHROMA_DIR=./data/chroma
FEISHU_RAG_MANIFEST=index_manifest.json
FEISHU_API_BASE=https://open.feishu.cn
FEISHU_REQUEST_TIMEOUT=20
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
uv run python -m feishu_wiki_rag_agent.indexer
```

This command:

1. Traverses the configured root tokens
2. Downloads supported Feishu Wiki or Doc content
3. Splits documents into chunks
4. Generates embeddings
5. Stores them in local Chroma
6. Writes an index manifest to `data/index_manifest.json`

## Run The Bot

```bash
uv run python -m feishu_wiki_rag_agent.feishu_channel
```

Then test it in Feishu:

- send a direct message to the bot
- or mention the bot in a group chat

## Notes

- Only `FEISHU_EVENT_MODE=websocket` is supported in this version
- This project currently replies with text only
- Group chats only trigger a response when the bot is mentioned
- The package name uses underscores because `python -m` module paths cannot reliably use hyphens

## GitHub Upload Checklist

Before pushing this project to GitHub:

- Remove or ignore your real `.env` file and keep only `.env.example`
- Do not commit real API keys, Feishu app secrets, or provider tokens
- Do not commit `data/` if it contains local vector indexes or document-derived content
- Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, or `*.egg-info/`
- Keep `uv.lock` checked in so others can reproduce the environment with `uv sync`
- Re-read `README.md` and make sure the setup steps still match your current provider configuration
- If the repository will be public, verify that your indexed Feishu content is allowed to leave your local machine

## License

MIT
