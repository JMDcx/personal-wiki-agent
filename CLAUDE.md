# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
uv sync

# Start Qdrant (required for indexing and querying)
docker compose up -d qdrant
# or
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/data/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Build the Feishu Wiki/Docs index
uv run python indexer.py

# Run Feishu bot channel
uv run python channel/feishu/feishu_channel.py

# Run Weixin bot channel
uv run python channel/weixin/weixin_channel.py

# Quick local smoke test
uv run python - <<'PY'
from agent import invoke_agent
print(invoke_agent("这个知识库里主要讲了什么？", thread_id="local-smoke-test"))
PY

# Verify Xiaohongshu deposit pipeline
python scripts/verify_xhs_deposit.py --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"

# Run tests
pytest
```

## Architecture Overview

This is a Feishu knowledge assistant built on **Deep Agents** runtime with a **multimodal RAG pipeline** backed by Qdrant.

### Core Runtime Flow

1. **Message Reception**: Feishu (websocket) or Weixin (iLink API) delivers incoming messages
2. **Channel Normalization**: `channel/feishu/feishu_channel.py` or `channel/weixin/weixin_channel.py` normalizes messages into a common `MessageContext` protocol
3. **Agent Invocation**: `agent.py` invokes the Deep Agent runtime with stable `thread_id` for conversation continuity
4. **Intent Classification**: Main agent decides if the request needs knowledge retrieval
5. **Subagent Delegation**: Retrieval-heavy questions delegate to `knowledge_retriever` subagent
6. **RAG Pipeline**: The `RAGQueryPipeline` (query understand → retrieval → rerank → merge → generate) prepares context from Qdrant
7. **Response Generation**: Main agent produces final reply with source citations
8. **Channel Response**: Channel sends reply back to the user

### Key Components

- **`agent.py`**: Main entry point using Deep Agents runtime with tool-based delegation
- **`schemas.py`**: Protocol objects (`MessageContext`, `IncomingMessage`, `IndexManifest`) used across channels
- **`config.py`**: Settings with environment variable loading and directory management
- **`AGENTS.md`**: Agent behavior guidance loaded at runtime
- **`skills/`**: Local SKILL.md files for knowledge-qa and knowledge-deposit behaviors
- **`multimodal_rag_agent/`**: Modular RAG pipeline with separate stages for query understanding, retrieval, reranking, and generation
- **`observability/`**: Structured logging system with JSONL file output and context-aware event emission
- **`protocols/`**: Shared protocol models for controller decisions and tool request/response

### Knowledge Deposit Flow

The agent can deposit external content (Xiaohongshu posts, URLs, text, images) into the knowledge base:

1. Detect deposit intent from user message
2. Fetch source content via dedicated adapters (Xiaohongshu MCP, generic URLs, OCR/caption for images)
3. Generate structured markdown draft with summary and metadata
4. Write to Feishu Docs/Wiki when `DEPOSIT_ENABLE_AUTO_WRITE=true`
5. Ingest final markdown into local Qdrant index
6. Return confirmation with Feishu doc link

### Configuration Patterns

- All settings use `config.py::Settings` with `.env` file loading
- Two-tier model configuration: base `FEISHU_RAG_*` variables with optional `MULTIMODAL_RAG_*` stage-specific overrides
- Chat and embedding models can use different providers
- Vector size must match embedding model dimensions exactly
- Logging levels are configurable per subsystem (httpx, openai, lark, etc.)

### Channel Architecture

Channels implement a common normalization pattern:

1. Raw message → `IncomingMessage` with metadata (mentions, replies, attachments)
2. Attachment processing → text + local image paths
3. `MessageContext` protocol object → passed to agent layer
4. Agent response → channel-specific send method

### Observability System

- Structured JSONL logging to `data/logs/app.jsonl` with rotation
- Context-aware logging via `observability/context.py` (bind request state, record timing)
- Event emission via `observability/events.py` (log_event, log_exception, preview_text)
- Deep Agents middleware in `observability/deepagents_middleware.py` for automatic instrumentation
- Configurable preview length and redaction for sensitive data

### Test Patterns

- Tests use pytest and are in `tests/`
- Test files demonstrate agent controller flow, query understanding, and channel message handling
- SQLite checkpoint database persists conversation state across sessions

### Important Constraints

- Only `FEISHU_EVENT_MODE=websocket` is supported
- Group chats only trigger responses when bot is mentioned
- Weixin channel currently supports personal chat only (text replies)
- If changing embedding models, update `MULTIMODAL_RAG_VECTOR_SIZE` before rebuilding index
- Qdrant must be running before indexing or querying
- Feishu app must have bot, message, wiki/doc permissions with `接收消息 v2.0` event subscribed

### Lark CLI Integration

The recommended Feishu write-back backend is `lark_cli`:

1. Install: `npm install -g @larksuite/cli`
2. Add skills: `npx skills add https://github.com/larksuite/cli -y -g`
3. Initialize profile with app credentials from `.env`
4. Configure `.env`: `FEISHU_DEPOSIT_WRITE_BACKEND=lark_cli`, `FEISHU_LARK_CLI_PROFILE=feishu-wiki-rag-agent`
5. Verify with direct doc creation before running full deposit flow

### Xiaohongshu MCP Setup

Local MCP binaries are expected in `tools/xiaohongshu-bin/`:

- `xiaohongshu-mcp-darwin-arm64` - MCP server
- `xiaohongshu-login-darwin-arm64` - Login utility

Configure `.env` with: `XHS_MCP_URL=http://127.0.0.1:18060/mcp`

### Data Directory Structure

- `data/qdrant_storage/` - Qdrant persistent storage
- `data/index_manifest.json` - Index metadata after build
- `data/deepagents/checkpoints.sqlite` - Conversation state persistence
- `data/logs/app.jsonl` - Structured application logs
- `data/weixin/` - Weixin credentials and temporary files
