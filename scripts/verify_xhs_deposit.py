#!/usr/bin/env python3
"""Verify local xiaohongshu-mcp connectivity and local deposit integration.

Usage:
  python scripts/verify_xhs_deposit.py --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"

Default behavior uses preview mode so Feishu write configuration is not required.
Pass --write to attempt the full Feishu write + local ingest flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from multimodal_rag_agent.deposit_pipeline.adapters import DepositSourceError  # noqa: E402
from multimodal_rag_agent.deposit_pipeline.models import DepositRequest  # noqa: E402
from multimodal_rag_agent.deposit_pipeline.pipeline import DepositPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify local xiaohongshu-mcp and deposit pipeline integration.")
    parser.add_argument("--url", required=True, help="A Xiaohongshu post URL containing xsec_token.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Attempt the full Feishu write path. Without this flag the script runs in preview mode.",
    )
    parser.add_argument(
        "--mcp-url",
        default="",
        help="Override XHS MCP endpoint. Defaults to XHS_MCP_URL or http://127.0.0.1:18060/mcp.",
    )
    return parser


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def post_json_with_session(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    session_id: str = "",
) -> tuple[dict[str, Any], requests.Response]:
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    response = session.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    if not response.content.strip():
        return {}, response
    return response.json(), response


def extract_feed_and_token(url: str) -> tuple[str, str]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    xsec_token = (query.get("xsec_token") or [""])[0].strip()
    path_parts = [part for part in parsed.path.split("/") if part]
    feed_id = ""
    if "explore" in path_parts:
        index = path_parts.index("explore")
        if index + 1 < len(path_parts):
            feed_id = path_parts[index + 1].strip()
    if not feed_id:
        for candidate in reversed(path_parts):
            if len(candidate) >= 8:
                feed_id = candidate.strip()
                break
    return feed_id, xsec_token


def preview_text(value: str, limit: int = 280) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    mcp_url = args.mcp_url or settings.xhs_mcp_url
    feed_id, xsec_token = extract_feed_and_token(args.url)
    if not feed_id or not xsec_token:
        print("错误：URL 中缺少 feed_id 或 xsec_token。", file=sys.stderr)
        return 2

    print(f"[1/3] 检查 MCP 初始化：{mcp_url}")
    session = requests.Session()
    initialize_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "feishu-wiki-rag-agent", "version": "0.1.0"},
        },
    }
    init_response, init_http_response = post_json_with_session(session, mcp_url, initialize_payload)
    print(json.dumps(init_response, ensure_ascii=False, indent=2))
    session_id = init_http_response.headers.get("Mcp-Session-Id", "").strip() or init_http_response.headers.get("mcp-session-id", "").strip()
    if session_id:
        print(f"检测到 MCP session: {session_id}")

    initialized_payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    post_json_with_session(session, mcp_url, initialized_payload, session_id=session_id)
    print("初始化通知已发送。")

    print("\n[2/3] 调用 get_feed_detail")
    detail_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_feed_detail",
            "arguments": {"feed_id": feed_id, "xsec_token": xsec_token},
        },
    }
    detail_response, _ = post_json_with_session(session, mcp_url, detail_payload, session_id=session_id)
    if "error" in detail_response:
        print(json.dumps(detail_response, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    result = detail_response.get("result", {})
    if isinstance(result, dict) and result.get("isError"):
        print(json.dumps(detail_response, ensure_ascii=False, indent=2), file=sys.stderr)
        print(
            "\n提示：xiaohongshu-mcp 上游 README 明确要求 `get_feed_detail` 使用从 `list_feeds` 或 `search_feeds` 返回结果里拿到的 "
            "`feed_id` 和 `xsec_token`。直接复制浏览器 URL 中的参数，未必总能命中详情。",
            file=sys.stderr,
        )
        return 4
    print(json.dumps(detail_response, ensure_ascii=False, indent=2)[:2000])

    print("\n[3/3] 运行本地 DepositPipeline")
    pipeline = DepositPipeline()
    request = DepositRequest(
        text=f"请把这个小红书链接沉淀到知识库：{args.url}",
        urls=[args.url],
        auto_write=args.write,
    )
    try:
        result = pipeline.run(request)
    except DepositSourceError as exc:
        print(f"DepositPipeline 失败：{exc}", file=sys.stderr)
        return 5
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2)[:4000])

    print("\n结果摘要")
    print(f"- deposit status: {result.status}")
    print(f"- source type: {result.draft.source_type}")
    print(f"- draft title: {result.draft.feishu_doc_title}")
    print(f"- summary preview: {preview_text(result.draft.summary_markdown)}")
    if result.feishu_doc_url:
        print(f"- feishu doc: {result.feishu_doc_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
