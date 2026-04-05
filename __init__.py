"""Feishu Wiki RAG Deep Agent"""

from __future__ import annotations

from typing import Any


def build_agent(*args: Any, **kwargs: Any) -> Any:
    try:
        from feishu_wiki_rag_agent.agent import build_agent as _build_agent
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from agent import build_agent as _build_agent
    return _build_agent(*args, **kwargs)


def invoke_agent(*args: Any, **kwargs: Any) -> Any:
    try:
        from feishu_wiki_rag_agent.agent import invoke_agent as _invoke_agent
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from agent import invoke_agent as _invoke_agent
    return _invoke_agent(*args, **kwargs)


def reset_thread(*args: Any, **kwargs: Any) -> Any:
    try:
        from feishu_wiki_rag_agent.agent import reset_thread as _reset_thread
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from agent import reset_thread as _reset_thread
    return _reset_thread(*args, **kwargs)


def list_thread_history(*args: Any, **kwargs: Any) -> Any:
    try:
        from feishu_wiki_rag_agent.agent import list_thread_history as _list_thread_history
    except ModuleNotFoundError:  # pragma: no cover - source tree fallback
        from agent import list_thread_history as _list_thread_history
    return _list_thread_history(*args, **kwargs)


__all__ = ["build_agent", "invoke_agent", "reset_thread", "list_thread_history"]
