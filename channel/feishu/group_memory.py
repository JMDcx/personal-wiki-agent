"""Lightweight per-group memory for Feishu group chats."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, log_exception


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class FeishuGroupMemoryStore:
    """Append-only JSONL memory scoped by Feishu group chat id."""

    def __init__(self, root_dir: Path, *, max_recent_turns: int = 6) -> None:
        self.root_dir = Path(root_dir)
        self.max_recent_turns = max(0, int(max_recent_turns))
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._global_file_lock = threading.Lock()

    def recent_turns(self, chat_id: str, limit: int | None = None) -> list[dict[str, object]]:
        """Return the most recent completed turns for one group chat."""
        normalized_chat_id = str(chat_id or "").strip()
        if not normalized_chat_id:
            return []
        resolved_limit = self.max_recent_turns if limit is None else max(0, int(limit))
        if resolved_limit <= 0:
            return []
        lock = self._lock_for_chat(normalized_chat_id)
        path = self._path_for_chat(normalized_chat_id)
        try:
            with self._global_file_lock, lock:
                if not path.exists():
                    return []
                turns: list[dict[str, object]] = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    parsed = self._parse_line(line)
                    if parsed is not None:
                        turns.append(parsed)
                return turns[-resolved_limit:]
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "group_memory_read_failed",
                exc,
                level=logging.WARNING,
                channel="feishu",
                chat_id=normalized_chat_id,
            )
            return []

    def append_turn(
        self,
        *,
        chat_id: str,
        sender_open_id: str,
        message_id: str,
        question: str,
        answer: str,
    ) -> None:
        """Append one completed user/assistant turn for later group context."""
        normalized_chat_id = str(chat_id or "").strip()
        if not normalized_chat_id:
            return
        record = {
            "chat_id": normalized_chat_id,
            "sender_open_id": str(sender_open_id or "").strip(),
            "message_id": str(message_id or "").strip(),
            "question": str(question or "").strip(),
            "answer": str(answer or "").strip(),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        if not record["question"] or not record["answer"]:
            return
        lock = self._lock_for_chat(normalized_chat_id)
        path = self._path_for_chat(normalized_chat_id)
        try:
            with self._global_file_lock, lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_event(
                "group_memory_turn_appended",
                channel="feishu",
                chat_id=normalized_chat_id,
                message_id=record["message_id"],
                sender_open_id=record["sender_open_id"],
            )
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "group_memory_append_failed",
                exc,
                level=logging.WARNING,
                channel="feishu",
                chat_id=normalized_chat_id,
                message_id=record["message_id"],
            )

    def compact_all_to_recent(self, limit: int | None = None) -> dict[str, int]:
        """Rewrite all group memory files so only recent turns are retained."""
        resolved_limit = self.max_recent_turns if limit is None else max(0, int(limit))
        stats = {"file_count": 0, "kept_turn_count": 0, "removed_turn_count": 0}
        if not self.root_dir.exists():
            return stats
        try:
            with self._global_file_lock:
                for path in sorted(self.root_dir.glob("*.jsonl")):
                    if not path.is_file():
                        continue
                    turns: list[dict[str, object]] = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        parsed = self._parse_line(line)
                        if parsed is not None:
                            turns.append(parsed)
                    kept_turns = turns[-resolved_limit:] if resolved_limit > 0 else []
                    if kept_turns:
                        path.write_text(
                            "".join(json.dumps(turn, ensure_ascii=False) + "\n" for turn in kept_turns),
                            encoding="utf-8",
                        )
                    else:
                        path.unlink(missing_ok=True)
                    stats["file_count"] += 1
                    stats["kept_turn_count"] += len(kept_turns)
                    stats["removed_turn_count"] += max(0, len(turns) - len(kept_turns))
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "group_memory_compact_failed",
                exc,
                level=logging.WARNING,
                channel="feishu",
            )
        return stats

    def _path_for_chat(self, chat_id: str) -> Path:
        safe_chat_id = _SAFE_ID_RE.sub("_", chat_id).strip("._") or "unknown"
        return self.root_dir / f"{safe_chat_id}.jsonl"

    def _lock_for_chat(self, chat_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(chat_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[chat_id] = lock
            return lock

    @staticmethod
    def _parse_line(line: str) -> dict[str, object] | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            parsed: Any = json.loads(stripped)
        except json.JSONDecodeError:
            log_event("group_memory_line_skipped", level=logging.WARNING, reason="invalid_json")
            return None
        if not isinstance(parsed, dict):
            log_event("group_memory_line_skipped", level=logging.WARNING, reason="non_object")
            return None
        question = str(parsed.get("question", "")).strip()
        answer = str(parsed.get("answer", "")).strip()
        if not question or not answer:
            log_event("group_memory_line_skipped", level=logging.WARNING, reason="missing_turn_text")
            return None
        return {
            "chat_id": str(parsed.get("chat_id", "")).strip(),
            "sender_open_id": str(parsed.get("sender_open_id", "")).strip(),
            "message_id": str(parsed.get("message_id", "")).strip(),
            "question": question,
            "answer": answer,
            "created_at": str(parsed.get("created_at", "")).strip(),
        }
