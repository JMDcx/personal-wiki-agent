"""Bounded message dispatching with optional per-thread serialization."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar

try:
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from observability.events import log_event, log_exception


T = TypeVar("T")


@dataclass(frozen=True)
class DispatchContext:
    """Timing and routing context for one accepted dispatch task."""

    thread_id: str
    queue_ms: float


@dataclass(frozen=True)
class DispatchResult:
    """Result returned immediately when a task is submitted."""

    accepted: bool
    future: Future[Any] | None = None
    rejected_reason: str = ""


class ConcurrentMessageDispatcher:
    """Run message handlers in a bounded thread pool.

    Capacity is capped at ``max_workers + queue_size`` because Python's
    ThreadPoolExecutor uses an unbounded internal queue.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        queue_size: int = 32,
        per_thread_serial: bool = True,
        thread_name_prefix: str = "message-dispatcher",
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.queue_size = max(0, int(queue_size))
        self.per_thread_serial = per_thread_serial
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._capacity = threading.BoundedSemaphore(self.max_workers + self.queue_size)
        self._lock = threading.Lock()
        self._closed = False
        self._thread_tails: dict[str, threading.Event] = {}

    def submit(
        self,
        thread_id: str,
        fn: Callable[..., T],
        *args: Any,
        on_rejected: Callable[[], None] | None = None,
        on_started: Callable[[DispatchContext], None] | None = None,
        metadata: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> DispatchResult:
        """Submit a task and return immediately.

        ``on_started`` runs inside the worker after any same-thread predecessor
        has completed, so its ``queue_ms`` includes executor and serial wait time.
        """
        fields = dict(metadata or {})
        normalized_thread_id = str(thread_id or "default")
        submitted_at = perf_counter()
        if not self._reserve_capacity():
            self._reject(
                normalized_thread_id,
                fields,
                reason="queue_full",
                on_rejected=on_rejected,
            )
            return DispatchResult(accepted=False, rejected_reason="queue_full")

        predecessor: threading.Event | None = None
        current_done: threading.Event | None = None
        if self.per_thread_serial:
            predecessor, current_done = self._append_thread_tail(normalized_thread_id)

        log_event(
            "dispatch_submitted",
            thread_id=normalized_thread_id,
            queue_size=self.queue_size,
            max_workers=self.max_workers,
            **fields,
        )

        def _run() -> T:
            if predecessor is not None:
                predecessor.wait()
            queue_ms = (perf_counter() - submitted_at) * 1000
            context = DispatchContext(thread_id=normalized_thread_id, queue_ms=round(queue_ms, 1))
            if on_started is not None:
                on_started(context)
            worker_started_at = perf_counter()
            log_event("dispatch_started", thread_id=normalized_thread_id, queue_ms=context.queue_ms, **fields)
            try:
                result = fn(*args, **kwargs)
                worker_ms = (perf_counter() - worker_started_at) * 1000
                log_event(
                    "dispatch_completed",
                    thread_id=normalized_thread_id,
                    queue_ms=context.queue_ms,
                    worker_ms=round(worker_ms, 1),
                    **fields,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                worker_ms = (perf_counter() - worker_started_at) * 1000
                log_exception(
                    "dispatch_failed",
                    exc,
                    stage="message_dispatch",
                    thread_id=normalized_thread_id,
                    queue_ms=context.queue_ms,
                    worker_ms=round(worker_ms, 1),
                    **fields,
                )
                raise
            finally:
                if current_done is not None:
                    current_done.set()
                    self._clear_thread_tail(normalized_thread_id, current_done)
                self._capacity.release()

        try:
            future = self._executor.submit(_run)
        except RuntimeError:
            if current_done is not None:
                current_done.set()
                self._clear_thread_tail(normalized_thread_id, current_done)
            self._capacity.release()
            self._reject(
                normalized_thread_id,
                fields,
                reason="dispatcher_closed",
                on_rejected=on_rejected,
            )
            return DispatchResult(accepted=False, rejected_reason="dispatcher_closed")

        return DispatchResult(accepted=True, future=future)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shut down the underlying executor."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _reserve_capacity(self) -> bool:
        with self._lock:
            if self._closed:
                return False
        return self._capacity.acquire(blocking=False)

    def _append_thread_tail(self, thread_id: str) -> tuple[threading.Event | None, threading.Event]:
        with self._lock:
            predecessor = self._thread_tails.get(thread_id)
            current_done = threading.Event()
            self._thread_tails[thread_id] = current_done
            return predecessor, current_done

    def _clear_thread_tail(self, thread_id: str, done_event: threading.Event) -> None:
        with self._lock:
            if self._thread_tails.get(thread_id) is done_event:
                self._thread_tails.pop(thread_id, None)

    @staticmethod
    def _reject(
        thread_id: str,
        fields: dict[str, object],
        *,
        reason: str,
        on_rejected: Callable[[], None] | None,
    ) -> None:
        log_event(
            "dispatch_rejected",
            level=logging.WARNING,
            thread_id=thread_id,
            reason=reason,
            **fields,
        )
        if on_rejected is None:
            return
        try:
            on_rejected()
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "dispatch_rejection_reply_failed",
                exc,
                stage="message_dispatch_rejection",
                thread_id=thread_id,
                reason=reason,
                **fields,
            )
