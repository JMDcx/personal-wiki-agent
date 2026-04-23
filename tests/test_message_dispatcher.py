from __future__ import annotations

import threading
import time

from channel.dispatcher import ConcurrentMessageDispatcher


def test_dispatcher_runs_different_threads_concurrently():
    dispatcher = ConcurrentMessageDispatcher(max_workers=2, queue_size=2)
    start_gate = threading.Event()
    running: list[str] = []
    running_lock = threading.Lock()

    def _task(name: str) -> str:
        with running_lock:
            running.append(name)
            if len(running) == 2:
                start_gate.set()
        start_gate.wait(timeout=1)
        time.sleep(0.05)
        return name

    first = dispatcher.submit("thread-a", _task, "a")
    second = dispatcher.submit("thread-b", _task, "b")

    try:
        assert first.accepted
        assert second.accepted
        assert first.future is not None
        assert second.future is not None
        assert start_gate.wait(timeout=0.5)
        assert first.future.result(timeout=1) == "a"
        assert second.future.result(timeout=1) == "b"
    finally:
        dispatcher.shutdown(wait=True)


def test_dispatcher_serializes_same_thread_in_submission_order():
    dispatcher = ConcurrentMessageDispatcher(max_workers=3, queue_size=4)
    order: list[str] = []
    lock = threading.Lock()

    def _task(name: str) -> str:
        time.sleep(0.02 if name == "first" else 0)
        with lock:
            order.append(name)
        return name

    first = dispatcher.submit("same-thread", _task, "first")
    second = dispatcher.submit("same-thread", _task, "second")
    third = dispatcher.submit("same-thread", _task, "third")

    try:
        assert first.future is not None
        assert second.future is not None
        assert third.future is not None
        assert first.future.result(timeout=1) == "first"
        assert second.future.result(timeout=1) == "second"
        assert third.future.result(timeout=1) == "third"
        assert order == ["first", "second", "third"]
    finally:
        dispatcher.shutdown(wait=True)


def test_dispatcher_rejects_when_capacity_is_full():
    dispatcher = ConcurrentMessageDispatcher(max_workers=1, queue_size=0)
    release = threading.Event()
    rejected: list[str] = []

    def _slow_task() -> str:
        release.wait(timeout=1)
        return "done"

    first = dispatcher.submit("thread-a", _slow_task)
    second = dispatcher.submit("thread-b", lambda: "rejected", on_rejected=lambda: rejected.append("busy"))

    try:
        assert first.accepted
        assert not second.accepted
        assert second.rejected_reason == "queue_full"
        assert rejected == ["busy"]
    finally:
        release.set()
        dispatcher.shutdown(wait=True)


def test_dispatcher_records_exception_and_continues_same_thread():
    dispatcher = ConcurrentMessageDispatcher(max_workers=1, queue_size=2)
    order: list[str] = []

    def _failing_task() -> None:
        order.append("failed")
        raise RuntimeError("boom")

    def _next_task() -> str:
        order.append("next")
        return "ok"

    first = dispatcher.submit("same-thread", _failing_task)
    second = dispatcher.submit("same-thread", _next_task)

    try:
        assert first.future is not None
        assert second.future is not None
        assert isinstance(first.future.exception(timeout=1), RuntimeError)
        assert second.future.result(timeout=1) == "ok"
        assert order == ["failed", "next"]
    finally:
        dispatcher.shutdown(wait=True)
