from __future__ import annotations

from agent import invoke_agent
from protocols.streaming import StreamEvent


def test_invoke_agent_can_aggregate_streaming_final(monkeypatch):
    def _fake_stream(*args, **kwargs):  # noqa: ANN002, ANN003
        yield StreamEvent.started("开始处理")
        yield StreamEvent.text_delta("无关增量")
        yield StreamEvent.final("最终答案")

    monkeypatch.setattr("agent.invoke_agent_stream", _fake_stream)

    assert invoke_agent("问题", thread_id="test-stream-aggregate") == "最终答案"
