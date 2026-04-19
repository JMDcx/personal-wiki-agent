from __future__ import annotations

import pytest

from multimodal_rag_agent.deposit_pipeline.adapters import (
    DepositSourceError,
    GenericUrlAdapter,
    ProvidedContentAdapter,
    extract_urls,
)
from multimodal_rag_agent.deposit_pipeline.models import DepositRequest
from multimodal_rag_agent.deposit_pipeline.pipeline import DepositPipeline
from multimodal_rag_agent.models import ParsedDocument
from protocols.tool_models import DepositRequestContext


class _FailingDocreader:
    def parse(self, request):  # noqa: ANN001
        raise AssertionError(f"docreader should not be called for {request.url}")


class _StaticDocreader:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document
        self.calls: list[str] = []

    def parse(self, request):  # noqa: ANN001
        self.calls.append(request.url or "")
        return self.document


def test_deposit_request_context_extracts_explicit_source_and_content():
    context = DepositRequestContext.from_inputs(
        text=(
            "将微信公众号链接 https://mp.weixin.qq.com/s/example 沉淀到飞书知识库\n"
            "## 来源链接 https://mp.weixin.qq.com/s/example)\n"
            "## 原文提取内容（已提供，无需再抓取）\n"
            "# 正确标题\n\n正文第一段。"
        )
    )

    assert context.source_url == "https://mp.weixin.qq.com/s/example"
    assert context.urls == ["https://mp.weixin.qq.com/s/example"]
    assert context.provided_content.startswith("# 正确标题")


def test_deposit_request_context_treats_text_plus_source_url_as_provided_content():
    context = DepositRequestContext.from_inputs(
        text="# 正文标题\n\n正文第一段。",
        source_url="https://mp.weixin.qq.com/s/example",
    )

    assert context.source_url == "https://mp.weixin.qq.com/s/example"
    assert context.urls == ["https://mp.weixin.qq.com/s/example"]
    assert context.provided_content == "# 正文标题\n\n正文第一段。"


def test_extract_urls_normalizes_trailing_punctuation():
    urls = extract_urls("请沉淀这个链接 [https://mp.weixin.qq.com/s/example)]，谢谢")

    assert urls == ["https://mp.weixin.qq.com/s/example"]


def test_provided_content_is_used_before_refetching_url():
    pipeline = DepositPipeline(
        adapters=[
            ProvidedContentAdapter(),
            GenericUrlAdapter(docreader=_FailingDocreader()),
        ]
    )
    request = DepositRequest(
        text="将链接沉淀到知识库",
        urls=["https://mp.weixin.qq.com/s/example)"],
        provided_content="# 无需向量库！阿里开源Sirchmunk\n\n正文第一段。",
        auto_write=False,
    )

    result = pipeline.run(request)

    assert result.status == "preview"
    assert result.draft.source_type == "url"
    assert result.draft.source_title == "无需向量库！阿里开源Sirchmunk"
    assert result.draft.source_uri == "https://mp.weixin.qq.com/s/example"
    assert result.draft.metadata["content_origin"] == "provided_content"


def test_url_only_request_still_uses_generic_url_adapter():
    docreader = _StaticDocreader(
        ParsedDocument(
            markdown_content="# URL 标题\n\n正文内容。",
            metadata={"title": "URL 标题", "author": "作者A"},
        )
    )
    adapter = GenericUrlAdapter(docreader=docreader)

    source = adapter.fetch(DepositRequest(urls=["https://sspai.com/post/42380#"]))

    assert docreader.calls == ["https://sspai.com/post/42380#"]
    assert source.title == "URL 标题"
    assert source.metadata["content_origin"] == "fetched_url"


def test_invalid_wechat_shell_page_is_rejected():
    docreader = _StaticDocreader(
        ParsedDocument(
            markdown_content=(
                "参数错误：，，，，\n"
                "视频小程序赞，轻点两下取消赞\n"
                "在看，轻点两下取消在看\n"
            ),
            metadata={},
        )
    )
    adapter = GenericUrlAdapter(docreader=docreader)

    with pytest.raises(DepositSourceError):
        adapter.fetch(DepositRequest(urls=["https://mp.weixin.qq.com/s/bad-shell"]))
