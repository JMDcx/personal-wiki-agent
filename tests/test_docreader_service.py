from __future__ import annotations

from multimodal_rag_agent.docreader_service.client import DocreaderService, DocreaderUnavailableError
from multimodal_rag_agent.docreader_service.schemas import ParseRequest


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        return None


def test_parse_url_falls_back_when_docreader_import_fails(monkeypatch):
    service = DocreaderService()
    html = """
    <html>
      <head>
        <meta property="og:title" content="测试公众号文章" />
      </head>
      <body>
        <h1 id="activity-name">测试公众号文章</h1>
        <span id="js_name">知识库助手</span>
        <em id="publish_time">2026-04-19</em>
        <div id="js_content">
          <p>第一段内容。</p>
          <p>第二段内容。</p>
        </div>
        <script type="text/javascript">var ct = "1776556800";</script>
      </body>
    </html>
    """

    def _fail_new_parser():
        raise DocreaderUnavailableError("missing optional dependency")

    monkeypatch.setattr(service, "_new_parser", _fail_new_parser)
    monkeypatch.setattr("multimodal_rag_agent.docreader_service.client.requests.get", lambda *args, **kwargs: _FakeResponse(html))

    parsed = service.parse(ParseRequest(url="https://mp.weixin.qq.com/s/example", title=""))

    assert "测试公众号文章" in parsed.markdown_content
    assert "第一段内容。" in parsed.markdown_content
    assert parsed.metadata["title"] == "测试公众号文章"
    assert parsed.metadata["author"] == "知识库助手"
    assert parsed.metadata["parser_backend"] == "requests_html_fallback"
    assert "DocreaderUnavailableError" in parsed.metadata["fallback_reason"]


def test_extract_wechat_publish_time_prefers_rendered_text():
    service = DocreaderService()
    html = '<em id="publish_time">2026-04-19</em><script>var ct = "1776556800";</script>'

    assert service._extract_wechat_published_at(html) == "2026-04-19"
