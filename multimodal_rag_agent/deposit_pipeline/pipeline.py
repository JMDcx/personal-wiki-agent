"""End-to-end knowledge deposit flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.deposit_pipeline.adapters import (
    BaseSourceAdapter,
    DepositSourceError,
    GenericUrlAdapter,
    ImageAdapter,
    PlainTextAdapter,
    XiaohongshuAdapter,
    extract_urls,
)
from multimodal_rag_agent.deposit_pipeline.feishu_writer import FeishuDepositWriter
from multimodal_rag_agent.deposit_pipeline.models import DepositRequest, DepositResult, KnowledgeDraft, SourceMaterial
from multimodal_rag_agent.ingest_pipeline.pipeline import IngestPipeline


class DepositPipeline:
    """Fetch, summarize, write, and index knowledge deposits."""

    def __init__(
        self,
        settings: Settings | None = None,
        multimodal_settings: MultimodalRAGSettings | None = None,
        *,
        adapters: list[BaseSourceAdapter] | None = None,
        writer: FeishuDepositWriter | None = None,
        ingest_pipeline: IngestPipeline | None = None,
        summarizer: Callable[[SourceMaterial, str], KnowledgeDraft] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.multimodal_settings = multimodal_settings or get_multimodal_settings()
        self.adapters = adapters or [
            XiaohongshuAdapter(self.settings),
            GenericUrlAdapter(self.multimodal_settings),
            PlainTextAdapter(),
            ImageAdapter(self.multimodal_settings),
        ]
        self.writer = writer or FeishuDepositWriter(self.settings)
        self.ingest_pipeline = ingest_pipeline or IngestPipeline(self.multimodal_settings)
        self.summarizer = summarizer or self._build_draft
        self.now_provider = now_provider or datetime.now

    def run(self, request: DepositRequest) -> DepositResult:
        normalized = self._normalize_request(request)
        source = self._select_adapter(normalized).fetch(normalized)
        draft = self.summarizer(source, normalized.text)
        final_markdown = self._render_markdown(draft)
        auto_write = self.settings.deposit_enable_auto_write if normalized.auto_write is None else normalized.auto_write
        if not auto_write:
            return DepositResult(
                status="preview",
                message="已生成沉淀草稿预览，未写入飞书或本地索引。",
                draft=draft,
                final_markdown=final_markdown,
                metadata={"source_type": source.source_type, "auto_write": False},
            )

        target_space_id = normalized.target_space_id or self.settings.feishu_deposit_space_id
        target_parent_node_token = normalized.target_parent_node_token or self.settings.feishu_deposit_parent_node_token
        if not target_space_id or not target_parent_node_token:
            raise DepositSourceError("缺少 FEISHU_DEPOSIT_SPACE_ID 或 FEISHU_DEPOSIT_PARENT_NODE_TOKEN，无法执行沉淀写入。")

        write_result = self.writer.write_markdown(
            title=draft.feishu_doc_title,
            markdown_content=final_markdown,
            target_space_id=target_space_id,
            target_parent_node_token=target_parent_node_token,
        )
        ingest_result = self.ingest_pipeline.ingest_markdown(
            final_markdown,
            title=draft.feishu_doc_title,
            metadata={
                **draft.metadata,
                "title": draft.feishu_doc_title,
                "source_uri": write_result.document_url or draft.source_uri,
                "source_type": f"deposit:{draft.source_type}",
                "feishu_doc_token": write_result.document_token,
                "wiki_node_token": write_result.wiki_node_token,
            },
        )
        return DepositResult(
            status="completed",
            message="已完成沉淀，内容已写入飞书并加入本地知识库索引。",
            draft=draft,
            final_markdown=final_markdown,
            local_document_id=ingest_result.document_id,
            feishu_doc_token=write_result.document_token,
            feishu_doc_url=write_result.document_url,
            wiki_node_token=write_result.wiki_node_token,
            metadata={"source_type": source.source_type, "auto_write": True},
        )

    def _normalize_request(self, request: DepositRequest) -> DepositRequest:
        urls = list(request.urls)
        extracted = extract_urls(request.text)
        for url in extracted:
            if url not in urls:
                urls.append(url)
        return DepositRequest(
            text=request.text.strip(),
            urls=urls,
            image_paths=list(request.image_paths),
            target_space_id=request.target_space_id.strip(),
            target_parent_node_token=request.target_parent_node_token.strip(),
            auto_write=request.auto_write,
        )

    def _select_adapter(self, request: DepositRequest) -> BaseSourceAdapter:
        for adapter in self.adapters:
            if adapter.can_handle(request):
                return adapter
        raise DepositSourceError("未识别到可沉淀的链接、文本或图片输入。")

    def _build_draft(self, source: SourceMaterial, user_text: str) -> KnowledgeDraft:
        lines = [line.strip("- ").strip() for line in source.raw_markdown.splitlines() if line.strip()]
        key_points = [line for line in lines if len(line) >= 8][:5]
        if not key_points:
            key_points = [source.title, source.extra_summary or source.raw_markdown[:120].strip()]
        key_points = [point for point in key_points if point][:5]
        summary_sentence = key_points[0] if key_points else source.title
        summary_lines = [
            f"一句话摘要：{summary_sentence}",
            "",
            "关键信息：",
            *[f"- {point}" for point in key_points],
        ]
        tags = self._infer_tags(source, user_text, key_points)
        source_title = source.title or "未命名来源"
        source_type_label = self._source_type_label(source.source_type)
        return KnowledgeDraft(
            source_type=source.source_type,
            source_uri=source.source_uri,
            source_title=source_title,
            author=source.author,
            published_at=source.published_at,
            raw_content_markdown=source.raw_markdown,
            summary_markdown="\n".join(summary_lines).strip(),
            key_points=key_points,
            tags=tags,
            feishu_doc_title=f"[{source_type_label}] {source_title} - 沉淀",
            metadata={**source.metadata, "deposit_generated_at": self.now_provider().isoformat(timespec="seconds")},
        )

    @staticmethod
    def _infer_tags(source: SourceMaterial, user_text: str, key_points: list[str]) -> list[str]:
        bag = " ".join([source.title, user_text, *key_points]).lower()
        tags: list[str] = [source.source_type]
        if "飞书" in bag:
            tags.append("飞书")
        if "小红书" in bag or source.source_type == "xiaohongshu":
            tags.append("小红书")
        if "公众号" in bag or "mp.weixin.qq.com" in source.source_uri:
            tags.append("公众号")
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _render_markdown(self, draft: KnowledgeDraft) -> str:
        deposited_at = self.now_provider().isoformat(timespec="seconds")
        details = "\n".join(f"- {point}" for point in draft.key_points) or "- 无"
        tags = "、".join(draft.tags) or "无"
        return "\n".join(
            [
                f"# {draft.feishu_doc_title}",
                "",
                draft.summary_markdown,
                "",
                "## 详细整理",
                draft.raw_content_markdown.strip() or "无",
                "",
                "## 来源信息",
                f"- 来源类型：{draft.source_type}",
                f"- 原始标题：{draft.source_title}",
                f"- 作者：{draft.author or '未知'}",
                f"- 发布时间：{draft.published_at or '未知'}",
                f"- 原始链接：{draft.source_uri or '无'}",
                "",
                "## 关键信息",
                details,
                "",
                "## 元数据",
                f"- 标签：{tags}",
                f"- 沉淀时间：{deposited_at}",
                "- 沉淀方式：自动沉淀到知识库",
                "",
            ]
        ).strip()

    @staticmethod
    def _source_type_label(source_type: str) -> str:
        return {
            "xiaohongshu": "小红书",
            "url": "链接",
            "text": "文本",
            "image": "图片",
        }.get(source_type, source_type or "来源")
