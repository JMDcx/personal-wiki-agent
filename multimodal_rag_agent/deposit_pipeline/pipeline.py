"""End-to-end knowledge deposit flow."""

from __future__ import annotations

from datetime import datetime
import re
from time import perf_counter
from typing import Callable

try:
    from feishu_wiki_rag_agent.config import Settings, get_settings
    from feishu_wiki_rag_agent.observability.events import log_event, log_exception, preview_text
except ModuleNotFoundError:  # pragma: no cover - source tree fallback
    from config import Settings, get_settings
    from observability.events import log_event, log_exception, preview_text

from multimodal_rag_agent.config import MultimodalRAGSettings, get_multimodal_settings
from multimodal_rag_agent.deposit_pipeline.adapters import (
    BaseSourceAdapter,
    DepositSourceError,
    GenericUrlAdapter,
    ImageAdapter,
    PlainTextAdapter,
    ProvidedContentAdapter,
    XiaohongshuAdapter,
    extract_urls,
    normalize_url,
)
from multimodal_rag_agent.deposit_pipeline.feishu_writer import FeishuDepositWriter
from multimodal_rag_agent.deposit_pipeline.models import (
    DepositRequest,
    DepositResult,
    InlineImage,
    KnowledgeDraft,
    SourceMaterial,
)
from multimodal_rag_agent.ingest_pipeline.pipeline import IngestPipeline


MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)\n]+)\)")
INLINE_IMAGE_PLACEHOLDER_TEMPLATE = "[[IMG_{order:03d}]]"


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
            ProvidedContentAdapter(),
            GenericUrlAdapter(self.multimodal_settings),
            PlainTextAdapter(),
            ImageAdapter(self.multimodal_settings),
        ]
        self.writer = writer or FeishuDepositWriter(self.settings)
        self.ingest_pipeline = ingest_pipeline or IngestPipeline(self.multimodal_settings)
        self.summarizer = summarizer or self._build_draft
        self.now_provider = now_provider or datetime.now

    def run(self, request: DepositRequest) -> DepositResult:
        started_at = perf_counter()
        normalized = self._normalize_request(request)
        auto_write = self.settings.deposit_enable_auto_write if normalized.auto_write is None else normalized.auto_write
        log_event(
            "deposit_started",
            text_preview=preview_text(normalized.text),
            url_count=len(normalized.urls),
            image_count=len(normalized.image_paths),
            auto_write=auto_write,
        )
        try:
            source = self._select_adapter(normalized).fetch(normalized)
            if normalized.image_paths:
                source.metadata = {
                    **source.metadata,
                    "image_paths": list(normalized.image_paths),
                    "image_count": len(normalized.image_paths),
                }
            draft = self.summarizer(source, normalized.text)
            placeholder_markdown, inline_images = self._render_markdown(
                draft,
                image_paths=normalized.image_paths,
            )
            final_markdown = placeholder_markdown
            ingest_markdown = self._render_ingest_markdown(placeholder_markdown, inline_images)
            self._log_rendered_markdown(
                event="deposit_feishu_markdown_prepared",
                markdown_content=placeholder_markdown,
                source_type=source.source_type,
            )
            self._log_rendered_markdown(
                event="deposit_ingest_markdown_prepared",
                markdown_content=ingest_markdown,
                source_type=source.source_type,
            )

            if not auto_write:
                elapsed_ms = (perf_counter() - started_at) * 1000
                log_event(
                    "deposit_completed",
                    status="preview",
                    source_type=source.source_type,
                    key_point_count=len(draft.key_points),
                    duration_ms=round(elapsed_ms, 1),
                )
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
                markdown_content=placeholder_markdown,
                image_paths=list(normalized.image_paths),
                inline_images=inline_images,
                target_space_id=target_space_id,
                target_parent_node_token=target_parent_node_token,
            )
            ingest_result = self.ingest_pipeline.ingest_markdown(
                ingest_markdown,
                title=draft.feishu_doc_title,
                metadata={
                    **draft.metadata,
                    "title": draft.feishu_doc_title,
                    "source_uri": write_result.document_url or draft.source_uri,
                    "source_type": f"deposit:{draft.source_type}",
                    "feishu_doc_token": write_result.document_token,
                    "wiki_node_token": write_result.wiki_node_token,
                    "image_paths": list(normalized.image_paths),
                },
            )
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_event(
                "deposit_completed",
                status="completed",
                source_type=source.source_type,
                local_document_id=ingest_result.document_id,
                feishu_doc_token=write_result.document_token,
                has_wiki_node=bool(write_result.wiki_node_token),
                duration_ms=round(elapsed_ms, 1),
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
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (perf_counter() - started_at) * 1000
            log_exception(
                "deposit_failed",
                exc,
                text_preview=preview_text(normalized.text),
                url_count=len(normalized.urls),
                image_count=len(normalized.image_paths),
                duration_ms=round(elapsed_ms, 1),
            )
            raise

    def _normalize_request(self, request: DepositRequest) -> DepositRequest:
        urls: list[str] = []
        for candidate in request.urls:
            normalized = normalize_url(candidate)
            if normalized and normalized not in urls:
                urls.append(normalized)
        if not urls and not request.provided_content.strip():
            for url in extract_urls(request.text):
                if url not in urls:
                    urls.append(url)
        return DepositRequest(
            text=request.text.strip(),
            urls=urls,
            source_title=request.source_title.strip(),
            provided_content=request.provided_content.strip(),
            image_paths=list(request.image_paths),
            inline_images=list(request.inline_images),
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
        lines = self._extract_summary_candidate_lines(source.raw_markdown)
        key_points = [line for line in lines if len(line) >= 8]
        if source.title:
            key_points = [source.title, *key_points]
        if not key_points:
            key_points = [source.title, source.extra_summary or source.raw_markdown[:120].strip()]
        deduped_key_points: list[str] = []
        seen_points: set[str] = set()
        for point in key_points:
            cleaned = str(point or "").strip()
            if not cleaned or cleaned in seen_points:
                continue
            seen_points.add(cleaned)
            deduped_key_points.append(cleaned)
        key_points = deduped_key_points[:5]
        summary_sentence = source.title or (key_points[0] if key_points else source.title)
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
            inline_images=list(source.inline_images),
            metadata={**source.metadata, "deposit_generated_at": self.now_provider().isoformat(timespec="seconds")},
        )

    @staticmethod
    def _extract_summary_candidate_lines(markdown: str) -> list[str]:
        candidates: list[str] = []
        in_fenced_block = False
        for raw_line in str(markdown or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```") or line == "```":
                in_fenced_block = not in_fenced_block
                continue
            if in_fenced_block:
                continue
            if MARKDOWN_IMAGE_RE.fullmatch(line):
                continue
            if line.startswith("http://") or line.startswith("https://"):
                continue
            compact = line.strip("- ").strip()
            compact = re.sub(r"^\s{0,3}#{1,6}\s+", "", compact).strip()
            if not compact:
                continue
            candidates.append(compact)
        return candidates

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

    def _render_markdown(self, draft: KnowledgeDraft, *, image_paths: list[str]) -> tuple[str, list[InlineImage]]:
        deposited_at = self.now_provider().isoformat(timespec="seconds")
        details = "\n".join(f"- {point}" for point in draft.key_points) or "- 无"
        tags = "、".join(draft.tags) or "无"
        rendered_body, inline_images = self._prepare_raw_content_with_inline_images(
            draft.raw_content_markdown.strip() or "无",
            draft.inline_images,
            image_paths=image_paths,
        )
        markdown = "\n".join(
            [
                f"# {draft.feishu_doc_title}",
                "",
                draft.summary_markdown,
                "",
                "## 详细整理",
                rendered_body,
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
        return markdown, inline_images

    def _render_ingest_markdown(self, markdown: str, inline_images: list[InlineImage]) -> str:
        rendered = markdown
        for order, image in enumerate(inline_images, start=1):
            image_ref = image.original_ref.strip() or image.image_path.strip()
            replacement = f"![图片{order}]({image_ref})" if image_ref else f"[图片{order}]"
            rendered = rendered.replace(image.placeholder, replacement)
        return rendered

    def _prepare_raw_content_with_inline_images(
        self,
        raw_markdown: str,
        source_inline_images: list[InlineImage],
        *,
        image_paths: list[str],
    ) -> tuple[str, list[InlineImage]]:
        normalized_paths = [str(path).strip() for path in image_paths if str(path).strip()]
        prepared_inline_images: list[InlineImage] = []
        path_cursor = 0
        consumed_paths: set[str] = set()

        def _next_path(preferred_path: str = "") -> str:
            nonlocal path_cursor
            candidate = preferred_path.strip()
            if candidate:
                consumed_paths.add(candidate)
                while path_cursor < len(normalized_paths) and normalized_paths[path_cursor] in consumed_paths:
                    path_cursor += 1
                return candidate
            while path_cursor < len(normalized_paths) and normalized_paths[path_cursor] in consumed_paths:
                path_cursor += 1
            if path_cursor >= len(normalized_paths):
                return ""
            candidate = normalized_paths[path_cursor]
            consumed_paths.add(candidate)
            path_cursor += 1
            return candidate

        ordered_source_images = [
            InlineImage(
                placeholder=image.placeholder,
                image_path=image.image_path,
                original_ref=image.original_ref,
                order=image.order,
            )
            for image in source_inline_images
        ]

        def _replace_markdown_image(match: re.Match[str]) -> str:
            image_index = len(prepared_inline_images)
            original_ref = match.group(1).strip()
            preferred_path = ""
            if image_index < len(ordered_source_images):
                preferred_path = ordered_source_images[image_index].image_path
                if not original_ref:
                    original_ref = ordered_source_images[image_index].original_ref
            placeholder = INLINE_IMAGE_PLACEHOLDER_TEMPLATE.format(order=image_index + 1)
            prepared_inline_images.append(
                InlineImage(
                    placeholder=placeholder,
                    image_path=_next_path(preferred_path),
                    original_ref=original_ref,
                    order=image_index,
                )
            )
            return f"\n\n{placeholder}\n\n"

        rendered = MARKDOWN_IMAGE_RE.sub(_replace_markdown_image, raw_markdown)
        remaining_source_images = ordered_source_images[len(prepared_inline_images) :]
        while path_cursor < len(normalized_paths):
            if normalized_paths[path_cursor] in consumed_paths:
                path_cursor += 1
                continue
            fallback_order = len(remaining_source_images)
            remaining_source_images.append(
                InlineImage(
                    placeholder="",
                    image_path=normalized_paths[path_cursor],
                    original_ref="",
                    order=fallback_order,
                )
            )
            path_cursor += 1

        if remaining_source_images:
            suggestions_markdown, suggestion_images = self._build_image_placement_suggestions(
                raw_markdown=rendered,
                pending_images=remaining_source_images,
                start_order=len(prepared_inline_images) + 1,
            )
            if suggestions_markdown:
                rendered = f"{rendered.strip()}\n\n{suggestions_markdown}".strip()
                prepared_inline_images.extend(suggestion_images)

        return rendered.strip(), prepared_inline_images

    def _build_image_placement_suggestions(
        self,
        *,
        raw_markdown: str,
        pending_images: list[InlineImage],
        start_order: int,
    ) -> tuple[str, list[InlineImage]]:
        if not pending_images:
            return "", []

        anchors = self._collect_image_placement_anchors(raw_markdown)
        suggestion_lines = ["## 图片放置建议"]
        suggestion_images: list[InlineImage] = []
        for offset, image in enumerate(pending_images):
            placeholder = INLINE_IMAGE_PLACEHOLDER_TEMPLATE.format(order=start_order + offset)
            anchor = anchors[min(offset, len(anchors) - 1)] if anchors else "正文对应位置附近"
            suggestion_lines.append(f"- {placeholder} 建议放在{anchor}")
            suggestion_images.append(
                InlineImage(
                    placeholder=placeholder,
                    image_path=image.image_path,
                    original_ref=image.original_ref,
                    order=start_order + offset - 1,
                )
            )
        return "\n".join(suggestion_lines), suggestion_images

    @staticmethod
    def _collect_image_placement_anchors(raw_markdown: str) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        blocks = [block.strip() for block in re.split(r"\n\s*\n", raw_markdown or "") if block.strip()]
        for block in blocks:
            if MARKDOWN_IMAGE_RE.search(block):
                continue
            heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", block)
            if heading_match:
                anchor = f"“{heading_match.group(1).strip()}”后"
            else:
                compact = re.sub(r"\s+", " ", block).strip("- ").strip()
                if not compact:
                    continue
                snippet = compact[:24]
                anchor = f"“{snippet}”后"
            if anchor not in seen:
                seen.add(anchor)
                anchors.append(anchor)
        return anchors or ["正文对应位置附近"]

    @staticmethod
    def _log_rendered_markdown(*, event: str, markdown_content: str, source_type: str) -> None:
        log_event(
            event,
            source_type=source_type,
            markdown_length=len(markdown_content),
            markdown_preview=preview_text(markdown_content, limit=800),
            placeholder_count=markdown_content.count("[[IMG_"),
            markdown_image_count=len(MARKDOWN_IMAGE_RE.findall(markdown_content)),
        )

    @staticmethod
    def _source_type_label(source_type: str) -> str:
        return {
            "xiaohongshu": "小红书",
            "url": "链接",
            "text": "文本",
            "image": "图片",
        }.get(source_type, source_type or "来源")
