---
name: knowledge-deposit
description: Deposit external links, text, and images into the Feishu knowledge base by fetching source content, summarizing it, writing a Feishu document, and indexing the final markdown locally.
---

# Knowledge Deposit Skill

## When to Use

Use this skill when the user explicitly asks to save, deposit, archive, ingest, or沉淀 content into the knowledge base.

## Required Workflow

1. Detect the source type before doing any write action.
2. Prefer dedicated adapters for supported sources such as Xiaohongshu.
3. Generate a cleaned knowledge draft with summary and source metadata.
4. Write the final markdown into Feishu Docs and mount it into Wiki when auto-write is enabled.
5. Only add the final markdown to the local index after Feishu write succeeds.
6. If required write configuration is missing, explain that clearly and do not pretend the deposit succeeded.

## Output Rules

- Keep the final user-facing confirmation concise.
- Include whether the result is preview-only or fully written.
- Include the Feishu doc link when available.
