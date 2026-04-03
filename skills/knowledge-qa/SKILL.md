---
name: knowledge-qa
description: Answer questions from the indexed Feishu Wiki and Docs knowledge base. Use when the user asks about team processes, product docs, internal knowledge, policies, or other information that should come from indexed Feishu documentation.
---

# Knowledge QA Skill

## When to Use

Use this skill whenever the answer should come from Feishu Wiki or Docs content that has been indexed locally.

## Required Workflow

1. Call `search_feishu_knowledge` before answering factual documentation questions.
2. Read the retrieved snippets carefully.
3. Answer only from the retrieved content.
4. If retrieval returns no relevant content, say `当前索引中未找到相关内容。`
5. End the answer with a short `来源：` line that includes the document titles or links you used.
