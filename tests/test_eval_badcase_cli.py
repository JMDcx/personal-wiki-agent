from __future__ import annotations

import json

from scripts import export_eval_badcases


def test_export_eval_badcases_cli_writes_draft_file(tmp_path):
    log_path = tmp_path / "app.jsonl"
    output_path = tmp_path / "drafts.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "event": "request_summary",
                "request_id": "thread:abc",
                "status": "error",
                "question_preview": "broken question",
                "answer_preview": "broken answer",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = export_eval_badcases.main(["--log", str(log_path), "--output", str(output_path)])

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["user_query"] == "broken question"
