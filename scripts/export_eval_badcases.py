"""Export runtime log records into eval-case drafts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from multimodal_rag_agent.eval.badcase import write_badcase_drafts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export structured logs into eval badcase draft JSONL.")
    parser.add_argument(
        "--log",
        default="data/logs/app.jsonl",
        help="Path to structured JSONL log file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Defaults to data/evals/badcase_drafts/<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="Also export successful requests for manual review.",
    )
    parser.add_argument(
        "--request-ids",
        default="",
        help="Comma-separated request ids to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of drafts to export.",
    )
    return parser


def _parse_ids(raw_ids: str) -> set[str]:
    return {item.strip() for item in str(raw_ids or "").split(",") if item.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    log_path = Path(args.log).resolve()
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (Path("data/evals/badcase_drafts") / f"badcases-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl").resolve()
    )
    drafts = write_badcase_drafts(
        log_path=log_path,
        output_path=output_path,
        include_ok=bool(args.include_ok),
        request_ids=_parse_ids(args.request_ids),
        limit=int(args.limit),
    )
    print(f"Exported {len(drafts)} badcase draft(s).")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
