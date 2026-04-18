"""Audit queue builder — flags responses that need human review."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from ..metrics.accuracy import parse_output


def build_audit_queue(
    result_rows: list[dict],
    examples_by_id: dict[str, Any],
    output_path: str,
) -> list[dict]:
    """Identify responses that need human review and export them as JSON.

    Review is triggered when:
    - The model output is not valid JSON.
    - The model answered an unanswerable question without abstaining.
    - Evidence quotes cannot be found verbatim in the source context.

    Args:
        result_rows: Raw result dicts from the runner.
        examples_by_id: Mapping of example ID to EvalExample.
        output_path: Destination path for the audit queue JSON file.

    Returns:
        List of audit items written to *output_path*.
    """
    queue: list[dict] = []

    for row in result_rows:
        ex = examples_by_id.get(row["example_id"])
        parsed, valid = parse_output(row["raw_text"], row["task"])
        needs_review = False
        reason: list[str] = []

        if not valid:
            needs_review = True
            reason.append("invalid_json")

        if ex is not None and valid and parsed is not None:
            task = row["task"]
            if task in ("grounded_qa", "multihop_qa"):
                if not ex.is_answerable and not parsed.get("abstain", False):
                    needs_review = True
                    reason.append("answered_unanswerable")
                quotes = parsed.get("evidence_quotes", [])
                if quotes and not any(q.strip() in ex.context for q in quotes):
                    needs_review = True
                    reason.append("unverified_quotes")

        if needs_review:
            queue.append(
                {
                    "example_id": row["example_id"],
                    "model_id": row["model_id"],
                    "task": row["task"],
                    "question": getattr(ex, "question", "") if ex else "",
                    "gold_answer": (
                        getattr(ex, "gold_answer", getattr(ex, "gold_label", ""))
                        if ex
                        else ""
                    ),
                    "model_answer": (
                        parsed.get("answer", parsed.get("verdict", ""))
                        if parsed
                        else ""
                    ),
                    "evidence_quotes": (
                        parsed.get("evidence_quotes", []) if parsed else []
                    ),
                    "raw_text": row["raw_text"],
                    "review_reasons": reason,
                    "annotation_status": "pending",
                }
            )

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path).write_text(json.dumps(queue, indent=2))
    return queue
