"""Hallucination detection metrics."""

from __future__ import annotations

from typing import Any


def compute_hallucination_metrics(
    result_rows: list[dict],
    examples_by_id: dict[str, Any],
) -> dict:
    """Detect hallucinated responses and return aggregate metrics.

    A response is considered hallucinated when:
    - The question is unanswerable (is_answerable=False) but the model
      provides a non-empty answer without abstaining, OR
    - The model provides evidence quotes that do not appear verbatim in
      the source context.
    """
    from .accuracy import parse_output  # avoid circular at module level

    total = len(result_rows)
    hallucinated_responses = 0

    for row in result_rows:
        ex = examples_by_id.get(row["example_id"])
        if ex is None:
            continue
        parsed, valid = parse_output(row["raw_text"], row["task"])
        if not valid or parsed is None:
            continue
        task = row["task"]
        if task in ("grounded_qa", "multihop_qa"):
            if (
                not ex.is_answerable
                and not parsed.get("abstain", False)
                and parsed.get("answer", "")
            ):
                hallucinated_responses += 1
            elif ex.is_answerable and not parsed.get("abstain", False):
                quotes = parsed.get("evidence_quotes", [])
                context = ex.context
                if quotes and not any(q.strip() in context for q in quotes):
                    hallucinated_responses += 1

    return {
        "response_hallucination_rate": (
            hallucinated_responses / total if total else 0.0
        ),
        "hallucinated_count": hallucinated_responses,
        "total_evaluated": total,
    }
