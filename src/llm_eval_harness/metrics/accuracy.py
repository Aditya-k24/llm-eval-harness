"""Accuracy metrics: EM, F1, abstention, evidence validity, and JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Literal, Any

from pydantic import BaseModel


class GroundedQAOutput(BaseModel):
    answer: str
    abstain: bool
    evidence_quotes: list[str]


class MultiHopQAOutput(BaseModel):
    answer: str
    abstain: bool
    evidence_quotes: list[str]
    reasoning_chain: str = ""


class FEVEROutput(BaseModel):
    verdict: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    evidence_quotes: list[str]
    reasoning: str = ""


def normalize_text(text: str) -> str:
    """Lowercase, remove articles, punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold answer strings."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = (
        sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
        / len(pred_tokens)
    )
    recall = (
        sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
        / len(gold_tokens)
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    """Case-insensitive, article-stripped exact match."""
    return normalize_text(pred) == normalize_text(gold)


def parse_output(text: str, task: str) -> tuple[dict | None, bool]:
    """Parse JSON from model output, optionally extracting from markdown fences.

    Returns (parsed_dict, is_valid).
    """
    try:
        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            text = match.group(1)
        data = json.loads(text.strip())
        if task == "grounded_qa":
            GroundedQAOutput(**data)
        elif task == "multihop_qa":
            MultiHopQAOutput(**data)
        elif task == "fever":
            FEVEROutput(**data)
        return data, True
    except Exception:
        return None, False


def evidence_quote_validity(quotes: list[str], context: str) -> float:
    """Fraction of evidence quotes that appear verbatim in the context."""
    if not quotes:
        return 0.0
    valid = sum(1 for q in quotes if q.strip() and q.strip() in context)
    return valid / len(quotes)


def compute_accuracy_metrics(result_row: dict, example: Any) -> dict:
    """Compute all accuracy-related metrics for a single result row."""
    task = result_row["task"]
    parsed, json_valid = parse_output(result_row["raw_text"], task)

    metrics: dict[str, Any] = {"json_valid": json_valid}

    if not json_valid or parsed is None:
        metrics.update(
            {
                "exact_match": 0,
                "token_f1": 0.0,
                "abstain_correct": 0,
                "evidence_quote_validity": 0.0,
                "label_correct": 0,
            }
        )
        return metrics

    context = getattr(example, "context", "")

    if task in ("grounded_qa", "multihop_qa"):
        pred_answer = parsed.get("answer", "")
        pred_abstain = parsed.get("abstain", False)
        gold_answer = example.gold_answer
        is_answerable = example.is_answerable

        metrics["exact_match"] = (
            int(exact_match(pred_answer, gold_answer))
            if is_answerable and not pred_abstain
            else 0
        )
        metrics["token_f1"] = (
            token_f1(pred_answer, gold_answer)
            if is_answerable and not pred_abstain
            else 0.0
        )
        metrics["abstain_correct"] = int(pred_abstain == (not is_answerable))
        metrics["evidence_quote_validity"] = evidence_quote_validity(
            parsed.get("evidence_quotes", []), context
        )
        # label_correct not used for QA tasks; set to 0
        metrics["label_correct"] = 0

    elif task == "fever":
        pred_verdict = parsed.get("verdict", "")
        metrics["label_correct"] = int(pred_verdict == example.gold_label)
        metrics["evidence_quote_validity"] = evidence_quote_validity(
            parsed.get("evidence_quotes", []), context
        )
        # QA-specific keys
        metrics["exact_match"] = 0
        metrics["token_f1"] = 0.0
        metrics["abstain_correct"] = 0

    return metrics
