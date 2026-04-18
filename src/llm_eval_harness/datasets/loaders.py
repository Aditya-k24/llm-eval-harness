"""Pydantic models and JSONL loaders for all three task types."""

from __future__ import annotations

import json
import pathlib
from typing import Literal

from pydantic import BaseModel


class GroundedQAExample(BaseModel):
    id: str
    task: Literal["grounded_qa"] = "grounded_qa"
    context: str
    question: str
    gold_answer: str
    is_answerable: bool
    gold_evidence_quotes: list[str] = []


class MultiHopQAExample(BaseModel):
    id: str
    task: Literal["multihop_qa"] = "multihop_qa"
    context: str  # multiple passages joined
    question: str
    gold_answer: str
    is_answerable: bool
    gold_evidence_quotes: list[str] = []
    supporting_facts: list[str] = []


class FEVERExample(BaseModel):
    id: str
    task: Literal["fever"] = "fever"
    context: str  # evidence passage(s)
    question: str  # the claim
    gold_label: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    gold_evidence_quotes: list[str] = []


EvalExample = GroundedQAExample | MultiHopQAExample | FEVERExample


def load_jsonl(path: str) -> list[EvalExample]:
    """Load examples from a JSONL file and return typed Pydantic objects."""
    examples: list[EvalExample] = []
    text = pathlib.Path(path).read_text().strip()
    if not text:
        return examples
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        task = raw.get("task", "grounded_qa")
        if task == "grounded_qa":
            examples.append(GroundedQAExample(**raw))
        elif task == "multihop_qa":
            examples.append(MultiHopQAExample(**raw))
        elif task == "fever":
            examples.append(FEVERExample(**raw))
        else:
            raise ValueError(f"Unknown task type: {task!r}")
    return examples
