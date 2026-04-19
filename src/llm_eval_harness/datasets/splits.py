"""Download and create benchmark splits from HuggingFace datasets."""

from __future__ import annotations

import json
import pathlib
import random


def build_squad_split(
    n: int,
    seed: int = 42,
    out_path: str | None = None,
) -> list[dict]:
    """Sample n examples from SQuAD 2.0, balanced answerable/unanswerable."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(seed)
    answerable = [ex for ex in ds if ex["answers"]["text"]]
    unanswerable = [ex for ex in ds if not ex["answers"]["text"]]
    half = n // 2
    sample = rng.sample(answerable, min(half, len(answerable))) + rng.sample(
        unanswerable, min(n - half, len(unanswerable))
    )
    rng.shuffle(sample)
    rows = []
    for ex in sample:
        is_ans = bool(ex["answers"]["text"])
        rows.append(
            {
                "id": f"squad_{ex['id']}",
                "task": "grounded_qa",
                "context": ex["context"],
                "question": ex["question"],
                "gold_answer": ex["answers"]["text"][0] if is_ans else "",
                "is_answerable": is_ans,
                "gold_evidence_quotes": ex["answers"]["text"][:1] if is_ans else [],
            }
        )
    if out_path:
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    return rows


def build_hotpotqa_split(
    n: int,
    seed: int = 42,
    out_path: str | None = None,
) -> list[dict]:
    """Sample n examples from HotpotQA distractor dev set."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    sample = rng.sample(list(ds), min(n, len(ds)))
    rows = []
    for ex in sample:
        context_parts = [
            f"[{title}] {''.join(sents)}"
            for title, sents in zip(
                ex["context"]["title"], ex["context"]["sentences"]
            )
        ]
        context = "\n\n".join(context_parts)
        rows.append(
            {
                "id": f"hotpot_{ex['id']}",
                "task": "multihop_qa",
                "context": context,
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "is_answerable": True,
                "gold_evidence_quotes": [],
                "supporting_facts": [
                    f"{t}: {s}"
                    for t, s in zip(
                        ex["supporting_facts"]["title"],
                        ex["supporting_facts"]["sent_id"],
                    )
                ],
            }
        )
    if out_path:
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    return rows


def build_fever_split(
    n: int,
    seed: int = 42,
    out_path: str | None = None,
) -> list[dict]:
    """Sample n examples from climate_fever, balanced across labels.

    climate_fever uses the same SUPPORTS/REFUTES/NOT_ENOUGH_INFO scheme
    as FEVER and is stored as Parquet (no legacy dataset scripts needed).
    Label integers: 0=SUPPORTS, 1=REFUTES, 2=NOT_ENOUGH_INFO, 3=DISPUTED.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("climate_fever", split="test")
    rng = random.Random(seed)
    int_to_str = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT_ENOUGH_INFO", 3: "NOT_ENOUGH_INFO"}
    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for ex in ds:
        lbl = ex.get("claim_label", -1)
        if lbl in by_label:
            by_label[lbl].append(ex)

    per_label = n // 3
    sample: list = []
    for lbl, examples in by_label.items():
        sample.extend(rng.sample(examples, min(per_label, len(examples))))
    rng.shuffle(sample)

    rows = []
    for ex in sample:
        evidence_lines = [
            f"[{ev['article']}] {ev['evidence']}"
            for ev in ex.get("evidences", [])
            if ev.get("evidence")
        ]
        context = "\n".join(evidence_lines) if evidence_lines else ex["claim"]
        rows.append(
            {
                "id": f"fever_{ex['claim_id']}",
                "task": "fever",
                "context": context,
                "question": ex["claim"],
                "gold_label": int_to_str.get(ex.get("claim_label", 2), "NOT_ENOUGH_INFO"),
                "gold_evidence_quotes": [],
            }
        )
    if out_path:
        pathlib.Path(out_path).write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    return rows


def build_smoke_split(out_dir: str = "datasets/public") -> str:
    """Build 30-example smoke split (10 per task)."""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/smoke.jsonl"
    squad = build_squad_split(10, seed=42)
    hotpot = build_hotpotqa_split(10, seed=42)
    fever = build_fever_split(10, seed=42)
    rows = squad + hotpot + fever
    pathlib.Path(out_path).write_text("\n".join(json.dumps(r) for r in rows))
    return out_path


def build_dev_split(out_dir: str = "datasets/public") -> str:
    """Build 200-example dev split."""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/dev.jsonl"
    squad = build_squad_split(80, seed=42)
    hotpot = build_hotpotqa_split(60, seed=42)
    fever = build_fever_split(60, seed=42)
    rows = squad + hotpot + fever
    pathlib.Path(out_path).write_text("\n".join(json.dumps(r) for r in rows))
    return out_path
