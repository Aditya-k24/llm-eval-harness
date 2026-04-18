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
    """Sample n examples from FEVER dev set, balanced across labels.

    Uses the copenlu/fever dataset (Parquet-backed, no legacy scripts).
    Each example uses the claim as both context and question; the
    evidence field is populated where available.
    """
    from datasets import load_dataset  # type: ignore

    # copenlu/fever is stored as Parquet and avoids the deprecated script
    ds = load_dataset("copenlu/fever", split="validation")
    rng = random.Random(seed)
    label_map = {
        "SUPPORTS": "SUPPORTED",
        "REFUTES": "REFUTED",
        "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
    }
    by_label: dict[str, list] = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
    for ex in ds:
        label = ex.get("label", "")
        if label in by_label:
            by_label[label].append(ex)

    per_label = n // 3
    sample: list = []
    for label, examples in by_label.items():
        sample.extend(rng.sample(examples, min(per_label, len(examples))))
    rng.shuffle(sample)

    rows = []
    for ex in sample:
        # Build a readable evidence context from the nested evidence list
        evidence_lines: list[str] = []
        for ev_group in ex.get("evidence", []):
            for ev in ev_group:
                page = ev.get("wikipedia_url", "")
                text = ev.get("sentence", "") or ev.get("evidence_sentence", "")
                if text:
                    evidence_lines.append(f"[{page}] {text}" if page else text)
        context = "\n".join(evidence_lines) if evidence_lines else ex.get("claim", "")
        rows.append(
            {
                "id": f"fever_{ex['id']}",
                "task": "fever",
                "context": context,
                "question": ex["claim"],
                "gold_label": label_map.get(ex.get("label", ""), "NOT_ENOUGH_INFO"),
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
