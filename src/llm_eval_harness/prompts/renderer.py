"""Prompt renderer — reads template files and formats them with example data."""

from __future__ import annotations

import pathlib

from ..datasets.loaders import EvalExample


def render(
    task: str,
    example: EvalExample,
    prompts_dir: str = "prompts",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given task and example.

    Templates live at ``{prompts_dir}/{task}/system.txt`` and ``user.txt``.
    The user template is formatted with ``context`` and ``question`` fields
    extracted from the example.
    """
    base = pathlib.Path(prompts_dir) / task
    system = (base / "system.txt").read_text().strip()
    user_tpl = (base / "user.txt").read_text().strip()
    user = user_tpl.format(
        context=getattr(example, "context", ""),
        question=getattr(example, "question", ""),
    )
    return system, user
