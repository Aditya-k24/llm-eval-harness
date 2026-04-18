"""Dataset manifest creation and persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from dataclasses import dataclass


@dataclass
class DatasetManifest:
    split: str
    version: str
    example_count: int
    file_path: str
    checksum: str
    tasks: list[str]


def create_manifest(
    split: str,
    version: str,
    file_path: str,
    tasks: list[str],
) -> DatasetManifest:
    """Compute SHA-256 checksum and count examples in a JSONL file."""
    data = pathlib.Path(file_path).read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    lines = [
        line
        for line in pathlib.Path(file_path).read_text().strip().splitlines()
        if line.strip()
    ]
    return DatasetManifest(
        split=split,
        version=version,
        example_count=len(lines),
        file_path=file_path,
        checksum=checksum,
        tasks=tasks,
    )


def save_manifest(manifest: DatasetManifest, out_path: str) -> None:
    """Serialise manifest to a JSON file."""
    pathlib.Path(out_path).write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2)
    )


def load_manifest(path: str) -> DatasetManifest:
    """Load a previously saved manifest from JSON."""
    raw = json.loads(pathlib.Path(path).read_text())
    return DatasetManifest(**raw)
