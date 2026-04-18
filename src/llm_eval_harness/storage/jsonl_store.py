"""Append-only JSONL store for streaming result persistence."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime


class JSONLStore:
    """Append-only file-backed JSONL store.

    Flushes after every write so that results survive mid-run crashes.
    """

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        """Append a single record to the store, prepending a timestamp."""
        stamped = {"_ts": datetime.utcnow().isoformat(), **record}
        self._file.write(json.dumps(stamped) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def read_all(self) -> list[dict]:
        """Return all records as a list of dicts."""
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def __enter__(self) -> "JSONLStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
