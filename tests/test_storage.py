"""Tests for JSONLStore and Parquet storage."""

from __future__ import annotations

import json
import pathlib

import pytest

from llm_eval_harness.storage.jsonl_store import JSONLStore
from llm_eval_harness.storage.parquet_store import save_parquet, load_parquet


# ---------------------------------------------------------------------------
# JSONLStore
# ---------------------------------------------------------------------------

class TestJSONLStore:
    def test_write_and_read(self, tmp_path):
        path = str(tmp_path / "results.jsonl")
        store = JSONLStore(path)
        store.write({"model": "gpt-4o", "score": 0.9})
        store.write({"model": "claude", "score": 0.85})
        store.close()

        records = store.read_all()
        assert len(records) == 2
        assert records[0]["model"] == "gpt-4o"
        assert records[1]["model"] == "claude"

    def test_timestamp_added(self, tmp_path):
        path = str(tmp_path / "ts_test.jsonl")
        store = JSONLStore(path)
        store.write({"x": 1})
        store.close()

        records = store.read_all()
        assert "_ts" in records[0]

    def test_context_manager(self, tmp_path):
        path = str(tmp_path / "ctx.jsonl")
        with JSONLStore(path) as store:
            store.write({"a": 1})
            store.write({"a": 2})

        # File should be closed but still readable
        records = JSONLStore(path).read_all()
        assert len(records) == 2

    def test_creates_parent_dir(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "out.jsonl")
        store = JSONLStore(path)
        store.write({"key": "value"})
        store.close()

        assert pathlib.Path(path).exists()

    def test_append_across_instances(self, tmp_path):
        path = str(tmp_path / "append.jsonl")
        with JSONLStore(path) as s1:
            s1.write({"n": 1})

        with JSONLStore(path) as s2:
            s2.write({"n": 2})

        records = JSONLStore(path).read_all()
        ns = [r["n"] for r in records]
        assert 1 in ns
        assert 2 in ns

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        pathlib.Path(path).touch()
        store = JSONLStore(path)
        records = store.read_all()
        store.close()
        assert records == []

    def test_flush_on_write(self, tmp_path):
        """Check that data is on disk immediately after write (flush)."""
        path = str(tmp_path / "flush.jsonl")
        store = JSONLStore(path)
        store.write({"immediate": True})
        # Read without closing the store
        content = pathlib.Path(path).read_text()
        store.close()
        assert "immediate" in content


# ---------------------------------------------------------------------------
# Parquet store
# ---------------------------------------------------------------------------

class TestParquetStore:
    def test_round_trip(self, tmp_path):
        records = [
            {"model_id": "gpt-4o", "score": 0.9, "cost": 0.001},
            {"model_id": "claude", "score": 0.85, "cost": 0.002},
        ]
        path = str(tmp_path / "results.parquet")
        save_parquet(records, path)
        loaded = load_parquet(path)
        assert len(loaded) == 2
        model_ids = {r["model_id"] for r in loaded}
        assert "gpt-4o" in model_ids
        assert "claude" in model_ids

    def test_creates_parent_dir(self, tmp_path):
        records = [{"x": 1}]
        path = str(tmp_path / "nested" / "out.parquet")
        save_parquet(records, path)
        assert pathlib.Path(path).exists()

    def test_empty_records(self, tmp_path):
        path = str(tmp_path / "empty.parquet")
        save_parquet([], path)
        loaded = load_parquet(path)
        assert loaded == []

    def test_varied_types(self, tmp_path):
        records = [
            {
                "id": "ex1",
                "score": 0.95,
                "tokens": 100,
                "valid": True,
                "text": "hello",
            }
        ]
        path = str(tmp_path / "typed.parquet")
        save_parquet(records, path)
        loaded = load_parquet(path)
        assert loaded[0]["score"] == pytest.approx(0.95)
        assert loaded[0]["tokens"] == 100
        assert loaded[0]["valid"] is True

    def test_large_record_set(self, tmp_path):
        records = [{"i": i, "v": float(i) * 0.1} for i in range(1000)]
        path = str(tmp_path / "large.parquet")
        save_parquet(records, path)
        loaded = load_parquet(path)
        assert len(loaded) == 1000
        assert loaded[500]["i"] == 500
