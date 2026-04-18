"""Parquet persistence helpers using PyArrow."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq


def save_parquet(records: list[dict], path: str) -> None:
    """Serialize a list of dicts to a Parquet file.

    Creates parent directories as needed.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def load_parquet(path: str) -> list[dict]:
    """Read a Parquet file and return its rows as a list of dicts."""
    table = pq.read_table(path)
    return table.to_pylist()
