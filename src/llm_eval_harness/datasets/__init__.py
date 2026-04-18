"""Dataset subpackage — loaders, manifests, and split builders."""

from .loaders import (
    GroundedQAExample,
    MultiHopQAExample,
    FEVERExample,
    EvalExample,
    load_jsonl,
)
from .manifests import DatasetManifest, create_manifest, save_manifest, load_manifest
from .splits import build_smoke_split, build_dev_split

__all__ = [
    "GroundedQAExample",
    "MultiHopQAExample",
    "FEVERExample",
    "EvalExample",
    "load_jsonl",
    "DatasetManifest",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "build_smoke_split",
    "build_dev_split",
]
