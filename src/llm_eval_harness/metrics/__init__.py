"""Metrics subpackage — accuracy, latency, cost, hallucination, significance."""

from .accuracy import (
    GroundedQAOutput,
    MultiHopQAOutput,
    FEVEROutput,
    normalize_text,
    token_f1,
    exact_match,
    parse_output,
    evidence_quote_validity,
    compute_accuracy_metrics,
)
from .latency import compute_latency_stats
from .cost import compute_cost
from .hallucination import compute_hallucination_metrics
from .significance import bootstrap_ci, mcnemar_test

__all__ = [
    "GroundedQAOutput",
    "MultiHopQAOutput",
    "FEVEROutput",
    "normalize_text",
    "token_f1",
    "exact_match",
    "parse_output",
    "evidence_quote_validity",
    "compute_accuracy_metrics",
    "compute_latency_stats",
    "compute_cost",
    "compute_hallucination_metrics",
    "bootstrap_ci",
    "mcnemar_test",
]
