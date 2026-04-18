"""Tests for all metrics modules."""

from __future__ import annotations

import pytest

from llm_eval_harness.metrics.accuracy import (
    exact_match,
    token_f1,
    normalize_text,
    parse_output,
    evidence_quote_validity,
    compute_accuracy_metrics,
)
from llm_eval_harness.metrics.cost import compute_cost
from llm_eval_harness.metrics.latency import compute_latency_stats
from llm_eval_harness.metrics.significance import bootstrap_ci, mcnemar_test


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_articles(self):
        assert "the" not in normalize_text("the quick brown fox")
        assert "a" not in normalize_text("a cat")
        assert "an" not in normalize_text("an apple")

    def test_removes_punctuation(self):
        result = normalize_text("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        result = normalize_text("  hello   world  ")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") is True

    def test_article_stripped(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") is True

    def test_different_answers(self):
        assert exact_match("London", "Paris") is False

    def test_empty_strings(self):
        assert exact_match("", "") is True

    def test_one_empty(self):
        assert exact_match("answer", "") is False


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat", "the dog sat")
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_pred_empty(self):
        assert token_f1("", "answer") == pytest.approx(0.0)

    def test_gold_empty(self):
        assert token_f1("answer", "") == pytest.approx(0.0)

    def test_subset_answer(self):
        # Prediction is a subset of gold — recall < 1, precision = 1
        f1 = token_f1("cat", "the cat sat on mat")
        assert 0.0 < f1 < 1.0


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

class TestParseOutput:
    def test_valid_grounded_qa(self):
        text = '{"answer": "Paris", "abstain": false, "evidence_quotes": ["Paris is the capital"]}'
        parsed, valid = parse_output(text, "grounded_qa")
        assert valid is True
        assert parsed["answer"] == "Paris"

    def test_valid_fever(self):
        text = '{"verdict": "SUPPORTED", "evidence_quotes": [], "reasoning": "ok"}'
        parsed, valid = parse_output(text, "fever")
        assert valid is True
        assert parsed["verdict"] == "SUPPORTED"

    def test_valid_multihop_qa(self):
        text = '{"answer": "42", "abstain": false, "evidence_quotes": [], "reasoning_chain": "step1"}'
        parsed, valid = parse_output(text, "multihop_qa")
        assert valid is True

    def test_invalid_json(self):
        text = "this is not json at all"
        parsed, valid = parse_output(text, "grounded_qa")
        assert valid is False
        assert parsed is None

    def test_markdown_fenced_json(self):
        text = '```json\n{"answer": "Rome", "abstain": false, "evidence_quotes": []}\n```'
        parsed, valid = parse_output(text, "grounded_qa")
        assert valid is True
        assert parsed["answer"] == "Rome"

    def test_missing_required_field(self):
        # Missing 'abstain' — Pydantic should reject
        text = '{"answer": "Rome", "evidence_quotes": []}'
        parsed, valid = parse_output(text, "grounded_qa")
        assert valid is False

    def test_invalid_fever_verdict(self):
        text = '{"verdict": "MAYBE", "evidence_quotes": [], "reasoning": "x"}'
        parsed, valid = parse_output(text, "fever")
        assert valid is False


# ---------------------------------------------------------------------------
# evidence_quote_validity
# ---------------------------------------------------------------------------

class TestEvidenceQuoteValidity:
    def test_all_valid(self):
        context = "The sky is blue and the grass is green."
        quotes = ["The sky is blue", "the grass is green"]
        assert evidence_quote_validity(quotes, context) == pytest.approx(1.0)

    def test_none_valid(self):
        context = "The sky is blue."
        quotes = ["The ocean is deep"]
        assert evidence_quote_validity(quotes, context) == pytest.approx(0.0)

    def test_partial(self):
        context = "The sky is blue and water is wet."
        quotes = ["The sky is blue", "fire is hot"]
        assert evidence_quote_validity(quotes, context) == pytest.approx(0.5)

    def test_empty_quotes(self):
        assert evidence_quote_validity([], "some context") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_accuracy_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeAccuracyMetrics:
    def _make_example(self, **kwargs):
        from llm_eval_harness.datasets.loaders import GroundedQAExample
        defaults = dict(
            id="ex1",
            task="grounded_qa",
            context="The capital of France is Paris.",
            question="What is the capital of France?",
            gold_answer="Paris",
            is_answerable=True,
        )
        defaults.update(kwargs)
        return GroundedQAExample(**defaults)

    def test_correct_answerable(self):
        row = {
            "task": "grounded_qa",
            "raw_text": '{"answer": "Paris", "abstain": false, "evidence_quotes": ["The capital of France is Paris."]}',
        }
        ex = self._make_example()
        metrics = compute_accuracy_metrics(row, ex)
        assert metrics["json_valid"] is True
        assert metrics["exact_match"] == 1
        assert metrics["token_f1"] == pytest.approx(1.0)
        assert metrics["abstain_correct"] == 1

    def test_invalid_json_zeros_all(self):
        row = {"task": "grounded_qa", "raw_text": "not json"}
        ex = self._make_example()
        metrics = compute_accuracy_metrics(row, ex)
        assert metrics["json_valid"] is False
        assert metrics["exact_match"] == 0
        assert metrics["token_f1"] == pytest.approx(0.0)

    def test_fever_label_correct(self):
        from llm_eval_harness.datasets.loaders import FEVERExample
        ex = FEVERExample(
            id="f1",
            task="fever",
            context="Evidence text here.",
            question="A claim.",
            gold_label="SUPPORTED",
        )
        row = {
            "task": "fever",
            "raw_text": '{"verdict": "SUPPORTED", "evidence_quotes": [], "reasoning": "ok"}',
        }
        metrics = compute_accuracy_metrics(row, ex)
        assert metrics["label_correct"] == 1

    def test_fever_label_incorrect(self):
        from llm_eval_harness.datasets.loaders import FEVERExample
        ex = FEVERExample(
            id="f2",
            task="fever",
            context="Evidence text here.",
            question="A claim.",
            gold_label="SUPPORTED",
        )
        row = {
            "task": "fever",
            "raw_text": '{"verdict": "REFUTED", "evidence_quotes": [], "reasoning": "nope"}',
        }
        metrics = compute_accuracy_metrics(row, ex)
        assert metrics["label_correct"] == 0


# ---------------------------------------------------------------------------
# compute_cost
# ---------------------------------------------------------------------------

class TestComputeCost:
    def test_basic(self):
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            input_rate_per_mtok=3.0,
            output_rate_per_mtok=15.0,
        )
        assert cost == pytest.approx(18.0)

    def test_with_cached_tokens(self):
        cost = compute_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            input_rate_per_mtok=3.0,
            output_rate_per_mtok=15.0,
            cached_input_tokens=500_000,
            cached_input_rate_per_mtok=1.5,
        )
        # 500k regular @ 3.0 + 500k cached @ 1.5
        assert cost == pytest.approx(1.5 + 0.75)

    def test_zero_tokens(self):
        cost = compute_cost(0, 0, 3.0, 15.0)
        assert cost == pytest.approx(0.0)

    def test_small_tokens(self):
        cost = compute_cost(
            input_tokens=100,
            output_tokens=50,
            input_rate_per_mtok=2.5,
            output_rate_per_mtok=10.0,
        )
        assert cost == pytest.approx(100 / 1e6 * 2.5 + 50 / 1e6 * 10.0)


# ---------------------------------------------------------------------------
# compute_latency_stats
# ---------------------------------------------------------------------------

class TestComputeLatencyStats:
    def test_basic(self):
        stats = compute_latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert stats["p50_ms"] == 300.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 500.0
        assert stats["mean_ms"] == pytest.approx(300.0)

    def test_empty(self):
        assert compute_latency_stats([]) == {}

    def test_single(self):
        stats = compute_latency_stats([42.0])
        assert stats["p50_ms"] == 42.0
        assert stats["min_ms"] == 42.0
        assert stats["max_ms"] == 42.0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_tuple(self):
        result = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 1.0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lower_le_upper(self):
        lo, hi = bootstrap_ci(list(range(100)))
        assert lo <= hi

    def test_tight_ci_for_constant(self):
        lo, hi = bootstrap_ci([5.0] * 100)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_custom_stat_fn(self):
        lo, hi = bootstrap_ci([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], stat_fn=max)
        assert hi <= 10


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------

class TestMcNemarTest:
    def test_no_discordant_pairs(self):
        # b == c == 0 → p-value should be 1
        p = mcnemar_test([1, 1, 0, 0], [1, 1, 0, 0])
        assert p == pytest.approx(1.0)

    def test_all_discordant_one_way(self):
        # All discordant in favour of A → very significant
        a = [1] * 50 + [0] * 50
        b = [0] * 50 + [1] * 50
        p = mcnemar_test(a, b)
        assert p < 0.05

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([1, 0], [1])

    def test_symmetric(self):
        a = [1, 0, 1, 0, 1]
        b = [0, 1, 0, 1, 0]
        p1 = mcnemar_test(a, b)
        p2 = mcnemar_test(b, a)
        assert p1 == pytest.approx(p2)
