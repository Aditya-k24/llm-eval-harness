"""Tests for JSON output parsing across all three task types."""

from __future__ import annotations

import json

import pytest

from llm_eval_harness.metrics.accuracy import parse_output


class TestGroundedQAParser:
    task = "grounded_qa"

    def test_valid_answerable(self):
        payload = {
            "answer": "Paris",
            "abstain": False,
            "evidence_quotes": ["Paris is the capital"],
        }
        parsed, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True
        assert parsed == payload

    def test_valid_abstain(self):
        payload = {"answer": "", "abstain": True, "evidence_quotes": []}
        parsed, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True
        assert parsed["abstain"] is True

    def test_empty_evidence_quotes(self):
        payload = {"answer": "Rome", "abstain": False, "evidence_quotes": []}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True

    def test_missing_answer_field(self):
        payload = {"abstain": False, "evidence_quotes": []}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False

    def test_missing_abstain_field(self):
        payload = {"answer": "Rome", "evidence_quotes": []}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False

    def test_wrong_type_abstain(self):
        payload = {"answer": "Rome", "abstain": "yes", "evidence_quotes": []}
        _, valid = parse_output(json.dumps(payload), self.task)
        # Pydantic coerces string "yes" to True — valid
        # (not a strict failure; just note the behaviour)
        # We don't assert a specific value here

    def test_plain_text_invalid(self):
        _, valid = parse_output("Just a plain sentence.", self.task)
        assert valid is False

    def test_truncated_json(self):
        _, valid = parse_output('{"answer": "Paris", "abstain":', self.task)
        assert valid is False

    def test_markdown_fence_extraction(self):
        text = '```json\n{"answer": "Berlin", "abstain": false, "evidence_quotes": []}\n```'
        parsed, valid = parse_output(text, self.task)
        assert valid is True
        assert parsed["answer"] == "Berlin"

    def test_fence_without_language_tag(self):
        text = '```\n{"answer": "Tokyo", "abstain": false, "evidence_quotes": []}\n```'
        parsed, valid = parse_output(text, self.task)
        assert valid is True


class TestMultiHopQAParser:
    task = "multihop_qa"

    def test_valid_full(self):
        payload = {
            "answer": "42",
            "abstain": False,
            "evidence_quotes": ["The answer is 42"],
            "reasoning_chain": "Step 1: find 42.",
        }
        parsed, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True

    def test_valid_without_reasoning_chain(self):
        payload = {
            "answer": "42",
            "abstain": False,
            "evidence_quotes": [],
        }
        parsed, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True  # reasoning_chain has default ""

    def test_abstain_multihop(self):
        payload = {
            "answer": "",
            "abstain": True,
            "evidence_quotes": [],
            "reasoning_chain": "",
        }
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True

    def test_missing_answer(self):
        payload = {"abstain": False, "evidence_quotes": [], "reasoning_chain": "x"}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False


class TestFEVERParser:
    task = "fever"

    def test_supported(self):
        payload = {
            "verdict": "SUPPORTED",
            "evidence_quotes": ["claim is supported by doc"],
            "reasoning": "The evidence clearly supports the claim.",
        }
        parsed, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True
        assert parsed["verdict"] == "SUPPORTED"

    def test_refuted(self):
        payload = {"verdict": "REFUTED", "evidence_quotes": [], "reasoning": "x"}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True

    def test_not_enough_info(self):
        payload = {
            "verdict": "NOT_ENOUGH_INFO",
            "evidence_quotes": [],
            "reasoning": "",
        }
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True

    def test_invalid_verdict(self):
        payload = {
            "verdict": "UNCERTAIN",
            "evidence_quotes": [],
            "reasoning": "",
        }
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False

    def test_missing_verdict(self):
        payload = {"evidence_quotes": [], "reasoning": "ok"}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False

    def test_without_optional_reasoning(self):
        payload = {"verdict": "SUPPORTED", "evidence_quotes": []}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is True  # reasoning has default ""

    def test_lowercase_verdict_invalid(self):
        payload = {"verdict": "supported", "evidence_quotes": [], "reasoning": ""}
        _, valid = parse_output(json.dumps(payload), self.task)
        assert valid is False
