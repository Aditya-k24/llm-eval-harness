"""Tests for adapter instantiation and cost estimation (no real API calls)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval_harness.adapters.base import ModelAdapter, ModelResult, TimingInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timing() -> TimingInfo:
    t = TimingInfo(t0=0.0, t1=0.1, t2=0.5, t3=1.0)
    return t


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def test_instantiation(self):
        from llm_eval_harness.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(model_id="claude-sonnet-4-6")
        assert adapter.model_id == "claude-sonnet-4-6"
        assert adapter.provider == "anthropic"

    def test_custom_params(self):
        from llm_eval_harness.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(
            model_id="claude-3-haiku-20240307",
            max_output_tokens=512,
            temperature=0.5,
            input_cost_per_mtok=0.25,
            output_cost_per_mtok=1.25,
        )
        assert adapter.max_output_tokens == 512
        assert adapter.temperature == 0.5

    def test_estimate_cost_usd(self):
        from llm_eval_harness.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(
            model_id="claude-sonnet-4-6",
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
        )
        cost = adapter.estimate_cost_usd(1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0)

    def test_implements_protocol(self):
        from llm_eval_harness.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(model_id="claude-sonnet-4-6")
        assert isinstance(adapter, ModelAdapter)

    @pytest.mark.asyncio
    async def test_generate_returns_model_result(self):
        from llm_eval_harness.adapters.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(model_id="claude-sonnet-4-6")

        # Build a mock response mimicking anthropic SDK structure
        mock_usage = MagicMock()
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 20
        mock_usage.cache_read_input_tokens = 0

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"answer": "Paris", "abstain": false, "evidence_quotes": []}'

        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_resp.usage = mock_usage
        mock_resp.model_dump.return_value = {}

        adapter._client = MagicMock()
        adapter._client.messages.create = AsyncMock(return_value=mock_resp)

        result = await adapter.generate("system", "user")
        assert isinstance(result, ModelResult)
        assert result.input_tokens == 50
        assert result.output_tokens == 20
        assert "Paris" in result.text


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def test_instantiation(self):
        from llm_eval_harness.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model_id="gpt-4o")
        assert adapter.model_id == "gpt-4o"
        assert adapter.provider == "openai"

    def test_estimate_cost_usd_no_cache(self):
        from llm_eval_harness.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(
            model_id="gpt-4o",
            input_cost_per_mtok=2.5,
            cached_input_cost_per_mtok=1.25,
            output_cost_per_mtok=10.0,
        )
        cost = adapter.estimate_cost_usd(1_000_000, 1_000_000, cached_input_tokens=0)
        assert cost == pytest.approx(2.5 + 10.0)

    def test_estimate_cost_usd_with_cache(self):
        from llm_eval_harness.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(
            model_id="gpt-4o",
            input_cost_per_mtok=2.5,
            cached_input_cost_per_mtok=1.25,
            output_cost_per_mtok=10.0,
        )
        cost = adapter.estimate_cost_usd(
            input_tokens=1_000_000,
            output_tokens=0,
            cached_input_tokens=500_000,
        )
        # 500k regular @ 2.5 + 500k cached @ 1.25
        assert cost == pytest.approx(1.25 + 0.625)

    def test_implements_protocol(self):
        from llm_eval_harness.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model_id="gpt-4o")
        assert isinstance(adapter, ModelAdapter)

    @pytest.mark.asyncio
    async def test_generate_returns_model_result(self):
        from llm_eval_harness.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model_id="gpt-4o")

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 40
        mock_usage.input_tokens_details = None

        mock_resp = MagicMock()
        mock_resp.output_text = '{"answer": "London", "abstain": false, "evidence_quotes": []}'
        mock_resp.usage = mock_usage
        mock_resp.model_dump.return_value = {}

        adapter._client = MagicMock()
        adapter._client.responses.create = AsyncMock(return_value=mock_resp)

        result = await adapter.generate("system", "user")
        assert isinstance(result, ModelResult)
        assert result.input_tokens == 100
        assert "London" in result.text


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------

class TestGeminiAdapter:
    @pytest.fixture(autouse=True)
    def set_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def test_instantiation(self):
        with patch("google.genai.Client"):
            from llm_eval_harness.adapters.gemini_adapter import GeminiAdapter

            adapter = GeminiAdapter(model_id="gemini-2.5-pro")
            assert adapter.model_id == "gemini-2.5-pro"
            assert adapter.provider == "gemini"

    def test_estimate_cost_usd(self):
        with patch("google.genai.Client"):
            from llm_eval_harness.adapters.gemini_adapter import GeminiAdapter

            adapter = GeminiAdapter(
                model_id="gemini-2.5-pro",
                input_cost_per_mtok=1.25,
                output_cost_per_mtok=10.0,
            )
            cost = adapter.estimate_cost_usd(1_000_000, 1_000_000)
            assert cost == pytest.approx(11.25)

    def test_implements_protocol(self):
        with patch("google.genai.Client"):
            from llm_eval_harness.adapters.gemini_adapter import GeminiAdapter

            adapter = GeminiAdapter(model_id="gemini-2.5-pro")
            assert isinstance(adapter, ModelAdapter)

    @pytest.mark.asyncio
    async def test_generate_returns_model_result(self):
        with patch("google.genai.Client"):
            from llm_eval_harness.adapters.gemini_adapter import GeminiAdapter

            adapter = GeminiAdapter(model_id="gemini-2.5-pro")

            mock_usage = MagicMock()
            mock_usage.prompt_token_count = 80
            mock_usage.candidates_token_count = 30
            mock_usage.cached_content_token_count = 0

            mock_resp = MagicMock()
            mock_resp.text = '{"verdict": "SUPPORTED", "evidence_quotes": [], "reasoning": "test"}'
            mock_resp.usage_metadata = mock_usage

            adapter._client = MagicMock()
            adapter._client.models.generate_content = MagicMock(return_value=mock_resp)

            result = await adapter.generate("system", "user")
            assert isinstance(result, ModelResult)
            assert result.input_tokens == 80
            assert "SUPPORTED" in result.text


# ---------------------------------------------------------------------------
# load_adapters factory
# ---------------------------------------------------------------------------

class TestLoadAdapters:
    def test_load_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
        monkeypatch.setenv("OPENAI_API_KEY", "ok")
        monkeypatch.setenv("GEMINI_API_KEY", "gk")

        yaml_content = """
models:
  - id: claude-sonnet-4-6
    provider: anthropic
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0
  - id: gpt-4o
    provider: openai
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 2.5
    cached_input_cost_per_mtok: 1.25
    output_cost_per_mtok: 10.0
"""
        cfg = tmp_path / "models.yaml"
        cfg.write_text(yaml_content)

        with patch("google.genai.Client"):
            from llm_eval_harness.adapters import load_adapters

            adapters = load_adapters(str(cfg))

        assert len(adapters) == 2
        assert adapters[0].model_id == "claude-sonnet-4-6"
        assert adapters[1].model_id == "gpt-4o"

    def test_unknown_provider_raises(self, tmp_path, monkeypatch):
        yaml_content = """
models:
  - id: foo-model
    provider: unknown_provider
    max_output_tokens: 256
    temperature: 0
"""
        cfg = tmp_path / "models.yaml"
        cfg.write_text(yaml_content)

        from llm_eval_harness.adapters import load_adapters

        with pytest.raises(ValueError, match="unknown_provider"):
            load_adapters(str(cfg))


# ---------------------------------------------------------------------------
# TimingInfo properties
# ---------------------------------------------------------------------------

class TestTimingInfo:
    def test_end_to_end_ms(self):
        t = TimingInfo(t0=0.0, t1=0.1, t2=0.5, t3=1.0)
        assert t.end_to_end_ms == pytest.approx(1000.0)

    def test_api_round_trip_ms(self):
        t = TimingInfo(t0=0.0, t1=0.1, t2=0.5, t3=1.0)
        # t3 - t1 = 0.9 s = 900 ms
        assert t.api_round_trip_ms == pytest.approx(900.0)
