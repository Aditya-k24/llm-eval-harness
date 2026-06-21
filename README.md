# LLM Eval Harness

Compare Claude, GPT-4o, and Gemini on document-grounded QA tasks — accuracy, latency, cost, and hallucination rate — from a single CLI.

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Add API keys (only one required)
cp .env.example .env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY

# Run smoke test (30 examples, ~2 min)
make smoke
```

`make smoke` runs three commands in sequence:

```bash
llm-eval prepare-data --split smoke   # download datasets from HuggingFace
llm-eval run --split smoke            # query all configured models
llm-eval report                       # score results, print summary table
```

---

## Tasks

| Task | Dataset | What it measures |
|------|---------|-----------------|
| `grounded_qa` | SQuAD 2.0 | Answer from context; abstain if unanswerable |
| `multihop_qa` | HotpotQA | Multi-document reasoning |
| `fever` | ClimateFEVER | Fact verification (SUPPORTED / REFUTED / NOT_ENOUGH_INFO) |

Two splits: **smoke** (30 examples) and **dev** (200 examples).

---

## Output

### Terminal — summary table after `llm-eval report`

```
                      Run a1b2c3d4 Summary
┌──────────────────┬────┬────────┬────────┬───────────┬─────────────┬─────────┬─────────┬─────────────┐
│ Model            │ N  │ EM     │ F1     │ Label Acc │ JSON Valid %│ p50 ms  │ p95 ms  │ Total Cost $│
├──────────────────┼────┼────────┼────────┼───────────┼─────────────┼─────────┼─────────┼─────────────┤
│ claude-sonnet-4-6│ 30 │ 62.50% │ 71.20% │ 80.00%    │ 96.67%      │ 1240    │ 2890    │ $0.0184     │
│ gpt-4o           │ 30 │ 58.33% │ 68.40% │ 76.67%    │ 100.00%     │  980    │ 2150    │ $0.0091     │
└──────────────────┴────┴────────┴────────┴───────────┴─────────────┴─────────┴─────────┴─────────────┘
Audit queue: 4 items -> reports/audit/audit_a1b2c3d4.json
```

**Metrics:** Exact Match, Token F1, Label Accuracy (FEVER), JSON validity rate, latency percentiles, total USD cost.

### Files written

| File | Contents |
|------|----------|
| `reports/raw_<run_id>.jsonl` | One JSON line per (example × model) — raw text, tokens, latency, cost |
| `reports/report_<run_id>.parquet` | Same rows + computed accuracy metrics |
| `reports/audit/audit_<run_id>.json` | Responses flagged for human review (invalid JSON, hallucinations, unverified quotes) |

### Dashboard

```bash
llm-eval dashboard        # opens Streamlit on localhost:8501
```

Three-panel view: scorecard per model, accuracy-vs-cost scatter, latency bar chart. Failure explorer table for JSON parse errors.

---

## Configuration

Edit `configs/models.yaml` to add/remove models. Only models whose API key is set in `.env` are loaded.

```yaml
models:
  - id: claude-sonnet-4-6
    provider: anthropic          # anthropic | openai | gemini
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0
```

---

## Docker

```bash
make docker-build
make docker-run              # runs smoke experiment + dashboard (port 8501)
```

---

## Dev

```bash
make lint    # ruff
make test    # pytest
```
