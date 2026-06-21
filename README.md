# LLM Eval Harness

A vendor-neutral evaluation harness for comparing large language models on document-grounded question answering. Run experiments across Claude, GPT-4o, and Gemini in a single command; get per-model accuracy, latency, cost, and hallucination metrics out of the box.

---

## Overview

| Dimension | Detail |
|-----------|--------|
| **Tasks** | Grounded QA (SQuAD 2.0), Multi-hop QA (HotpotQA), Fact Verification (ClimateFEVER) |
| **Models** | Anthropic Claude, OpenAI GPT-4o, Google Gemini (all configurable) |
| **Metrics** | Exact Match, Token F1, Label Accuracy, JSON validity, latency p50/p95, cost USD, hallucination rate |
| **Storage** | JSONL raw results → Parquet reports → DuckDB-queryable |
| **UI** | Streamlit dashboard with filters, scatter plots, latency bars, failure explorer |
| **Execution** | Async runner with bounded concurrency (`asyncio.Semaphore`) |

---

## Architecture

```
llm-eval prepare-data          # downloads from HuggingFace, writes JSONL splits
    ↓
llm-eval run                   # fans out (example × model) with async concurrency
    ↓                            stores raw JSONL per run
llm-eval report                # scores raw results, writes Parquet, prints table
    ↓
llm-eval dashboard             # Streamlit UI reads all Parquet reports
```

```
src/llm_eval_harness/
├── adapters/          # ModelAdapter protocol + Anthropic / OpenAI / Gemini impls
├── datasets/          # HuggingFace loaders, JSONL splits, manifest files
├── runners/           # async_runner: semaphore-bounded fan-out
├── metrics/           # accuracy (EM, F1), latency, cost, hallucination, significance
├── prompts/           # Jinja-style text templates per task
├── storage/           # JSONLStore (streaming writes), Parquet export
├── annotation/        # audit queue builder (flags responses for human review)
├── dashboard/         # Streamlit app (Polars + Plotly)
└── cli.py             # Typer CLI: prepare-data / run / report / dashboard
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
cp .env.example .env
# edit .env and fill in at least one key
```

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Models whose key is missing are skipped with a warning — you only need one key to run.

### 3. Run the smoke test (30 examples, ~2 min)

```bash
make smoke
# equivalent to:
llm-eval prepare-data --split smoke
llm-eval run --split smoke
llm-eval report
```

### 4. Full dev split (200 examples)

```bash
make dev-run
llm-eval report
```

---

## CLI Reference

### `llm-eval prepare-data`

Downloads benchmark data from HuggingFace and writes a JSONL split file.

```
Options:
  --split TEXT          smoke (30 ex) | dev (200 ex)   [default: smoke]
  --out-dir TEXT        Output directory for JSONL files [default: datasets/public]
  --manifests-dir TEXT  Where to write manifest JSON    [default: datasets/manifests]
```

### `llm-eval run`

Runs all configured models against a split. Results stream to `reports/raw_<run_id>.jsonl`.

```
Options:
  --split TEXT          smoke | dev                     [default: smoke]
  --run-id TEXT         Custom run ID (auto-generated if omitted)
  --concurrency INT     Max concurrent API calls        [default: 5]
  --models-config TEXT  Path to models.yaml             [default: configs/models.yaml]
  --prompts-dir TEXT    Directory with prompt templates [default: prompts]
  --data-dir TEXT       Directory with JSONL splits     [default: datasets/public]
  --output-dir TEXT     Directory for raw results       [default: reports]
```

### `llm-eval report`

Scores raw results, saves `reports/report_<run_id>.parquet`, prints a Rich summary table, and writes an audit queue.

```
Options:
  --run-id TEXT         Run ID (defaults to last run)
  --output-dir TEXT     Directory with raw JSONL        [default: reports]
  --audit-dir TEXT      Directory for audit queue JSON  [default: reports/audit]
  --data-dir TEXT       Directory with JSONL splits     [default: datasets/public]
```

### `llm-eval dashboard`

Launches the Streamlit analytics UI.

```
Options:
  --port INT            Streamlit server port           [default: 8501]
```

---

## Configuration

### `configs/models.yaml`

Add or remove models here. Each entry maps to an adapter class determined by `provider`.

```yaml
models:
  - id: claude-sonnet-4-6
    provider: anthropic
    display_name: "Claude Sonnet 4.6"
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0

  - id: gpt-4o
    provider: openai
    display_name: "GPT-4o"
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 2.5
    cached_input_cost_per_mtok: 1.25
    output_cost_per_mtok: 10.0

  - id: gemini-2.5-pro
    provider: gemini
    display_name: "Gemini 2.5 Pro"
    max_output_tokens: 256
    temperature: 0
    input_cost_per_mtok: 1.25
    output_cost_per_mtok: 10.0
```

Supported providers: `anthropic`, `openai`, `gemini`.

### Prompt templates

Each task has a `system.txt` and `user.txt` under `prompts/<task>/`. The user template receives `{context}` and `{question}` as variables.

```
prompts/
├── grounded_qa/    system.txt  user.txt
├── multihop_qa/    system.txt  user.txt
└── fever/          system.txt  user.txt
```

---

## Datasets

| Split | Size | Tasks |
|-------|------|-------|
| `smoke` | 30 examples (10 per task) | grounded_qa, multihop_qa, fever |
| `dev` | 200 examples (~67 per task) | grounded_qa, multihop_qa, fever |

**Sources (all via HuggingFace `datasets`):**
- `grounded_qa` — SQuAD 2.0 validation, balanced answerable / unanswerable
- `multihop_qa` — HotpotQA distractor dev set
- `fever` — ClimateFEVER test set, balanced SUPPORTED / REFUTED / NOT_ENOUGH_INFO

---

## Metrics

### Accuracy

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalized case-insensitive EM between predicted and gold answer |
| `token_f1` | Token-overlap F1 (SQuAD-style) |
| `label_correct` | FEVER verdict accuracy (SUPPORTED / REFUTED / NOT_ENOUGH_INFO) |
| `abstain_correct` | 1 if model correctly abstained on unanswerable question |
| `evidence_quote_validity` | Fraction of model-provided quotes found verbatim in the source context |
| `json_valid` | Whether the model output parses as a valid JSON matching the task schema |

### Latency

| Metric | Description |
|--------|-------------|
| `end_to_end_ms` | Wall time from before prompt assembly to after response parse |
| `api_round_trip_ms` | Wall time from API call send to response receive |
| `p50_ms` / `p95_ms` | Per-model latency percentiles across examples |

### Cost

Estimated USD cost per call based on token counts × per-million-token rates from `configs/models.yaml`. Cached tokens billed at the reduced `cached_input_cost_per_mtok` rate (OpenAI only by default).

### Hallucination

Flagged when:
- Question is unanswerable (`is_answerable=False`) but model provides a non-empty answer without abstaining, **or**
- Evidence quotes provided by the model cannot be found verbatim in the source context.

### Statistical significance

`metrics/significance.py` exposes:
- `bootstrap_ci(values, stat_fn, n_boot=1000, alpha=0.05)` — bootstrap confidence interval
- `mcnemar_test(correct_a, correct_b)` — continuity-corrected McNemar's test p-value

---

## Expected Output

### `llm-eval run`

```
Run ID: a1b2c3d4
Loading examples from: datasets/public/smoke.jsonl
  30 examples loaded.
Loading adapters from: configs/models.yaml
  2 adapters: ['claude-sonnet-4-6', 'gpt-4o']
Starting experiment (concurrency=5)...
Done. 60 results written to reports/raw_a1b2c3d4.jsonl
```

### `llm-eval report` — Rich summary table

```
                      Run a1b2c3d4 Summary
┌──────────────────┬────┬────────┬────────┬───────────┬─────────────┬─────────┬─────────┬─────────────┐
│ Model            │ N  │ EM     │ F1     │ Label Acc │ JSON Valid %│ p50 ms  │ p95 ms  │ Total Cost $│
├──────────────────┼────┼────────┼────────┼───────────┼─────────────┼─────────┼─────────┼─────────────┤
│ claude-sonnet-4-6│ 30 │ 62.50% │ 71.20% │ 80.00%    │ 96.67%      │ 1240    │ 2890    │ $0.0184     │
│ gpt-4o           │ 30 │ 58.33% │ 68.40% │ 76.67%    │ 100.00%     │ 980     │ 2150    │ $0.0091     │
└──────────────────┴────┴────────┴────────┴───────────┴─────────────┴─────────┴─────────┴─────────────┘
Audit queue: 4 items -> reports/audit/audit_a1b2c3d4.json
```

### Raw result row (`reports/raw_<run_id>.jsonl`)

```json
{
  "_ts": "2026-06-21T12:00:00.000000",
  "run_id": "a1b2c3d4",
  "example_id": "squad_57263cfc...",
  "task": "grounded_qa",
  "model_id": "claude-sonnet-4-6",
  "provider": "anthropic",
  "raw_text": "{\"answer\": \"Super Bowl 50\", \"abstain\": false, \"evidence_quotes\": [\"Super Bowl 50 was an American football game\"]}",
  "input_tokens": 412,
  "output_tokens": 47,
  "cached_input_tokens": 0,
  "estimated_cost_usd": 0.000942,
  "end_to_end_ms": 1238.5,
  "api_round_trip_ms": 1201.3,
  "attempt_count": 1,
  "error": null
}
```

### Parquet report columns (`reports/report_<run_id>.parquet`)

All raw result fields plus:

```
json_valid · exact_match · token_f1 · abstain_correct · evidence_quote_validity · label_correct
```

### Audit queue item (`reports/audit/audit_<run_id>.json`)

```json
[
  {
    "example_id": "squad_abc123",
    "model_id": "gpt-4o",
    "task": "grounded_qa",
    "question": "Who won the award?",
    "gold_answer": "",
    "model_answer": "John Smith",
    "evidence_quotes": ["John Smith received the award"],
    "raw_text": "...",
    "review_reasons": ["answered_unanswerable"],
    "annotation_status": "pending"
  }
]
```

Possible `review_reasons`: `invalid_json`, `answered_unanswerable`, `unverified_quotes`.

### Streamlit dashboard (`llm-eval dashboard`)

Three-column layout:
- **Scorecard** — per-model EM, F1, Label Accuracy, JSON validity %, hallucination rate
- **Accuracy vs Cost** — scatter plot (mean cost/call vs mean EM per model)
- **Latency** — grouped bar chart (p50 and p95 per model)
- **Failure Explorer** — table of rows where `json_valid = False`
- **Raw data** — expandable full dataframe

---

## Docker

```bash
# Build image
make docker-build

# Run smoke experiment + dashboard via Docker Compose
make docker-run
# or
docker-compose up
```

`docker-compose.yml` spins up two services: `runner` (runs the experiment) and `dashboard` (Streamlit on port 8501). Both mount `./reports` and `./datasets` as volumes so results persist on the host.

---

## Development

```bash
make install    # pip install -e ".[dev]"
make lint       # ruff check src/ tests/
make test       # pytest tests/ -v
make smoke      # end-to-end smoke run
```

### Adding a new model provider

1. Create `src/llm_eval_harness/adapters/<name>_adapter.py` implementing the `ModelAdapter` protocol (`generate`, `estimate_cost_usd`).
2. Register it in `adapters/__init__.py`: add to `_PROVIDER_MAP` and `_PROVIDER_ENV`.
3. Add model entries to `configs/models.yaml` with `provider: <name>`.

### Adding a new task

1. Add `prompts/<task>/system.txt` and `user.txt`.
2. Add a Pydantic output schema in `metrics/accuracy.py`.
3. Add the task name to `configs/prompts.yaml` and wire up scoring in `compute_accuracy_metrics`.

---

## Project Structure

```
LLM_eval/
├── configs/
│   ├── models.yaml          # model definitions and pricing
│   ├── prompts.yaml         # task → template dir mapping
│   └── retry.yaml           # tenacity retry settings
├── datasets/
│   ├── manifests/           # split manifest JSON files
│   └── public/              # generated JSONL split files
├── docker/
│   └── entrypoint.sh
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── prompts/
│   ├── grounded_qa/
│   ├── multihop_qa/
│   └── fever/
├── reports/                 # raw JSONL + Parquet + audit JSON output
├── src/llm_eval_harness/    # main package
└── tests/
```

---

## Requirements

- Python 3.11+
- At least one of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
- Internet access to download datasets from HuggingFace on first run
