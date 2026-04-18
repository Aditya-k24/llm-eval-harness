"""Command-line interface for the LLM Evaluation Harness."""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
import uuid
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="llm-eval",
    help="LLM Evaluation Harness — vendor-neutral LLM comparison on document-grounded QA.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------

@app.command("prepare-data")
def prepare_data(
    split: str = typer.Option("smoke", help="Which split to build: 'smoke' or 'dev'."),
    out_dir: str = typer.Option("datasets/public", help="Output directory for JSONL files."),
    manifests_dir: str = typer.Option("datasets/manifests", help="Where to write manifest JSON."),
) -> None:
    """Download benchmark data and build JSONL splits."""
    from .datasets.splits import build_smoke_split, build_dev_split
    from .datasets.manifests import create_manifest, save_manifest

    console.print(f"[bold cyan]Building '{split}' split...[/bold cyan]")

    if split == "smoke":
        jsonl_path = build_smoke_split(out_dir=out_dir)
        tasks = ["grounded_qa", "multihop_qa", "fever"]
    elif split == "dev":
        jsonl_path = build_dev_split(out_dir=out_dir)
        tasks = ["grounded_qa", "multihop_qa", "fever"]
    else:
        console.print(f"[red]Unknown split '{split}'. Choose 'smoke' or 'dev'.[/red]")
        raise typer.Exit(1)

    manifest = create_manifest(
        split=split,
        version="1.0",
        file_path=jsonl_path,
        tasks=tasks,
    )
    pathlib.Path(manifests_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = f"{manifests_dir}/{split}.json"
    save_manifest(manifest, manifest_path)

    console.print(
        f"[green]Split built:[/green] {jsonl_path} "
        f"({manifest.example_count} examples)"
    )
    console.print(f"[green]Manifest saved:[/green] {manifest_path}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command("run")
def run(
    split: str = typer.Option("smoke", help="Which split to evaluate: 'smoke' or 'dev'."),
    run_id: Optional[str] = typer.Option(None, help="Unique run identifier (auto-generated if omitted)."),
    concurrency: int = typer.Option(5, help="Maximum concurrent API calls."),
    models_config: str = typer.Option("configs/models.yaml", help="Path to models.yaml."),
    prompts_dir: str = typer.Option("prompts", help="Directory containing prompt templates."),
    data_dir: str = typer.Option("datasets/public", help="Directory with JSONL split files."),
    output_dir: str = typer.Option("reports", help="Directory to write raw JSONL results."),
) -> None:
    """Run the evaluation experiment against all configured models."""
    from .adapters import load_adapters
    from .datasets.loaders import load_jsonl
    from .runners.async_runner import run_experiment
    from .storage.jsonl_store import JSONLStore

    run_id = run_id or str(uuid.uuid4())[:8]
    jsonl_path = f"{data_dir}/{split}.jsonl"

    if not pathlib.Path(jsonl_path).exists():
        console.print(
            f"[red]Data file not found: {jsonl_path}[/red]\n"
            f"Run: llm-eval prepare-data --split {split}"
        )
        raise typer.Exit(1)

    console.print(f"[bold cyan]Run ID:[/bold cyan] {run_id}")
    console.print(f"[bold cyan]Loading examples from:[/bold cyan] {jsonl_path}")
    examples = load_jsonl(jsonl_path)
    console.print(f"  {len(examples)} examples loaded.")

    console.print(f"[bold cyan]Loading adapters from:[/bold cyan] {models_config}")
    adapters = load_adapters(models_config)
    console.print(f"  {len(adapters)} adapters: {[a.model_id for a in adapters]}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    store_path = f"{output_dir}/raw_{run_id}.jsonl"

    with JSONLStore(store_path) as store:
        console.print(
            f"[bold cyan]Starting experiment (concurrency={concurrency})...[/bold cyan]"
        )
        results = asyncio.run(
            run_experiment(
                examples=examples,
                adapters=adapters,
                run_id=run_id,
                prompts_dir=prompts_dir,
                store=store,
                concurrency=concurrency,
            )
        )

    console.print(
        f"[green]Done. {len(results)} results written to {store_path}[/green]"
    )

    # Write run_id to a temp file so `report` can find the latest run
    pathlib.Path(f"{output_dir}/.last_run_id").write_text(run_id)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command("report")
def report(
    run_id: Optional[str] = typer.Option(None, help="Run ID to report on (defaults to last run)."),
    output_dir: str = typer.Option("reports", help="Directory containing raw JSONL results."),
    audit_dir: str = typer.Option("reports/audit", help="Directory for audit queue JSON."),
    data_dir: str = typer.Option("datasets/public", help="Directory with JSONL split files."),
) -> None:
    """Compute metrics from raw results and save a Parquet report."""
    from .datasets.loaders import load_jsonl
    from .metrics.accuracy import compute_accuracy_metrics
    from .metrics.latency import compute_latency_stats
    from .metrics.hallucination import compute_hallucination_metrics
    from .storage.parquet_store import save_parquet
    from .annotation.audit import build_audit_queue

    # Resolve run_id
    last_id_file = pathlib.Path(output_dir) / ".last_run_id"
    if run_id is None:
        if last_id_file.exists():
            run_id = last_id_file.read_text().strip()
        else:
            console.print(
                "[red]No run_id provided and no .last_run_id file found.[/red]"
            )
            raise typer.Exit(1)

    raw_path = pathlib.Path(output_dir) / f"raw_{run_id}.jsonl"
    if not raw_path.exists():
        console.print(f"[red]Raw results file not found: {raw_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Loading raw results from:[/bold cyan] {raw_path}")
    import json
    raw_rows = [
        json.loads(line)
        for line in raw_path.read_text().strip().splitlines()
        if line.strip()
    ]
    console.print(f"  {len(raw_rows)} rows loaded.")

    # Build example lookup from all public JSONL files
    examples_by_id: dict = {}
    for jf in pathlib.Path(data_dir).glob("*.jsonl"):
        for ex in load_jsonl(str(jf)):
            examples_by_id[ex.id] = ex

    # Compute per-row accuracy metrics
    enriched_rows = []
    for row in raw_rows:
        ex = examples_by_id.get(row["example_id"])
        if ex is not None:
            acc = compute_accuracy_metrics(row, ex)
        else:
            acc = {
                "json_valid": False,
                "exact_match": 0,
                "token_f1": 0.0,
                "abstain_correct": 0,
                "evidence_quote_validity": 0.0,
                "label_correct": 0,
            }
        enriched_rows.append({**row, **acc})

    # Save parquet
    parquet_path = pathlib.Path(output_dir) / f"report_{run_id}.parquet"
    save_parquet(enriched_rows, str(parquet_path))
    console.print(f"[green]Parquet report saved:[/green] {parquet_path}")

    # Aggregate summary table
    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for row in enriched_rows:
        by_model[row["model_id"]].append(row)

    table = Table(title=f"Run {run_id} Summary", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("N")
    table.add_column("EM")
    table.add_column("F1")
    table.add_column("Label Acc")
    table.add_column("JSON Valid %")
    table.add_column("p50 ms")
    table.add_column("p95 ms")
    table.add_column("Total Cost $")

    for model_id, rows in sorted(by_model.items()):
        n = len(rows)
        em = sum(r.get("exact_match", 0) for r in rows) / n
        f1 = sum(r.get("token_f1", 0.0) for r in rows) / n
        lc = sum(r.get("label_correct", 0) for r in rows) / n
        jv = sum(1 for r in rows if r.get("json_valid")) / n
        latency_stats = compute_latency_stats(
            [r["end_to_end_ms"] for r in rows if r.get("end_to_end_ms") is not None]
        )
        total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in rows)
        table.add_row(
            model_id,
            str(n),
            f"{em:.2%}",
            f"{f1:.2%}",
            f"{lc:.2%}",
            f"{jv:.2%}",
            f"{latency_stats.get('p50_ms', 0):.0f}",
            f"{latency_stats.get('p95_ms', 0):.0f}",
            f"${total_cost:.4f}",
        )

    console.print(table)

    # Build audit queue
    audit_queue = build_audit_queue(
        raw_rows, examples_by_id, f"{audit_dir}/audit_{run_id}.json"
    )
    console.print(
        f"[yellow]Audit queue:[/yellow] {len(audit_queue)} items "
        f"-> {audit_dir}/audit_{run_id}.json"
    )


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@app.command("dashboard")
def dashboard(
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
) -> None:
    """Launch the Streamlit analytics dashboard."""
    app_path = pathlib.Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[bold cyan]Launching dashboard on port {port}...[/bold cyan]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
