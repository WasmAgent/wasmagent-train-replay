"""CLI entry point for wasmagent-train-replay."""

from __future__ import annotations

import os
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Resolved explicitly (rather than click's runtime `package_name=`) because the
# click type stub bundled here does not declare that kwarg; `version=` is
# stub-safe and still tracks pyproject.toml via installed dist metadata.
try:
    _VERSION = _dist_version("wasmagent-train-replay")
except PackageNotFoundError:  # source checkout without `pip install`
    _VERSION = "0.0.0+unknown"

console = Console()


@click.group()
@click.version_option(version=_VERSION)
def cli() -> None:
    """wasmagent-train-replay — causal evidence layer for distributed GPU training."""


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True))
@click.option("--rank", "-r", type=int, default=None, help="Filter to specific rank")
def ingest(dump_path: str, rank: int | None) -> None:
    """Ingest a PyTorch Flight Recorder dump and build the causal graph."""
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events

    dump = Path(dump_path)
    console.print(f"[bold]Loading[/bold] {dump}")
    events = load_flight_recorder(dump)
    if rank is not None:
        events = [e for e in events if e.rank == rank]
    console.print(f"Loaded [cyan]{len(events)}[/cyan] collective events")

    graph = build_from_events(events)
    nodes = list(graph.nodes())
    console.print(f"Built causal graph with [cyan]{len(nodes)}[/cyan] nodes")


@cli.command()
@click.argument("entity_id")
@click.argument("dump_path", type=click.Path(exists=True))
def trace(entity_id: str, dump_path: str) -> None:
    """Trace the causal ancestors of a tensor entity."""
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.replay.replayer import EpochReplayer

    events = load_flight_recorder(Path(dump_path))
    graph = build_from_events(events)
    replayer = EpochReplayer(graph)
    ancestors = replayer.find_root_cause(entity_id)

    table = Table(title=f"Causal ancestors of {entity_id}")
    table.add_column("Activity ID", style="cyan")
    for a in ancestors:
        table.add_row(a)
    console.print(table)


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True))
@click.option("--run-id", default="dev-run", show_default=True)
@click.option("--epoch", default=0, type=int, show_default=True)
def record(dump_path: str, run_id: str, epoch: int) -> None:
    """Record AEP evidence for all collectives in a Flight Recorder dump."""

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.recording.recorder import EpochRecorder

    events = load_flight_recorder(Path(dump_path))
    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = recorder.bundle()
    console.print(f"Recorded [cyan]{len(bundle.actions)}[/cyan] actions")
    console.print(f"Bundle digest: [bold]{bundle.digest()}[/bold]")


@cli.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--entity-id", required=True, help="Anomalous tensor entity ID to analyze")
@click.option("--dump-path", type=click.Path(exists=True), required=True,
              help="Flight Recorder dump for causal graph construction")
@click.option("--llm-endpoint", default="http://localhost:8000/v1/chat/completions",
              show_default=True, help="OpenAI-compatible LLM endpoint")
@click.option("--model", default="gpt-4o-mini", show_default=True, help="LLM model name")
@click.option(
    "--api-key", default="",
    help="[DEPRECATED] use LLM_API_KEY env var (avoids shell history leaks)",
)
@click.option("--rank", type=int, default=None, help="Filter suspicious actions to a specific rank")
def analyze(
    bundle_path: str,
    entity_id: str,
    dump_path: str,
    llm_endpoint: str,
    model: str,
    api_key: str,
    rank: int | None,
) -> None:
    """Analyze an evidence bundle with LLM-assisted root-cause hypothesis generation."""

    import json

    from train_replay.agent_reasoner import analyze_bundle
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle

    # Resolve API key: env var takes precedence; --api-key flag is deprecated.
    effective_key = os.environ.get("LLM_API_KEY", "")
    if api_key:
        if effective_key:
            console.print(
                "[yellow]Warning:[/yellow] --api-key ignored; "
                "LLM_API_KEY env var takes precedence."
            )
        else:
            effective_key = api_key
            warnings.warn(
                "Passing --api-key via CLI flag is deprecated and may leak to shell history. "
                "Set the LLM_API_KEY environment variable instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    console.print(f"[bold]Loading[/bold] evidence bundle from {bundle_path}")
    with open(bundle_path) as f:
        bundle_data = json.load(f)
    # Convert nested dicts to proper AEPRecord instances
    actions = [AEPRecord(**a) for a in bundle_data.pop("actions", [])]
    bundle = EpochEvidenceBundle(actions=actions, **bundle_data)

    console.print(f"[bold]Building[/bold] causal graph from {dump_path}")
    events = load_flight_recorder(Path(dump_path))
    graph = build_from_events(events)

    console.print(f"[bold]Analyzing[/bold] entity {entity_id} …")
    report = analyze_bundle(
        bundle,
        graph,
        entity_id,
        rank=rank,
        llm_endpoint=llm_endpoint,
        model=model,
        api_key=effective_key,
    )

    console.print("\n[bold]Root-Cause Report[/bold]")
    console.print(f"  Anomaly type:  [cyan]{report.anomaly_type}[/cyan]")
    console.print(f"  Summary:       {report.summary}")

    if report.hypotheses:
        for i, hyp in enumerate(report.hypotheses, 1):
            console.print(f"\n  Hypothesis {i} (confidence {hyp.confidence:.0%}):")
            console.print(f"    {hyp.description}")
            if hyp.affected_ranks:
                console.print(f"    Affected ranks: {hyp.affected_ranks}")
            if hyp.evidence_activity_ids:
                console.print(f"    Evidence: {hyp.evidence_activity_ids}")
