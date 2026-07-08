"""CLI entry point for wasmagent-train-replay."""

from __future__ import annotations
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """wasmagent-train-replay — causal evidence layer for distributed GPU training."""


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True, path_type=Path))
@click.option("--rank", "-r", type=int, default=None, help="Filter to specific rank")
def ingest(dump_path: Path, rank: int | None) -> None:
    """Ingest a PyTorch Flight Recorder dump and build the causal graph."""
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events

    console.print(f"[bold]Loading[/bold] {dump_path}")
    events = load_flight_recorder(dump_path)
    if rank is not None:
        events = [e for e in events if e.rank == rank]
    console.print(f"Loaded [cyan]{len(events)}[/cyan] collective events")

    graph = build_from_events(events)
    nodes = list(graph.nodes())
    console.print(f"Built causal graph with [cyan]{len(nodes)}[/cyan] nodes")


@cli.command()
@click.argument("entity_id")
@click.argument("dump_path", type=click.Path(exists=True, path_type=Path))
def trace(entity_id: str, dump_path: Path) -> None:
    """Trace the causal ancestors of a tensor entity."""
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.replay.replayer import EpochReplayer

    events = load_flight_recorder(dump_path)
    graph = build_from_events(events)
    replayer = EpochReplayer(graph)
    ancestors = replayer.find_root_cause(entity_id)

    table = Table(title=f"Causal ancestors of {entity_id}")
    table.add_column("Activity ID", style="cyan")
    for a in ancestors:
        table.add_row(a)
    console.print(table)


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-id", default="dev-run", show_default=True)
@click.option("--epoch", default=0, type=int, show_default=True)
def record(dump_path: Path, run_id: str, epoch: int) -> None:
    """Record AEP evidence for all collectives in a Flight Recorder dump."""
    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.recording.recorder import EpochRecorder
    import json, dataclasses

    events = load_flight_recorder(dump_path)
    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = recorder.bundle()
    console.print(f"Recorded [cyan]{len(bundle.actions)}[/cyan] actions")
    console.print(f"Bundle digest: [bold]{bundle.digest()}[/bold]")
