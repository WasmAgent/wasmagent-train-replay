"""CLI entry point for wasmagent-train-replay."""

from __future__ import annotations

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
@click.argument("dump_path", type=click.Path(exists=True))
@click.option("--entity-id", "-e", default=None, help="Entity ID to trace causal ancestors for")
@click.option("--rank", "-r", type=int, default=None, help="Filter to specific rank")
@click.option(
    "--bundle-path", "-b",
    type=click.Path(exists=True),
    default=None,
    help="Evidence bundle JSON for suspicious-action analysis",
)
def replay(
    dump_path: str,
    entity_id: str | None,
    rank: int | None,
    bundle_path: str | None,
) -> None:
    """Replay an epoch to identify causal chains and suspicious actions."""

    import json

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
    from train_replay.recording.modes import RecordingMode
    from train_replay.replay.replayer import EpochReplayer

    events = load_flight_recorder(Path(dump_path))
    if rank is not None:
        events = [e for e in events if e.rank == rank]
    console.print(f"Loaded [cyan]{len(events)}[/cyan] collective events")

    graph = build_from_events(events)
    replayer = EpochReplayer(graph)

    if entity_id:
        ancestors = replayer.find_root_cause(entity_id)
        table = Table(title=f"Causal ancestors of {entity_id}")
        table.add_column("Activity ID", style="cyan")
        for a in ancestors:
            table.add_row(a)
        console.print(table)
    else:
        console.print(
            "[dim]No --entity-id provided; skipping root-cause analysis.[/dim]"
        )

    if bundle_path:
        with open(bundle_path) as f:
            raw = json.load(f)
        bundle = EpochEvidenceBundle(
            run_id=raw.get("run_id", ""),
            epoch=raw.get("epoch", 0),
            actions=[
                AEPRecord(
                    action_id=act.get("action_id", ""),
                    rank=act.get("rank", 0),
                    step=act.get("step", 0),
                    collective_type=act.get("collective_type", ""),
                    recording_mode=RecordingMode(
                        act.get("recording_mode", "validation"),
                    ),
                    timestamp_ns=act.get("timestamp_ns", 0),
                )
                for act in raw.get("actions", [])
            ],
        )
        suspicious = replayer.suspicious_actions(bundle)
        if rank is not None:
            suspicious = [s for s in suspicious if s.rank == rank]
        table = Table(title="Suspicious actions (FULL mode)")
        table.add_column("Action ID", style="red")
        table.add_column("Rank")
        table.add_column("Type")
        for s in suspicious:
            table.add_row(s.action_id, str(s.rank), s.collective_type)
        console.print(table)
        if not suspicious:
            console.print("[green]No suspicious actions found.[/green]")
    else:
        console.print(
            "[dim]No --bundle-path provided; skipping suspicious-action"
            " analysis.[/dim]"
        )

    if not entity_id and not bundle_path:
        console.print(
            "[yellow]Provide --entity-id and/or --bundle-path to analyze.[/yellow]"
        )


@cli.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--entity-id", "-e", default=None, help="Specific entity ID to trace")
@click.option(
    "--llm-endpoint",
    default="",
    envvar="LLM_ENDPOINT",
    help="OpenAI-compatible API base URL (default: empty, no LLM call).  "
    "Alternatively, set LLM_ENDPOINT env var.",
)
def analyze(
    bundle_path: str,
    entity_id: str | None,
    llm_endpoint: str,
) -> None:
    """Analyze an evidence bundle and generate a root-cause hypothesis.

    Uses causal ancestor traversal (PROV-DM graph) and optionally an LLM
    to produce a structured root-cause report.

    The LLM API key is read from the LLM_API_KEY environment variable.
    """

    import json

    from train_replay.agent_reasoner import AgentReasoner
    from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
    from train_replay.recording.modes import RecordingMode

    # Load the evidence bundle
    with open(bundle_path) as f:
        raw = json.load(f)

    bundle = EpochEvidenceBundle(
        run_id=raw.get("run_id", ""),
        epoch=raw.get("epoch", 0),
        actions=[
            AEPRecord(
                action_id=act.get("action_id", ""),
                rank=act.get("rank", 0),
                step=act.get("step", 0),
                collective_type=act.get("collective_type", ""),
                recording_mode=RecordingMode(
                    act.get("recording_mode", "validation"),
                ),
                timestamp_ns=act.get("timestamp_ns", 0),
            )
            for act in raw.get("actions", [])
        ],
    )

    # Build reasoner and analyze
    reasoner = AgentReasoner(llm_endpoint=llm_endpoint)
    report = reasoner.analyze(bundle=bundle, entity_id=entity_id)

    # Display results
    console.print("\n[bold]Root-Cause Analysis Report[/bold]")
    console.print("=" * 50)

    if report.root_cause_activity_ids:
        console.print(
            f"\n[bold red]Root cause activities:[/bold red]"
        )
        for aid in report.root_cause_activity_ids:
            console.print(f"  • {aid}")

    console.print(f"\n[bold]Description:[/bold] {report.root_cause_description}")
    console.print(f"[bold]Confidence:[/bold] {report.confidence}")

    if report.supporting_evidence:
        console.print("\n[bold]Supporting evidence:[/bold]")
        for ev in report.supporting_evidence:
            console.print(f"  • {ev}")

    if report.recommended_action:
        console.print(
            f"\n[bold]Recommended action:[/bold] {report.recommended_action}"
        )
