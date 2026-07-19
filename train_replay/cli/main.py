"""CLI entry point for wasmagent-train-replay."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from train_replay.cli.admin import admin
from train_replay.cli.safemode import SafeMode, SafeModeError

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
@click.pass_context
def cli(ctx: click.Context) -> None:
    """wasmagent-train-replay — causal evidence layer for distributed GPU training."""
    # Each CLI invocation carries its own SafeMode through the command tree via
    # the click context. ``ensure_object`` keeps an ``obj`` that a caller (or a
    # test harness) passed in via ``CliRunner.invoke(..., obj=...)`` so the
    # instance is shared within a single invocation but never across them.
    ctx.ensure_object(dict)
    if "safe_mode" not in ctx.obj:
        ctx.obj["safe_mode"] = SafeMode()


cli.add_command(admin)


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
@click.pass_context
def record(ctx: click.Context, dump_path: str, run_id: str, epoch: int) -> None:
    """Record AEP evidence for all collectives in a Flight Recorder dump."""
    safe_mode: SafeMode = ctx.obj["safe_mode"]
    try:
        safe_mode.check("record")
    except SafeModeError as exc:
        raise click.ClickException(str(exc))

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
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "cbor"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Evidence bundle serialization format.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Destination file for the signed evidence bundle.",
)
@click.option(
    "--sign-key",
    required=True,
    help="Raw Ed25519 private key as a 64-character hex string.",
)
@click.option("--run-id", default="dev-run", show_default=True)
@click.option("--epoch", default=0, type=int, show_default=True)
@click.pass_context
def export(
    ctx: click.Context,
    dump_path: str,
    output_format: str,
    output_path: str,
    sign_key: str,
    run_id: str,
    epoch: int,
) -> None:
    """Export a signed AEP evidence bundle to JSON or CBOR."""
    safe_mode: SafeMode = ctx.obj["safe_mode"]
    try:
        safe_mode.check("export")
    except SafeModeError as exc:
        raise click.ClickException(str(exc))

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.recording.recorder import EpochRecorder
    from train_replay.signing.signer import BundleSigner

    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(sign_key))
    except ValueError as exc:
        raise click.ClickException(
            "--sign-key must be a 64-character hex Ed25519 private key"
        ) from exc

    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = hashlib.sha256(public_key_bytes).hexdigest()[:16]

    events = load_flight_recorder(Path(dump_path))
    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = BundleSigner(private_key, key_id=key_id).sign(recorder.bundle())

    output = Path(output_path)
    if output_format.lower() == "json":
        output.write_text(bundle.to_json(), encoding="utf-8")
    else:
        output.write_bytes(bundle.to_cbor())

    console.print(f"Exported [cyan]{len(bundle.actions)}[/cyan] actions to {output}")
    console.print(f"Bundle digest: [bold]{bundle.digest()}[/bold]")
    console.print(f"Signature key id: [bold]{key_id}[/bold]")


@cli.command()
@click.pass_context
def resume(ctx: click.Context) -> None:
    """Exit safe mode and resume normal operation.

    This is the operator's quick way to clear a safe-mode lock without
    specifying the full ``admin safe-mode --off`` subcommand.
    """
    safe_mode: SafeMode = ctx.obj["safe_mode"]
    safe_mode.clear()
    console.print("[green]Safe mode cleared — normal operation resumed.[/green]")


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True))
@click.argument("entity_id")
@click.option("--rank", "-r", type=int, default=0, help="Rank to replay")
@click.option("--run-id", default="dev-run", show_default=True)
@click.option("--epoch", default=0, type=int, show_default=True)
@click.pass_context
def replay(
    ctx: click.Context,
    dump_path: str,
    entity_id: str,
    rank: int,
    run_id: str,
    epoch: int,
) -> None:
    """Replay an epoch and trace causal chains for a tensor entity."""
    safe_mode: SafeMode = ctx.obj["safe_mode"]
    try:
        safe_mode.check("replay")
    except SafeModeError as exc:
        raise click.ClickException(str(exc))

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.recording.recorder import EpochRecorder
    from train_replay.replay.replayer import EpochReplayer

    events = load_flight_recorder(Path(dump_path))
    graph = build_from_events(events)
    replayer = EpochReplayer(graph)

    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = recorder.bundle()

    result = replayer.replay_rank(bundle, rank, entity_id)

    console.print(f"[bold]Replay Result[/bold] for epoch {result.epoch}, rank {result.rank}")
    console.print(f"Causal ancestors of [cyan]{entity_id}[/cyan]:")
    for anc in result.causal_ancestors:
        console.print(f"  {anc}")
    console.print(f"\nSuspicious actions ([cyan]{len(result.suspicious_actions)}[/cyan]):")
    for a in result.suspicious_actions:
        info = (
            f"  rank={a.rank} step={a.step}"
            f" type={a.collective_type} mode={a.recording_mode}"
        )
        console.print(info)
