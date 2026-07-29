"""CLI entry point for wasmagent-train-replay."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import click
from rich.console import Console
from rich.table import Table

from train_replay.cli.admin import admin
from train_replay.cli.safemode import SafeMode, SafeModeError

if TYPE_CHECKING:
    # Imported only for type checkers; the runtime imports live inside the
    # command body so this module stays importable before the package is built.
    from train_replay.anomaly.profile import TrainingProfile
    from train_replay.collector.flight_recorder import CollectiveEvent

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


@cli.command("agent-query")
@click.argument("dump_path", type=click.Path(exists=True))
@click.option("--tool", required=True, help="Agent tool to dispatch")
@click.option("--args", "args_json", default="{}", show_default=True, help="JSON tool arguments")
def agent_query(dump_path: str, tool: str, args_json: str) -> None:
    """Dispatch an agent tool against a Flight Recorder dump and print JSON."""
    from train_replay.agent.tools import dispatch_tool

    try:
        parsed_args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON for --args: {exc.msg}") from exc

    if not isinstance(parsed_args, dict):
        raise click.ClickException("--args must be a JSON object")

    try:
        result = dispatch_tool(tool, Path(dump_path), parsed_args)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(result, sort_keys=True))


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
@click.option(
    "--signing-key-hex",
    default=None,
    help="Raw Ed25519 private key as 64-char hex; signs the recorded bundle.",
)
@click.option(
    "--signing-key-id",
    default="dev-key",
    show_default=True,
    help="key_id stamped into the bundle signature envelope.",
)
@click.pass_context
def record(
    ctx: click.Context,
    dump_path: str,
    run_id: str,
    epoch: int,
    signing_key_hex: str | None,
    signing_key_id: str,
) -> None:
    """Record AEP evidence for all collectives in a Flight Recorder dump."""
    safe_mode: SafeMode = ctx.obj["safe_mode"]
    try:
        safe_mode.check("record")
    except SafeModeError as exc:
        raise click.ClickException(str(exc))

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.recording.recorder import EpochRecorder
    from train_replay.signing.signer import BundleSigner, load_private_key_hex

    events = load_flight_recorder(Path(dump_path))
    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = recorder.bundle()

    if signing_key_hex is not None:
        # ``load_private_key_hex`` keeps cryptography object construction out of
        # callers (and the CLI): a raw hex string becomes a signing key here.
        try:
            private_key = load_private_key_hex(signing_key_hex)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        BundleSigner(private_key, signing_key_id).sign(bundle)
        console.print(f"Signed bundle with key_id [cyan]{signing_key_id}[/cyan]")

    console.print(f"Recorded [cyan]{len(bundle.actions)}[/cyan] actions")
    console.print(f"Bundle digest: [bold]{bundle.digest()}[/bold]")
    if bundle.signature is not None:
        console.print(f"Signature: [bold]{bundle.signature['sig']}[/bold]")


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


# ---------------------------------------------------------------------------
# `train-replay anomaly` — batch statistical anomaly scanning.
#
# The Milestone 5 detector / alerting modules (``train_replay.anomaly.detector``,
# ``train_replay.alerting.notifier``) are tracked by sibling issues and are not
# on this branch yet.  To keep the subcommand self-contained and forward
# compatible, the baseline is loaded from a JSON :class:`TrainingProfile` and
# the Z-score scan + Slack delivery are implemented here against the public
# ``TrainingProfile`` contract; once the detector lands the body can delegate to
# ``StatisticalAnomalyDetector.detect`` without changing the CLI surface.
# ---------------------------------------------------------------------------


@dataclass
class _AnomalyHit:
    """One statistically outlying event surfaced by the batch scan."""

    rank: int
    step: int
    metric_name: str
    z_score: float
    severity: float
    description: str


def _profile_to_dict(profile: TrainingProfile) -> dict[str, Any]:
    """Serialise a :class:`TrainingProfile` to a JSON-friendly dict.

    ``ranks`` (a :class:`frozenset`) is emitted as a sorted list so the result
    round-trips through :func:`json.dumps` / :func:`_profile_from_dict`.
    """
    return {
        "interval_mean_ns": profile.interval_mean_ns,
        "interval_std_ns": profile.interval_std_ns,
        "interval_min_ns": profile.interval_min_ns,
        "interval_max_ns": profile.interval_max_ns,
        "interval_count": profile.interval_count,
        "tensor_size_mean": profile.tensor_size_mean,
        "tensor_size_std": profile.tensor_size_std,
        "tensor_size_min": profile.tensor_size_min,
        "tensor_size_max": profile.tensor_size_max,
        "tensor_count": profile.tensor_count,
        "collective_type_counts": dict(profile.collective_type_counts),
        "ranks": sorted(profile.ranks),
        "event_count": profile.event_count,
    }


def _profile_from_dict(data: dict[str, Any]) -> TrainingProfile:
    """Reconstruct a :class:`TrainingProfile` from :func:`_profile_to_dict` output."""
    from train_replay.anomaly.profile import TrainingProfile

    return TrainingProfile(
        interval_mean_ns=float(data.get("interval_mean_ns", 0.0)),
        interval_std_ns=float(data.get("interval_std_ns", 0.0)),
        interval_min_ns=int(data.get("interval_min_ns", 0)),
        interval_max_ns=int(data.get("interval_max_ns", 0)),
        interval_count=int(data.get("interval_count", 0)),
        tensor_size_mean=float(data.get("tensor_size_mean", 0.0)),
        tensor_size_std=float(data.get("tensor_size_std", 0.0)),
        tensor_size_min=int(data.get("tensor_size_min", 0)),
        tensor_size_max=int(data.get("tensor_size_max", 0)),
        tensor_count=int(data.get("tensor_count", 0)),
        collective_type_counts={
            str(k): int(v) for k, v in data.get("collective_type_counts", {}).items()
        },
        ranks=frozenset(int(r) for r in data.get("ranks", [])),
        event_count=int(data.get("event_count", 0)),
    )


def _load_profile(path: Path) -> TrainingProfile:
    """Load a JSON-serialised :class:`TrainingProfile` from *path*.

    Raises :class:`ValueError` when the file is not valid JSON or does not hold
    a JSON object; the caller translates that into a CLI error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Profile {path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Profile {path} must contain a JSON object")
    return _profile_from_dict(data)


def _scan_tensor_sizes(
    events: Sequence[CollectiveEvent],
    profile: TrainingProfile,
    threshold: float,
) -> list[_AnomalyHit]:
    """Flag events whose ``tensor_size`` Z-score exceeds *threshold*."""
    std = profile.tensor_size_std
    if std <= 0:
        return []
    mean = profile.tensor_size_mean
    hits: list[_AnomalyHit] = []
    for ev in events:
        z_score = (ev.tensor_size - mean) / std
        if abs(z_score) > threshold:
            hits.append(
                _AnomalyHit(
                    rank=ev.rank,
                    step=ev.sequence_id,
                    metric_name="tensor_size_zscore",
                    z_score=z_score,
                    severity=min(abs(z_score), 1.0),
                    description=(
                        f"tensor_size {ev.tensor_size}B on rank {ev.rank} step "
                        f"{ev.sequence_id} deviates from baseline mean {mean:.0f}B"
                    ),
                )
            )
    return hits


def _scan_timing(
    events: Sequence[CollectiveEvent],
    profile: TrainingProfile,
    threshold: float,
) -> list[_AnomalyHit]:
    """Flag per-rank inter-event intervals whose Z-score exceeds *threshold*."""
    std = profile.interval_std_ns
    if std <= 0:
        return []
    mean = profile.interval_mean_ns
    by_rank: dict[int, list[CollectiveEvent]] = {}
    for ev in events:
        by_rank.setdefault(ev.rank, []).append(ev)
    hits: list[_AnomalyHit] = []
    for rank, rank_events in by_rank.items():
        ordered = sorted(rank_events, key=lambda e: e.start_time_ns)
        for prev, curr in zip(ordered, ordered[1:]):
            gap = curr.start_time_ns - prev.start_time_ns
            if gap <= 0:
                continue
            z_score = (gap - mean) / std
            if abs(z_score) > threshold:
                hits.append(
                    _AnomalyHit(
                        rank=rank,
                        step=curr.sequence_id,
                        metric_name="timing_zscore",
                        z_score=z_score,
                        severity=min(abs(z_score), 1.0),
                        description=(
                            f"inter-event interval {gap}ns on rank {rank} step "
                            f"{curr.sequence_id} deviates from baseline mean {mean:.0f}ns"
                        ),
                    )
                )
    return hits


def _scan_anomalies(
    events: Sequence[CollectiveEvent],
    profile: TrainingProfile,
    threshold: float,
) -> list[_AnomalyHit]:
    """Run all batch Z-score scans; return hits ranked by absolute Z-score."""
    hits = _scan_tensor_sizes(events, profile, threshold)
    hits.extend(_scan_timing(events, profile, threshold))
    hits.sort(key=lambda hit: abs(hit.z_score), reverse=True)
    return hits


def _default_slack_opener(request: Request, timeout_seconds: float) -> Any:
    return urlopen(request, timeout=timeout_seconds)


def _send_slack_notification(
    webhook_url: str,
    message: str,
    *,
    timeout_seconds: float = 5.0,
    opener: Callable[[Request, float], Any] | None = None,
) -> None:
    """POST a Slack incoming-webhook payload carrying *message*.

    The HTTP opener is injectable so tests can capture the request without
    touching the network, mirroring :class:`PrometheusAnomalySource`.
    """
    payload = json.dumps({"text": message}).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    open_fn = opener or _default_slack_opener
    with open_fn(request, timeout_seconds) as response:
        response.read()


@cli.command()
@click.argument("dump_path", type=click.Path(exists=True))
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a JSON-serialised baseline TrainingProfile. When omitted a "
    "self-referential baseline is derived from the dump itself.",
)
@click.option(
    "--threshold",
    type=float,
    default=3.0,
    show_default=True,
    help="Absolute Z-score above which an event is flagged as anomalous.",
)
@click.option(
    "--notify",
    default=None,
    help="Alert target as '<channel>:<target>', e.g. 'slack:<webhook_url>'. "
    "Posts a summary when anomalies are found.",
)
def anomaly(
    dump_path: str,
    profile_path: str | None,
    threshold: float,
    notify: str | None,
) -> None:
    """Batch-scan a Flight Recorder dump for statistical anomalies.

    Events whose tensor-size or inter-event-timing Z-score exceeds
    ``--threshold`` (measured against the ``--profile`` baseline, or a baseline
    derived from the dump) are reported and optionally pushed to Slack.
    """
    from train_replay.anomaly.profile import TrainingProfile
    from train_replay.collector.flight_recorder import load_flight_recorder

    events = load_flight_recorder(Path(dump_path))
    console.print(f"Loaded [cyan]{len(events)}[/cyan] collective events")

    if profile_path is not None:
        try:
            profile = _load_profile(Path(profile_path))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        console.print(f"Baseline profile loaded from [cyan]{profile_path}[/cyan]")
    else:
        profile = TrainingProfile.fit_on_normal_run(events)
        console.print("Derived self-referential baseline from the dump")

    hits = _scan_anomalies(events, profile, threshold)
    console.print(f"Found [cyan]{len(hits)}[/cyan] anomalies (|z| > {threshold})")

    if hits:
        table = Table(title="Anomaly scan results")
        table.add_column("Rank", style="cyan")
        table.add_column("Step")
        table.add_column("Metric")
        table.add_column("Z-score")
        table.add_column("Severity")
        table.add_column("Description")
        for hit in hits:
            table.add_row(
                str(hit.rank),
                str(hit.step),
                hit.metric_name,
                f"{hit.z_score:+.2f}",
                f"{hit.severity:.2f}",
                hit.description,
            )
        console.print(table)

    if notify is not None:
        channel, _, target = notify.partition(":")
        if channel != "slack" or not target:
            raise click.ClickException(
                f"Unsupported --notify target {notify!r} "
                "(expected 'slack:<webhook_url>')"
            )
        if hits:
            message = (
                f"train-replay anomaly scan: {len(hits)} anomalies "
                f"(|z| > {threshold}) in {dump_path}"
            )
            try:
                _send_slack_notification(target, message)
            except URLError as exc:
                raise click.ClickException(f"Slack notification failed: {exc}") from exc
            console.print(f"Sent Slack alert to [cyan]{target}[/cyan]")
        else:
            console.print("No anomalies — Slack notification skipped.")
