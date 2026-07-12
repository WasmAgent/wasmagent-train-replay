"""Admin CLI subcommands for wasmagent-train-replay.

Currently provides:
  - admin safe-mode --on|--off|--status   : manage safe mode
"""

from __future__ import annotations

import click
from rich.console import Console

from train_replay.cli.safemode import SafeMode

console = Console()


@click.group()
def admin() -> None:
    """Operator administration commands."""


@admin.command(name="safe-mode")
@click.option("--on", "activate", is_flag=True, default=False,
              help="Activate safe mode (lock the system).")
@click.option("--off", "deactivate", is_flag=True, default=False,
              help="Deactivate safe mode (unlock the system).")
@click.option("--status", "show_status", is_flag=True, default=False,
              help="Show whether safe mode is active.")
@click.pass_context
def safe_mode(
    ctx: click.Context,
    activate: bool,
    deactivate: bool,
    show_status: bool,
) -> None:
    """Query or change safe mode.

    Safe mode blocks side-effecting operations (recording, replaying, etc.)
    until an operator explicitly clears it.

    Examples:

        train-replay admin safe-mode --on

        train-replay admin safe-mode --status

        train-replay admin safe-mode --off
    """
    safe: SafeMode = ctx.obj["safe_mode"]
    if activate and deactivate:
        console.print("[red]--on and --off are mutually exclusive.[/red]")
        raise click.Abort()

    if activate:
        safe.trigger()
        console.print("[green]Safe mode activated.[/green]")
        return

    if deactivate:
        safe.clear()
        console.print("[green]Safe mode deactivated.[/green]")
        return

    # Default: show current status
    active = safe.status()
    if active:
        console.print("[yellow]Safe mode is ON[/yellow]")
    else:
        console.print("[green]Safe mode is OFF[/green]")
    return
