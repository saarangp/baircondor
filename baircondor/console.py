"""Shared stderr console and log line prefix."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

console = Console(stderr=True, soft_wrap=True)
PREFIX = f"[dim]{escape('[baircondor]')}[/dim]"


def log(msg: str, quiet: bool = False, style: str | None = None) -> None:
    if quiet:
        return
    text = escape(msg)
    console.print(f"{PREFIX} [{style}]{text}[/{style}]" if style else f"{PREFIX} {text}")
