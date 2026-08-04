"""Interactive browser for recent runs: list -> enter -> live-tailing detail view."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, RichLog, Static, TabbedContent, TabPane
from textual.worker import get_current_worker

from .history import STATUS_COLORS, get_job_status

TAIL_LINES = 500
TAIL_BYTES = 256 * 1024

LOG_FILES = [
    ("stdout", "stdout.txt"),
    ("stderr", "stderr.txt"),
    ("condor", "condor.log"),
]

# statuses where the job may still produce output / can be killed
_ACTIVE_STATUSES = ("?", "idle", "running", "held")


def read_tail(path: Path, max_lines: int = TAIL_LINES, max_bytes: int = TAIL_BYTES) -> str | None:
    """Last max_lines of a file, reading at most max_bytes; None if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return None
    lines = data.decode(errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # drop the partial first line
    return "\n".join(lines[-max_lines:])


def status_text(status: str) -> Text:
    return Text(f"● {status}", style=STATUS_COLORS.get(status, "dim"))


def entry_timestamp(entry: dict) -> str:
    return entry.get("timestamp", "")[:16].replace("T", " ")


def entry_command(entry: dict, max_len: int = 80) -> str:
    cmd = " ".join(entry.get("command", []))
    return cmd if len(cmd) <= max_len else cmd[: max_len - 3] + "..."


class ConfirmKill(ModalScreen[bool]):
    """y/n confirmation before condor_rm."""

    BINDINGS = [
        Binding("y", "dismiss(True)", "yes"),
        Binding("n,escape", "dismiss(False)", "no"),
    ]

    DEFAULT_CSS = """
    ConfirmKill { align: center middle; }
    ConfirmKill > Static {
        width: auto; max-width: 80%; padding: 1 2;
        background: $surface; border: thick $error;
    }
    """

    def __init__(self, jobname: str, cluster_id: str) -> None:
        super().__init__()
        self._label = f"condor_rm {cluster_id} ({jobname})?  [b]y[/b]es / [b]n[/b]o"

    def compose(self) -> ComposeResult:
        yield Static(self._label)


class RunDetailScreen(Screen):
    """One run: status header + tabbed, live-tailing stdout/stderr/condor.log."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "back"),
        Binding("left", "app.pop_screen", "back", priority=True),
        Binding("1", "show_tab('stdout')", "stdout", show=False),
        Binding("2", "show_tab('stderr')", "stderr", show=False),
        Binding("3", "show_tab('condor')", "condor.log", show=False),
        Binding("k", "kill", "kill job"),
        Binding("y", "copy_path", "copy run dir"),
    ]

    DEFAULT_CSS = """
    RunDetailScreen > Static { padding: 0 1; height: auto; }
    RunDetailScreen > TabbedContent { height: 1fr; }
    RichLog { padding: 0 1; }
    """

    def __init__(self, entry: dict, status: str) -> None:
        super().__init__()
        self._entry = entry
        self._status = status
        self._run_dir = Path(entry.get("run_dir", ""))
        self._shown: dict[str, str | None] = {}

    def compose(self) -> ComposeResult:
        yield Static(self._header(), id="detail-header")
        with TabbedContent():
            for tab_id, filename in LOG_FILES:
                with TabPane(filename, id=tab_id):
                    yield RichLog(wrap=True, highlight=False, markup=False, id=f"log-{tab_id}")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_log("stdout")
        self.set_interval(2.0, self._refresh_active_log)
        self.set_interval(5.0, self._refresh_status)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # hidden panes have zero width, so (re)render a tab when it becomes visible
        tab_id = event.pane.id
        self._shown.pop(tab_id, None)
        self.call_after_refresh(self._refresh_log, tab_id)

    def _header(self) -> Text:
        e = self._entry
        header = Text()
        header.append(f"{e.get('jobname', '?')}  ", style="bold")
        header.append_text(status_text(self._status))
        header.append(f"  gpus={e.get('gpus', 0)}  [{entry_timestamp(e)}]\n", style="dim")
        header.append(f"$ {entry_command(e)}\n", style="dim")
        header.append(str(self._run_dir), style="dim cyan")
        return header

    def _refresh_active_log(self) -> None:
        active = self.query_one(TabbedContent).active
        if active:
            self._refresh_log(active)

    def _refresh_log(self, tab_id: str) -> None:
        filename = dict(LOG_FILES)[tab_id]
        content = read_tail(self._run_dir / filename)
        if tab_id in self._shown and content == self._shown[tab_id]:
            return  # avoid resetting scroll position when nothing changed
        self._shown[tab_id] = content
        log = self.query_one(f"#log-{tab_id}", RichLog)
        log.clear()
        if content is None:
            log.write(f"({filename} not readable from this host)")
        elif content:
            log.write(content)

    @work(thread=True, exclusive=True, group="detail-status")
    def _refresh_status(self) -> None:
        if self._status not in _ACTIVE_STATUSES:
            return
        status = get_job_status(self._entry.get("cluster_id"))
        if not get_current_worker().is_cancelled:
            self.app.call_from_thread(self._set_status, status)

    def _set_status(self, status: str) -> None:
        self._status = status
        self.query_one("#detail-header", Static).update(self._header())

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def action_copy_path(self) -> None:
        self.app.copy_to_clipboard(str(self._run_dir))
        self.notify("run dir path copied to clipboard")

    def action_kill(self) -> None:
        self.app.confirm_kill(self._entry, self._status)


class RunBrowserApp(App):
    """Arrow-key browser over recent submissions."""

    TITLE = "baircondor history"
    BINDINGS = [
        Binding("q,escape", "quit", "quit"),
        Binding("r", "refresh_statuses", "refresh"),
        Binding("k", "kill", "kill job"),
    ]

    def __init__(self, entries: list[dict]) -> None:
        super().__init__()
        self._entries = entries
        self._statuses = {i: "?" for i in range(len(entries))}

    def compose(self) -> ComposeResult:
        table = DataTable(cursor_type="row")
        table.add_column("when", key="when")
        table.add_column("job", key="job")
        table.add_column("status", key="status")
        table.add_column("gpus", key="gpus")
        table.add_column("run dir", key="run_dir")
        yield table
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        for i, entry in enumerate(self._entries):
            table.add_row(
                Text(entry_timestamp(entry), style="dim"),
                Text(entry.get("jobname", "?"), style="bold"),
                status_text("?"),
                str(entry.get("gpus", 0)),
                Text(entry.get("run_dir", ""), style="dim cyan"),
                key=str(i),
            )
        table.focus()
        self.action_refresh_statuses()
        self.set_interval(10.0, self.action_refresh_statuses)

    def action_refresh_statuses(self) -> None:
        self._load_statuses()

    @work(thread=True, exclusive=True, group="list-status")
    def _load_statuses(self) -> None:
        worker = get_current_worker()
        for i, entry in enumerate(self._entries):
            if worker.is_cancelled:
                return
            status = get_job_status(entry.get("cluster_id"))
            self.call_from_thread(self._set_status, i, status)

    def _set_status(self, index: int, status: str) -> None:
        self._statuses[index] = status
        self.query_one(DataTable).update_cell(
            str(index), "status", status_text(status), update_width=True
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        index = int(event.row_key.value)
        self.push_screen(RunDetailScreen(self._entries[index], self._statuses[index]))

    def action_kill(self) -> None:
        table = self.query_one(DataTable)
        if not self._entries or table.cursor_row is None:
            return
        index = table.cursor_row
        self.confirm_kill(self._entries[index], self._statuses[index])

    def confirm_kill(self, entry: dict, status: str) -> None:
        """Shared kill flow (list and detail views): confirm modal, then condor_rm."""
        cluster_id = entry.get("cluster_id")
        if not cluster_id:
            self.notify("no cluster id recorded for this run", severity="warning")
            return
        if status not in _ACTIVE_STATUSES:
            self.notify(f"job is already {status}", severity="warning")
            return

        def _on_confirm(confirmed: bool | None) -> None:
            if confirmed:
                self._kill(str(cluster_id))

        self.push_screen(ConfirmKill(entry.get("jobname", "?"), str(cluster_id)), _on_confirm)

    @work(thread=True, group="kill")
    def _kill(self, cluster_id: str) -> None:
        try:
            result = subprocess.run(
                ["condor_rm", cluster_id], capture_output=True, text=True, timeout=10
            )
            ok = result.returncode == 0
            msg = (result.stdout or result.stderr).strip() or f"condor_rm {cluster_id}"
        except (subprocess.TimeoutExpired, OSError) as e:
            ok, msg = False, f"condor_rm failed: {e}"
        self.call_from_thread(self.notify, msg, severity="information" if ok else "error")
        if ok:
            self.call_from_thread(self.action_refresh_statuses)
