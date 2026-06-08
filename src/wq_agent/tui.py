from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Any, Awaitable, Callable

from rich.console import Console
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static

from .agent.orchestrator import Orchestrator
from .config import get_settings
from .db import Database
from .models import AlphaStatus, GenerationStrategy


_STRATEGIES = [
    ("llm", GenerationStrategy.LLM.value),
    ("template", GenerationStrategy.TEMPLATE.value),
    ("factor_mining", GenerationStrategy.FACTOR_MINING.value),
]


class _TuiConsole:
    def __init__(self, write: Callable[[str], None], width: int = 110):
        self.write = write
        self.width = width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        buf = io.StringIO()
        console = Console(file=buf, width=self.width, color_system=None, force_terminal=False)
        console.print(*objects, **kwargs)
        text = buf.getvalue().rstrip()
        if text:
            self.write(text)


class WQAgentTui(App[None]):
    """Keyboard-first terminal workbench for common wq-agent flows."""

    CSS = """
    Screen {
        background: #10110f;
        color: #d7ded3;
    }

    Header {
        background: #18211b;
        color: #e7f0df;
    }

    Footer {
        background: #111713;
        color: #aebaa8;
    }

    #workspace {
        height: 1fr;
    }

    #sidebar {
        width: 36;
        min-width: 32;
        padding: 1 1;
        border-right: tall #3d473c;
        background: #151915;
    }

    #main {
        width: 1fr;
        padding: 1 1 0 1;
        background: #10110f;
    }

    #brand {
        height: 4;
        padding: 1;
        margin-bottom: 1;
        border: tall #66825d;
        background: #1d251d;
        color: #f1f7e9;
        text-style: bold;
    }

    #context {
        height: 3;
        padding: 0 1;
        margin-bottom: 1;
        color: #aebaa8;
        border: tall #343b33;
        background: #121713;
    }

    .section-title {
        height: 1;
        margin-top: 1;
        color: #c6d9b8;
        text-style: bold;
    }

    .field-label {
        height: 1;
        color: #9bb38e;
        margin-top: 1;
    }

    Select, Input {
        height: 3;
        margin-bottom: 0;
        border: tall #374033;
        background: #0f130f;
    }

    Input:focus, Select:focus {
        border: tall #b2c98a;
    }

    Button {
        width: 100%;
        height: 3;
        margin-top: 1;
        border: tall #3f4a3c;
    }

    #run {
        margin-top: 2;
    }

    #topbar {
        height: 4;
        margin-bottom: 1;
    }

    #run-state {
        width: 16;
        height: 4;
        padding: 1;
        border: tall #6a7e55;
        background: #1b2419;
        color: #dceccd;
        text-align: center;
        text-style: bold;
    }

    #paths {
        width: 1fr;
        height: 4;
        padding: 1 2;
        margin-left: 1;
        border: tall #343b33;
        background: #141814;
        color: #aebaa8;
    }

    #metrics {
        height: 5;
        margin-bottom: 1;
    }

    .metric {
        width: 1fr;
        height: 5;
        padding: 1 2;
        margin-right: 1;
        border: tall #3a4538;
        background: #151b15;
        color: #d8e5cf;
    }

    #metric-submitted {
        margin-right: 0;
    }

    #log-title, #alphas-title {
        height: 1;
        color: #c6d9b8;
        text-style: bold;
    }

    #log {
        height: 1fr;
        min-height: 12;
        border: tall #3d473c;
        background: #0d100d;
        padding: 0 1;
    }

    #alphas-title {
        margin-top: 1;
    }

    #alphas {
        height: 12;
        border: tall #3d473c;
        background: #0d100d;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "refresh", "Refresh"),
        ("g", "generate", "Generate"),
        ("r", "run_batch", "Run"),
        ("f", "refine", "Refine"),
        ("b", "backtest_pending", "Backtest"),
        ("l", "refresh", "List"),
    ]

    TITLE = "wq-agent"
    SUB_TITLE = "alpha research workbench"

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        settings = get_settings()
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Static("wq-agent\nalpha workbench", id="brand")
                yield Static(
                    f"{settings.WQ_REGION} / {settings.WQ_UNIVERSE}\n"
                    f"delay {settings.WQ_DELAY} | {settings.WQ_NEUTRALIZATION.lower()}",
                    id="context",
                )
                yield Static("Session", classes="section-title")
                yield Static("Strategy", classes="field-label")
                yield Select(_STRATEGIES, value=GenerationStrategy.LLM.value, id="strategy")
                yield Static("Count", classes="field-label")
                yield Input(value="18", placeholder="18", id="count")
                yield Static("Batches", classes="field-label")
                yield Input(value="1", placeholder="1", id="batches")
                yield Static("Idea", classes="field-label")
                yield Input(placeholder="alpha thesis", id="idea")
                yield Static("Actions", classes="section-title")
                yield Button("Run full pipeline", id="run", variant="success")
                yield Button("Generate only", id="generate", variant="primary")
                yield Button("Refine near-miss", id="refine")
                yield Button("Backtest pending", id="backtest")
                yield Button("Refresh dashboard", id="refresh")
            with Vertical(id="main"):
                with Horizontal(id="topbar"):
                    yield Static("IDLE", id="run-state")
                    yield Static("Loading workspace paths...", id="paths")
                with Horizontal(id="metrics"):
                    yield Static("Generated\n--", id="metric-generated", classes="metric")
                    yield Static("Backtesting\n--", id="metric-backtesting", classes="metric")
                    yield Static("High\n--", id="metric-high", classes="metric")
                    yield Static("Submitted\n--", id="metric-submitted", classes="metric")
                yield Static("Task Log", id="log-title")
                yield RichLog(id="log", wrap=True, highlight=True, markup=True)
                yield Static("Recent Alphas", id="alphas-title")
                yield DataTable(id="alphas", zebra_stripes=True)
        yield Footer()

    async def on_mount(self) -> None:
        self.write_log("[bold cyan]wq-agent TUI ready[/bold cyan]")
        await self.refresh_dashboard()

    def write_log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    async def action_refresh(self) -> None:
        await self.refresh_dashboard()

    async def action_generate(self) -> None:
        await self._run_task("generate", self._generate_only)

    async def action_run_batch(self) -> None:
        await self._run_task("run", self._run_pipeline)

    async def action_refine(self) -> None:
        await self._run_task("refine", self._refine)

    async def action_backtest_pending(self) -> None:
        await self._run_task("backtest pending", self._backtest_pending)

    @on(Button.Pressed, "#refresh")
    async def _button_refresh(self) -> None:
        await self.action_refresh()

    @on(Button.Pressed, "#generate")
    async def _button_generate(self) -> None:
        await self.action_generate()

    @on(Button.Pressed, "#run")
    async def _button_run(self) -> None:
        await self.action_run_batch()

    @on(Button.Pressed, "#refine")
    async def _button_refine(self) -> None:
        await self.action_refine()

    @on(Button.Pressed, "#backtest")
    async def _button_backtest(self) -> None:
        await self.action_backtest_pending()

    async def refresh_dashboard(self) -> None:
        settings = get_settings()
        db = Database(settings.DB_PATH)
        await db.connect()
        try:
            stats = await db.get_stats()
            recent = await db.list_alphas(limit=12)
            high = await db.list_high_quality_alphas(min_fitness=0.0)
        finally:
            await db.close()

        high_count = len([r for r in high if (r.get("fitness") or 0) >= settings.MIN_FITNESS])
        self.query_one("#paths", Static).update(
            f"db [bold]{settings.DB_PATH}[/bold]\n"
            f"wiki [bold]{settings.WIKI_DIR}[/bold] + [bold]{settings.WIKI_AUTO_RECORD_DIR}[/bold]"
        )
        self.query_one("#metric-generated", Static).update(
            f"Generated\n[bold cyan]{stats.get('generated', 0)}[/bold cyan]"
        )
        self.query_one("#metric-backtesting", Static).update(
            f"Backtesting\n[bold yellow]{stats.get('backtesting', 0)}[/bold yellow]"
        )
        self.query_one("#metric-high", Static).update(f"High\n[bold green]{high_count}[/bold green]")
        self.query_one("#metric-submitted", Static).update(
            f"Submitted\n[bold magenta]{stats.get('submitted', 0)}[/bold magenta]"
        )

        table = self.query_one("#alphas", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Status", "Strategy", "Created", "Expression")
        for alpha in recent:
            table.add_row(
                str(alpha.id or ""),
                alpha.status.value,
                alpha.strategy.value,
                alpha.created_at.strftime("%m-%d %H:%M"),
                _truncate(alpha.expression, 84),
            )

    async def _run_task(self, label: str, job: Callable[[], Awaitable[None]]) -> None:
        if self._busy:
            self.write_log("[yellow]A task is already running.[/yellow]")
            return
        self._busy = True
        succeeded = False
        self._set_run_state("RUNNING", "#f2c36b")
        self.write_log(f"\n[bold cyan]> {label}[/bold cyan]")
        try:
            await job()
            succeeded = True
            self.write_log(f"[bold green]OK {label} finished[/bold green]")
            self._set_run_state("DONE", "#92c47d")
        except Exception as exc:
            self.write_log(f"[bold red]ERR {label} failed:[/bold red] {exc}")
            self._set_run_state("ERROR", "#d77969")
        finally:
            self._busy = False
            await self.refresh_dashboard()
            if succeeded:
                self.set_timer(1.5, lambda: self._set_run_state("IDLE", "#dceccd"))

    async def _generate_only(self) -> None:
        count = self._positive_int("#count", default=18)
        strategy = self._strategy()
        idea = self._idea()
        await self._with_orchestrator(
            lambda orch: orch.run(
                strategy=strategy,
                count=count,
                auto_backtest=False,
                user_idea=idea,
            )
        )

    async def _run_pipeline(self) -> None:
        count = self._positive_int("#count", default=18)
        batches = self._positive_int("#batches", default=1)
        strategy = self._strategy()
        idea = self._idea()

        async def _job(orch: Orchestrator) -> None:
            for i in range(1, batches + 1):
                self.write_log(f"[bold magenta]Batch {i}/{batches}[/bold magenta]")
                await orch.run(strategy=strategy, count=count, auto_backtest=True, user_idea=idea)

        await self._with_orchestrator(_job)

    async def _refine(self) -> None:
        count = self._positive_int("#count", default=10)
        await self._with_orchestrator(lambda orch: orch.refine(count=count, auto_backtest=True))

    async def _backtest_pending(self) -> None:
        async def _job(orch: Orchestrator) -> None:
            pending = await orch.db.list_alphas(status=AlphaStatus.GENERATED, limit=1000)
            ids = [a.id for a in pending if a.id]
            if not ids:
                self.write_log("[yellow]No pending generated alphas.[/yellow]")
                return
            self.write_log(f"Backtesting {len(ids)} pending alphas")
            await orch.backtest(ids)

        await self._with_orchestrator(_job)

    async def _with_orchestrator(self, job: Callable[[Orchestrator], Awaitable[Any]]) -> None:
        orch = Orchestrator()
        with self._capture_orchestrator_console():
            try:
                await orch.initialize()
                await job(orch)
            finally:
                await orch.close()

    @contextmanager
    def _capture_orchestrator_console(self):
        from .agent import orchestrator as orchestrator_module

        old_console = orchestrator_module.console
        orchestrator_module.console = _TuiConsole(self.write_log)
        try:
            yield
        finally:
            orchestrator_module.console = old_console

    def _set_run_state(self, label: str, color: str) -> None:
        state = self.query_one("#run-state", Static)
        state.update(label)
        state.styles.color = color

    def _strategy(self) -> GenerationStrategy:
        value = self.query_one("#strategy", Select).value
        return GenerationStrategy(str(value or GenerationStrategy.LLM.value))

    def _idea(self) -> str | None:
        value = self.query_one("#idea", Input).value.strip()
        return value or None

    def _positive_int(self, selector: str, default: int) -> int:
        raw = self.query_one(selector, Input).value.strip()
        try:
            value = int(raw)
        except ValueError:
            value = default
        return max(1, value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def run_tui() -> None:
    WQAgentTui().run()
