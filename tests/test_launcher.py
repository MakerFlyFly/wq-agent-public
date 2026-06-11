from __future__ import annotations

import os
import runpy
import sys

import pytest

from wq_agent import launcher
from wq_agent.workspace import WORKSPACE_ENV_VAR, resolve_workspace


class InputQueue:
    def __init__(self, values: list[str]):
        self.values = values

    def __call__(self, prompt: str) -> str:
        assert prompt
        if not self.values:
            raise EOFError
        return self.values.pop(0)


def test_launcher_no_args_menu_can_exit():
    commands: list[list[str]] = []
    output: list[str] = []

    launcher.main(
        [],
        input_func=InputQueue(["0"]),
        print_func=lambda *parts: output.append(" ".join(str(p) for p in parts)),
        command_runner=commands.append,
    )

    assert commands == []
    assert any("wq-agent" in line for line in output)


def test_launcher_with_args_delegates_to_cli():
    commands: list[list[str]] = []

    launcher.main(
        ["wiki", "stats", "--verbose"],
        input_func=InputQueue([]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [["wiki", "stats", "--verbose"]]


def test_launcher_script_entry_imports_when_run_as_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--help"])
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "wq_agent", "launcher.py")

    with pytest.raises(SystemExit):
        runpy.run_path(script_path, run_name="__main__")


def test_launcher_run_menu_builds_command_with_defaults_and_idea():
    commands: list[list[str]] = []

    launcher.run_menu(
        input_func=InputQueue(["2", "", "", "analyst revision", "0"]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [
        ["run", "--count", "18", "--batches", "1", "--idea", "analyst revision"]
    ]


def test_launcher_command_mode_accepts_full_wq_agent_prefix(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(os, "name", "posix")

    launcher.run_menu(
        input_func=InputQueue(["5", 'wq-agent.exe generate --idea "low turnover" -n 3', "0"]),
        print_func=lambda *parts: None,
        command_runner=commands.append,
    )

    assert commands == [["generate", "--idea", "low turnover", "-n", "3"]]


def test_launcher_configures_cwd_to_workspace_when_frozen_in_dist(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / ".env").write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
    (workspace / "private_wiki").mkdir()
    exe_dir = workspace / "dist" / "wq-agent"
    exe_dir.mkdir(parents=True)
    exe_path = exe_dir / "wq-agent.exe"
    exe_path.write_text("", encoding="utf-8")
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))

    old_cwd = os.getcwd()
    try:
        root = launcher.configure_runtime_cwd()
        assert root == workspace.resolve()
        assert os.getcwd() == str(workspace.resolve())
        assert os.environ[WORKSPACE_ENV_VAR] == str(workspace.resolve())
    finally:
        os.chdir(old_cwd)


def test_workspace_env_var_overrides_frozen_dist_location(tmp_path, monkeypatch):
    configured = tmp_path / "configured-workspace"
    configured.mkdir()
    exe_dir = tmp_path / "dist" / "wq-agent"
    exe_dir.mkdir(parents=True)
    exe_path = exe_dir / "wq-agent.exe"
    exe_path.write_text("", encoding="utf-8")

    assert resolve_workspace(
        executable=exe_path,
        frozen=True,
        environ={WORKSPACE_ENV_VAR: str(configured)},
    ) == configured.resolve()


def test_frozen_workspace_uses_executable_before_dist_cwd(tmp_path):
    exe_workspace = tmp_path / "exe-project"
    exe_dir = exe_workspace / "dist" / "wq-agent"
    exe_dir.mkdir(parents=True)
    exe_path = exe_dir / "wq-agent.exe"
    exe_path.write_text("", encoding="utf-8")
    unrelated_workspace = tmp_path / "unrelated"
    unrelated_cwd = unrelated_workspace / "dist" / "wq-agent"
    unrelated_cwd.mkdir(parents=True)

    assert (
        resolve_workspace(
            cwd=unrelated_cwd,
            executable=exe_path,
            frozen=True,
            environ={},
        )
        == exe_workspace.resolve()
    )


def test_workspace_resolves_dist_cwd_to_project_root(tmp_path):
    workspace = tmp_path / "project"
    dist_cwd = workspace / "dist" / "wq-agent"
    dist_cwd.mkdir(parents=True)

    assert resolve_workspace(cwd=dist_cwd, frozen=False, environ={}) == workspace.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows command parsing uses CommandLineToArgvW")
def test_launcher_windows_command_line_parser_handles_quotes():
    assert launcher.split_command_line('generate --idea "low turnover" -n 3') == [
        "generate",
        "--idea",
        "low turnover",
        "-n",
        "3",
    ]
