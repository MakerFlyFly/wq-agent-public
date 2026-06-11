from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


WORKSPACE_ENV_VAR = "WQ_AGENT_WORKSPACE"


def resolve_workspace(
    *,
    cwd: Path | None = None,
    executable: Path | None = None,
    frozen: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    configured = (env.get(WORKSPACE_ENV_VAR) or "").strip().strip('"')
    if configured:
        return Path(configured).expanduser().resolve()

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if is_frozen:
        executable_path = Path(sys.executable if executable is None else executable).resolve()
        executable_dir = executable_path.parent
        workspace = _workspace_from_dist_dir(executable_dir)
        return workspace if workspace is not None else executable_dir

    cwd_path = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
    cwd_workspace = _workspace_from_dist_dir(cwd_path)
    if cwd_workspace is not None:
        return cwd_workspace

    return cwd_path


def configure_process_workspace() -> Path:
    workspace = resolve_workspace()
    os.environ[WORKSPACE_ENV_VAR] = str(workspace)
    os.chdir(workspace)
    return workspace


def _workspace_from_dist_dir(executable_dir: Path) -> Path | None:
    if executable_dir.name.lower() != "wq-agent":
        return None
    dist_dir = executable_dir.parent
    if dist_dir.name.lower() != "dist":
        return None
    return dist_dir.parent.resolve()
