# SPDX-License-Identifier: BSD-3-Clause

"""
Workbench environment.

reset(task_id) builds a fresh workspace for the task's family and level, step()
runs shell commands in it, and submit (or running out of steps) grades the
output files. Reward (v4):
  0                                         if any protected check fails (broken output
                                            schema, regressions, or an input file changed)
  0.9 + 0.1 * steps_left / MAX_STEPS        if every check passes (in [0.9, 1])
  0.5 * share of checks passed              otherwise (below 0.5)
"""

import fnmatch
import hashlib
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WorkbenchAction, WorkbenchObservation
    from .families import downtime, expenses, loans, plant, pyrepo, reflow, repair, routing
    from .sandbox import run_bash
except ImportError:
    from models import WorkbenchAction, WorkbenchObservation
    from server.families import downtime, expenses, loans, plant, pyrepo, reflow, repair, routing
    from server.sandbox import run_bash

FAMILIES = {"downtime": downtime, "expenses": expenses, "routing": routing, "repair": repair,
            "pyrepo": pyrepo, "reflow": reflow, "plant": plant, "loans": loans}
TASKS = [f"{fam}-{lvl}" for fam, mod in FAMILIES.items() for lvl in mod.LEVELS]
DEFAULT_TASK = "downtime-medium"
MAX_STEPS = 12
EFFICIENCY_BONUS = 0.1
INPUTS_CHECK = "input files left unchanged"
MARKER = ".workbench_root"
_NOT_INPUTS = ("output/*", "*/.git/*", "*/__pycache__/*", "*/.pytest_cache/*", MARKER)
HINT = """

Reply with exactly one JSON object per turn, for example:
{"type": "run_bash", "command": "ls -R /app | head -50"}
{"type": "run_bash", "command": "grep -rn -i 'limit' /app/docs"}
{"type": "submit", "answer": "done"}
Escape double quotes inside the command as \\" (or use single quotes in the shell). You have 12 actions; command output is truncated to about 3000 characters, so search and print selectively. Leave the files you were given unchanged (except code the task asks you to fix): write your results where the task says. Call submit when your work is done."""


def _workspace_root() -> Path:
    """$WORKBENCH_ROOT (the container sets /app), else a private temp dir."""
    root = Path(os.environ.get("WORKBENCH_ROOT") or tempfile.mkdtemp(prefix="workbench-"))
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).touch()
    return root


def _snapshot(root: Path, editable) -> dict:
    """{relative path: sha256} of the files the agent must not change."""
    out = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and not any(
                fnmatch.fnmatch(rel, pat) for pat in (*_NOT_INPUTS, *editable)):
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _inputs_unchanged(root: Path, snapshot: dict) -> bool:
    for rel, digest in snapshot.items():
        path = root / rel
        try:
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
        except OSError:
            return False
    return True


def reward_for(checks, protected, steps_left: int) -> float:
    ok = dict(checks)
    if any(not ok.get(name, True) for name in protected):
        return 0.0
    passed = sum(v for _, v in checks)
    if passed == len(checks):
        return 1.0 - EFFICIENCY_BONUS + EFFICIENCY_BONUS * steps_left / MAX_STEPS
    return 0.5 * passed / len(checks)


def _wipe(root: Path) -> None:
    """Empty the workspace, but only a directory we marked as ours."""
    if not (root / MARKER).exists() or root in (Path("/"), Path.home()):
        raise RuntimeError(f"refusing to wipe {root}")
    for child in root.iterdir():
        if child.name == MARKER:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _give_to_agent(root: Path) -> None:
    """In the container, the agent user owns the workspace (it writes scripts and output there)."""
    if os.geteuid() != 0:
        return
    import pwd
    try:
        pw = pwd.getpwnam("agent")
    except KeyError:
        return
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [dirpath, *(os.path.join(dirpath, n) for n in dirnames + filenames)]:
            os.chown(name, pw.pw_uid, pw.pw_gid)


class WorkbenchEnvironment(Environment):
    """Workspace tasks solved with run_bash, graded on the files the agent writes."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # one shared workspace directory per server

    def __init__(self):
        self._root = _workspace_root()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None
        self._inputs = {}
        self._done = True

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> WorkbenchObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        name = task_id if task_id in TASKS else DEFAULT_TASK
        family, level = name.rsplit("-", 1)
        _wipe(self._root)
        self._task = FAMILIES[family].generate(random.Random(seed), level, self._root)
        self._inputs = _snapshot(self._root, self._task.editable)
        _give_to_agent(self._root)
        self._done = False
        return WorkbenchObservation(output=self._task.instruction + HINT, exit_code=0,
                                    steps_left=MAX_STEPS, done=False, reward=0.0)

    def step(self, action: WorkbenchAction) -> WorkbenchObservation:  # type: ignore[override]
        if self._done or self._task is None:
            return WorkbenchObservation(output="The task is finished. Call reset to start a new one.",
                                        done=True, reward=0.0)
        self._state.step_count += 1
        steps_left = MAX_STEPS - self._state.step_count

        if action.type == "submit":
            return self._grade(steps_left, "Submitted.")

        result = run_bash(action.command, str(self._root))
        if steps_left <= 0:
            return self._grade(0, result.output + "\n\nOut of steps; grading the output files now.")
        output = result.output
        if steps_left == 2:
            output += "\n\n[2 actions left: write your final output now and submit.]"
        return WorkbenchObservation(output=output, exit_code=result.exit_code,
                                    steps_left=steps_left, done=False, reward=0.0)

    def _grade(self, steps_left: int, prefix: str) -> WorkbenchObservation:
        checks = self._task.verify(self._root)
        protected = self._task.protected
        if self._inputs:
            checks = checks + [(INPUTS_CHECK, _inputs_unchanged(self._root, self._inputs))]
            protected = protected + (INPUTS_CHECK,)
        passed = sum(ok for _, ok in checks)
        reward = reward_for(checks, protected, steps_left)
        self._done = True
        return WorkbenchObservation(
            output=f"{prefix} {passed}/{len(checks)} checks passed.", exit_code=0, steps_left=steps_left,
            done=True, reward=round(reward, 4), metadata={"checks": checks, "passed": passed, "total": len(checks)})

    @property
    def state(self) -> State:
        return self._state
