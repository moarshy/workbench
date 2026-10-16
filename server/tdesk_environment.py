# SPDX-License-Identifier: BSD-3-Clause

"""
T-Desk: the Workbench task families served through the tbench2_env interface.

Mirrors OpenEnv's tbench2_env as used by the arena operators' smoke tests:
  - actions {action_type, command, session_id, block, wait_seconds, file_path, content}
      exec        run a shell command in the workspace (output is NOT truncated)
      write_file  write `content` to `file_path`
      evaluate    run the hidden checks: reward 1.0 if all pass else 0.0, episode ends
      close       ends the episode (graded like evaluate, so every episode gets a reward)
      write/view/wait/kill  interactive sessions: not supported here (error, episode continues)
  - every observation repeats the full instruction
  - the arena's finish_action is {"action_type": "evaluate"}

Only the interface and budget differ from Workbench; the tasks are the same families.
"""

import os
import random
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..tdesk_models import Tbench2Action, Tbench2Observation
    from .sandbox import run_bash
    from .workbench_environment import FAMILIES, _give_to_agent, _wipe, _workspace_root
except ImportError:
    from tdesk_models import Tbench2Action, Tbench2Observation
    from server.sandbox import run_bash
    from server.workbench_environment import FAMILIES, _give_to_agent, _wipe, _workspace_root

# Budget-fit task set: each must be solvable in ~4-5 economical actions under 4,096 / 8,192 tokens.
# plant is excluded: in calibration the model decoded its sealed parameters instead of experimenting.
TASKSETS = {
    # v6: budget-fit tasks for the tight smoke budgets (4,096 / 8,192)
    "fit": ["pyrepo-located", "pyrepo-located2", "pyrepo-bug1", "reflow-lite", "downtime-lite", "expenses-lite",
            "routing-medium", "routing-hard"],
    # v7: the Workbench v3 (v5) task set minus plant, for the roomy budgets (12,288 / 16,384)
    "roomy": ["pyrepo-bug1", "pyrepo-bug2", "pyrepo-implement", "repair-medium", "repair-hard", "reflow-hard",
              "downtime-medium", "expenses-easy"],
}
TASKSET = os.environ.get("TDESK_TASKSET", "fit")
TASKS = TASKSETS[TASKSET]
DEFAULT_TASK = TASKS[0]
MAX_ACTIONS = 32          # the arena's tool_calls_total for the smoke tasks
MAX_OUTPUT = 200_000      # memory safety only; the token budget is the real limit, as in tbench2_env
SESSION = "tb2-session"


def _instruction(text: str) -> str:
    """Workbench instructions, minus Workbench-specific endings (tbench2 has no submit action)."""
    return text.replace("Call submit when you are done.\n", "").rstrip() + "\n"


class TDeskEnvironment(Environment):
    """Workbench tasks behind a tbench2_env-identical interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._root = _workspace_root()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None
        self._task_id = ""
        self._done = True

    def _obs(self, action_type, output="", success=True, error="", reward=None, done=False, info=None):
        return Tbench2Observation(instruction=self._task.instruction if self._task else "", output=output,
                                  success=success, error=error, task_id=self._task_id, task_path=str(self._root),
                                  session_id=None if action_type == "reset" else SESSION, action_type=action_type,
                                  info=info or {}, reward=reward, done=done)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> Tbench2Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        known = {t for ts in TASKSETS.values() for t in ts}
        self._task_id = task_id if task_id in known else DEFAULT_TASK
        family, level = self._task_id.rsplit("-", 1)
        _wipe(self._root)
        self._task = FAMILIES[family].generate(random.Random(seed), level, self._root)
        self._task.instruction = _instruction(self._task.instruction)
        (self._root / "TASK.md").write_text(self._task.instruction)
        _give_to_agent(self._root)
        self._done = False
        return self._obs("reset", reward=0.0)

    def _evaluate(self, action_type: str) -> Tbench2Observation:
        checks = self._task.verify(self._root)
        passed = sum(ok for _, ok in checks)
        reward = 1.0 if passed == len(checks) else 0.0
        lines = [f"{'PASSED' if ok else 'FAILED'} {name}" for name, ok in checks]
        self._done = True
        return self._obs(action_type, output="\n".join(lines) + f"\n{passed}/{len(checks)} checks passed",
                         reward=reward, done=True, info={"tests_passed": reward == 1.0, "passed": passed,
                                                         "total": len(checks)})

    def step(self, action: Tbench2Action) -> Tbench2Observation:  # type: ignore[override]
        if self._done or self._task is None:
            return self._obs(action.action_type, error="The task is finished. Call reset to start a new one.",
                             success=False, reward=0.0, done=True)
        self._state.step_count += 1
        kind = action.action_type

        if kind in ("evaluate", "close"):
            return self._evaluate(kind)
        if self._state.step_count >= MAX_ACTIONS:  # safety net; the arena enforces its own tool-call budget
            return self._evaluate("evaluate")

        if kind == "exec":
            r = run_bash(action.command, str(self._root), max_chars=MAX_OUTPUT)
            return self._obs(kind, output=r.output, success=r.exit_code == 0,
                             error="" if r.exit_code == 0 else f"exit code {r.exit_code}")
        if kind == "write_file":
            path = Path(action.file_path)
            path = (self._root / path) if not path.is_absolute() else path
            try:
                path = path.resolve()
                if self._root.resolve() not in path.parents:
                    raise ValueError(f"file_path must be inside {self._root}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(action.content)
                _give_to_agent(self._root)
                return self._obs(kind, output=f"Wrote content to {action.file_path}")
            except Exception as e:
                return self._obs(kind, success=False, error=str(e))
        if kind in ("write", "view", "wait", "kill"):
            return self._obs(kind, success=False,
                             error=f"{kind}: interactive sessions are not available here; use exec with a complete command")
        return self._obs(kind, success=False, error=f"Unsupported action_type: {kind}")

    @property
    def state(self) -> State:
        return self._state
