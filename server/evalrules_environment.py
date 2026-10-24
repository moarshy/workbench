# SPDX-License-Identifier: BSD-3-Clause

"""
Eval-rules desk (v9): Workbench task families served under the hidden test's rules.

Interface, field for field the organizers' PostTrain OpenEnv adapter for BenchFlow tasks:
  - actions {type: run_bash | submit | finalize, command, answer}
  - observations {output, task_id, rollout_dir, last_returncode}
  - command output cut to its first 8,192 characters, marked "[benchflow output truncated]"
  - the first observation is the instruction alone (no worked examples or warnings)
  - the workspace is /root in the container; inputs include real .xlsx and .pdf files
  - submit or finalize ends the episode (the arena's finish_action is finalize)

Reward:
  task_score = share of the task's checks passed, 0 if a protected check fails
               (output schema, inputs unchanged, regressions; as in Workbench v4)
  reward     = 0.85 * task_score
             + 0.15 * (ended with submit)       only if task_score >= 0.5
             + 0.02 * valid actions taken, counting at most 5      (survival: keep the format, v9 amendment 1)
             - 0.05 * repeated turns (same command and same output as an earlier turn), at most 0.2
             clipped to [0, 1]
"""

import hashlib
import os
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..evalrules_models import PostTrainAction, PostTrainObservation
    from . import formats
    from .sandbox import run_bash
    from .workbench_environment import (FAMILIES, INPUTS_CHECK, _give_to_agent, _inputs_unchanged, _snapshot, _wipe,
                                        _workspace_root)
except ImportError:
    from evalrules_models import PostTrainAction, PostTrainObservation
    from server import formats
    from server.sandbox import run_bash
    from server.workbench_environment import (FAMILIES, INPUTS_CHECK, _give_to_agent, _inputs_unchanged, _snapshot,
                                              _wipe, _workspace_root)

MAX_OUTPUT_CHARS = 8192
TRUNCATION_MARK = "\n[benchflow output truncated]"
MAX_TOOL_CALLS = 128            # the arena's tool_calls_total; a safety net here
TASK_WEIGHT, FINISH_BONUS, REPEAT_PENALTY, MAX_REPEAT_PENALTY = 0.85, 0.15, 0.05, 0.2
SURVIVAL_PER_TURN, SURVIVAL_TURNS = 0.02, 5

# task id -> which generated files become .xlsx / .pdf
TASKS = {
    # v9.1: lite levels for finance/office (flat at ~0.1 in v9 training); dropped loans-hard, expenses-easy and
    # pyrepo-implement (no reward spread in v9 training)
    "loans-lite": dict(xlsx=["data/loans.csv", "data/payments.csv"], pdf=["policy/credit_policy.md"]),
    "loans-easy": dict(xlsx=["data/loans.csv", "data/payments.csv"], pdf=["policy/credit_policy.md"]),
    "loans-medium": dict(xlsx=["data/loans.csv", "data/payments.csv"], pdf=["policy/credit_policy.md"]),
    "expenses-lite": dict(xlsx=["data/claims.csv"], pdf=[]),
    "expenses-medium": dict(xlsx=["data/claims.csv", "data/employees.csv"], pdf=["policy/travel_policy.md"]),
    "downtime-medium": dict(xlsx=[], pdf=["handbook/maintenance_manual.md"]),
    "reflow-hard": dict(xlsx=["data/runs.csv"], pdf=["docs/handbook.md"]),
    "pyrepo-bug1": dict(xlsx=[], pdf=[]),
    "repair-medium": dict(xlsx=[], pdf=[]),
}
DEFAULT_TASK = "loans-medium"


def head_only(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + TRUNCATION_MARK


def reward_for(checks, protected, submitted: bool, repeats: int, turns: int = 0) -> float:
    ok = dict(checks)
    task_score = 0.0 if any(not ok.get(n, True) for n in protected) else sum(v for _, v in checks) / len(checks)
    r = TASK_WEIGHT * task_score + (FINISH_BONUS if submitted and task_score >= 0.5 else 0.0)
    r += SURVIVAL_PER_TURN * min(turns, SURVIVAL_TURNS)
    r -= min(MAX_REPEAT_PENALTY, REPEAT_PENALTY * repeats)
    return round(min(1.0, max(0.0, r)), 4)


class EvalRulesEnvironment(Environment):
    """Workbench tasks under the hidden test's interface, budgets and file formats."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._root = _workspace_root()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = None
        self._task_id = None
        self._inputs = {}
        self._seen = set()
        self._repeats = 0
        self._turns = 0
        self._done = True

    def _obs(self, output: str, rc=None, reward=0.0, done=False, metadata=None) -> PostTrainObservation:
        return PostTrainObservation(output=output, task_id=self._task_id, rollout_dir=None, last_returncode=rc,
                                    reward=reward, done=done, metadata=metadata or {})

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> PostTrainObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task_id = task_id if task_id in TASKS else DEFAULT_TASK
        family, level = self._task_id.rsplit("-", 1)
        _wipe(self._root)
        self._task = FAMILIES[family].generate(random.Random(seed), level, self._root)
        spec = TASKS[self._task_id]
        self._task.instruction = formats.convert(self._root, self._task.instruction, spec["xlsx"], spec["pdf"])
        (self._root / "TASK.md").write_text(self._task.instruction)
        self._inputs = _snapshot(self._root, self._task.editable)
        _give_to_agent(self._root)
        self._seen, self._repeats, self._turns, self._done = set(), 0, 0, False
        return self._obs(self._task.instruction)

    def _grade(self, submitted: bool, prefix: str) -> PostTrainObservation:
        checks = self._task.verify(self._root)
        protected = self._task.protected
        if self._inputs:
            checks = checks + [(INPUTS_CHECK, _inputs_unchanged(self._root, self._inputs))]
            protected = protected + (INPUTS_CHECK,)
        reward = reward_for(checks, protected, submitted, self._repeats, self._turns)
        self._done = True
        passed = sum(ok for _, ok in checks)
        return self._obs(f"{prefix} {passed}/{len(checks)} checks passed. reward={reward}", reward=reward, done=True,
                         metadata={"checks": checks, "submitted": submitted, "repeats": self._repeats,
                                   "turns": self._turns})

    def step(self, action: PostTrainAction, **kwargs) -> PostTrainObservation:  # type: ignore[override]
        if self._done or self._task is None:
            return self._obs("The task is finished. Call reset to start a new one.", reward=0.0, done=True)
        self._state.step_count += 1
        if action.type == "submit":
            (self._root / "answer.txt").write_text(action.answer or "")
            return self._grade(True, "Submitted.")
        if action.type == "finalize":
            return self._grade(False, "Finalized.")
        if not action.command:
            return self._obs("run_bash requires a command", rc=2)
        self._turns += 1
        r = run_bash(action.command, str(self._root), max_chars=10 ** 7)
        output = head_only(r.output)
        key = hashlib.sha256((" ".join(action.command.split()) + "\0" + r.output).encode()).hexdigest()
        if key in self._seen:
            self._repeats += 1
        self._seen.add(key)
        if self._state.step_count >= MAX_TOOL_CALLS:
            return self._grade(False, output + "\n\nOut of tool calls; grading the workspace.")
        return self._obs(output, rc=r.exit_code)

    @property
    def state(self) -> State:
        return self._state
