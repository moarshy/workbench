# SPDX-License-Identifier: BSD-3-Clause

"""
Data models for the Workbench environment.

The action shape mirrors the run_bash / submit interface of the PostTrain Arena
OpenEnv adapter, so the habits learned here use the same interface the model is
likely to meet elsewhere.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class WorkbenchAction(Action):
    """One step: run a shell command in the workspace, or submit."""

    type: Literal["run_bash", "submit"] = Field(
        ..., description='"run_bash" runs a shell command in the task workspace; "submit" ends the task and grades the output files'
    )
    command: str = Field(default="", description="For run_bash: the shell command to run")
    answer: str = Field(default="", description="For submit: a short final note (the output files are what gets graded)")


class WorkbenchObservation(Observation):
    """The task instruction on reset, then the output of each command."""

    output: str = Field(default="", description="Task instruction, or the command's combined stdout/stderr (truncated if long)")
    exit_code: int = Field(default=0, description="Exit code of the last command")
    steps_left: int = Field(default=0, description="Actions remaining before the task is graded automatically")
