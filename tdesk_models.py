# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the TB2 environment.

Copied field-for-field from OpenEnv's tbench2_env (as served by the arena
operators' smoke-test Spaces) so that the JSON schema the policy is trained on
is identical to the one it is most likely evaluated with.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class Tbench2Action(Action):
    """Action for interacting with a TB2 task session."""

    action_type: str = Field(default="exec")
    command: str = Field(default="")
    session_id: str | None = Field(default=None)
    block: bool = Field(default=True)
    wait_seconds: float | None = Field(default=None)
    file_path: str = Field(default="")
    content: str = Field(default="")


class Tbench2Observation(Observation):
    """Observation returned from the TB2 environment."""

    instruction: str = Field(default="")
    output: str = Field(default="")
    success: bool = Field(default=True)
    error: str = Field(default="")
    task_id: str = Field(default="")
    task_path: str = Field(default="")
    session_id: str | None = Field(default=None)
    action_type: str = Field(default="")
    info: dict = Field(default_factory=dict)
