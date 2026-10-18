# SPDX-License-Identifier: BSD-3-Clause

"""
Eval-rules desk wire models: field for field the organizers' PostTrain OpenEnv protocol
(benchflow-ai/posttrainarena, benchflow_pipeline/openenv/models.py), which serves
BenchFlow (SkillsBench-format) tasks over OpenEnv.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation


class PostTrainAction(Action):
    type: Literal["run_bash", "submit", "finalize"]
    command: Optional[str] = None
    answer: Optional[str] = None


class PostTrainObservation(Observation):
    output: str = ""
    task_id: Optional[str] = None
    rollout_dir: Optional[str] = None
    last_returncode: Optional[int] = None
