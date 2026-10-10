# SPDX-License-Identifier: BSD-3-Clause

"""Workbench Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import WorkbenchAction, WorkbenchObservation


class WorkbenchEnv(
    EnvClient[WorkbenchAction, WorkbenchObservation, State]
):
    """
    Client for the Workbench Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with WorkbenchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(WorkbenchAction(type="run_bash", command="ls"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect (.sync() for sync use)
        >>> client = WorkbenchEnv.from_docker_image("workbench-env:latest").sync()
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(WorkbenchAction(type="submit"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: WorkbenchAction) -> Dict:
        """
        Convert WorkbenchAction to JSON payload for step message.

        Args:
            action: WorkbenchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "type": action.type,
            "command": action.command,
            "answer": action.answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WorkbenchObservation]:
        """
        Parse server response into StepResult[WorkbenchObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with WorkbenchObservation
        """
        obs_data = payload.get("observation", {})
        observation = WorkbenchObservation(
            output=obs_data.get("output", ""),
            exit_code=obs_data.get("exit_code", 0),
            steps_left=obs_data.get("steps_left", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
