"""Client for T-Desk (tbench2_env-shaped)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .tdesk_models import Tbench2Action, Tbench2Observation


class TDeskEnv(EnvClient[Tbench2Action, Tbench2Observation, State]):
    def _step_payload(self, action: Tbench2Action) -> Dict:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[Tbench2Observation]:
        obs = dict(payload.get("observation", {}))
        obs.pop("metadata", None)
        observation = Tbench2Observation(**obs, done=payload.get("done", False), reward=payload.get("reward"))
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
