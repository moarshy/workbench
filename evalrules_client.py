"""Client for the eval-rules desk (the organizers' PostTrain protocol)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .evalrules_models import PostTrainAction, PostTrainObservation


class EvalRulesEnv(EnvClient[PostTrainAction, PostTrainObservation, State]):
    def _step_payload(self, action: PostTrainAction) -> Dict:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[PostTrainObservation]:
        obs = dict(payload.get("observation", {}))
        obs.pop("metadata", None)
        observation = PostTrainObservation(**obs, done=payload.get("done", False), reward=payload.get("reward"))
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
