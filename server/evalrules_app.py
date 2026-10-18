# SPDX-License-Identifier: BSD-3-Clause

"""HTTP/WebSocket server for the eval-rules desk (v9)."""

from openenv.core.env_server.http_server import create_app

try:
    from ..evalrules_models import PostTrainAction, PostTrainObservation
    from .evalrules_environment import EvalRulesEnvironment
except ImportError:
    from evalrules_models import PostTrainAction, PostTrainObservation
    from server.evalrules_environment import EvalRulesEnvironment

app = create_app(EvalRulesEnvironment, PostTrainAction, PostTrainObservation, env_name="evalrules",
                 max_concurrent_envs=1)
