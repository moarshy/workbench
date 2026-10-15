# SPDX-License-Identifier: BSD-3-Clause

"""HTTP/WebSocket server for T-Desk (Workbench tasks behind the tbench2_env interface)."""

from openenv.core.env_server.http_server import create_app

try:
    from ..tdesk_models import Tbench2Action, Tbench2Observation
    from .tdesk_environment import TDeskEnvironment
except ImportError:
    from tdesk_models import Tbench2Action, Tbench2Observation
    from server.tdesk_environment import TDeskEnvironment

app = create_app(TDeskEnvironment, Tbench2Action, Tbench2Observation, env_name="tdesk", max_concurrent_envs=1)
