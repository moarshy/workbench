"""T-Desk: tbench2-identical interface over the Workbench families."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import pyrepo  # noqa: E402
from server.families import repair  # noqa: E402
from server.tdesk_environment import TASKS, TASKSETS, TDeskEnvironment  # noqa: E402
from tdesk_models import Tbench2Action  # noqa: E402
from tests.test_downtime import solve  # noqa: E402
from tests.test_expenses import solve_expenses  # noqa: E402
from tests.test_plant import solve_plant  # noqa: E402
from tests.test_reflow import solve_reflow  # noqa: E402
from tests.test_routing import solve_routing  # noqa: E402

SOLVERS = {"downtime": (solve, "output/downtime.json"), "expenses": (solve_expenses, "output/expense_report.json"), "routing": (solve_routing, "output/plan.json"),
           "reflow": (solve_reflow, "output/reflow_report.json"), "plant": (solve_plant, None)}


def env_at(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    return TDeskEnvironment()


def test_reference_solutions_score_one_via_write_file_and_evaluate(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    for task_id in sorted(set(TASKSETS["fit"] + TASKSETS["roomy"])):
        family = task_id.rsplit("-", 1)[0]
        obs = env.reset(seed=5, task_id=task_id)
        assert obs.instruction and obs.action_type == "reset" and not obs.done
        if family == "repair":
            meta = env._task.meta
            for m, src in repair.sources(meta["c"], set()).items():
                obs = env.step(Tbench2Action(action_type="write_file", file_path=f"project/src/{meta['pkg']}/{m}.py", content=src))
                assert obs.success, obs.error
        elif family == "pyrepo":
            meta = env._task.meta
            for e in meta["entries"]:
                obs = env.step(Tbench2Action(action_type="write_file", file_path=f"repo/{meta['lib']}/{e['module']}",
                                             content=(pyrepo.VENDOR / meta["lib"] / e["module"]).read_text()))
                assert obs.success, obs.error
        else:
            solver, path = SOLVERS[family]
            if path is None:
                path = "output/gains.json" if "gains.json" in obs.instruction else "output/params.json"
            obs = env.step(Tbench2Action(action_type="write_file", file_path=path, content=json.dumps(solver(env._root))))
            assert obs.success, obs.error
        obs = env.step(Tbench2Action(action_type="evaluate"))
        assert obs.done and obs.reward == 1.0, (task_id, obs.output)


def test_doing_nothing_scores_zero_and_reward_is_binary(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    for task_id in sorted(set(TASKSETS["fit"] + TASKSETS["roomy"])):
        env.reset(seed=1, task_id=task_id)
        obs = env.step(Tbench2Action(action_type="evaluate"))
        assert obs.done and obs.reward == 0.0, task_id


def test_instruction_repeats_and_output_is_not_truncated(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    first = env.reset(seed=2, task_id="reflow-lite")
    obs = env.step(Tbench2Action(action_type="exec", command="python3 -c \"print('x' * 9000)\""))
    assert obs.instruction == first.instruction and len(obs.output.strip()) == 9000 and obs.success
    obs = env.step(Tbench2Action(action_type="exec", command="false"))
    assert not obs.success and obs.instruction == first.instruction and not obs.done


def test_interactive_actions_error_but_continue_and_writes_stay_inside(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    env.reset(seed=3, task_id="reflow-lite")
    obs = env.step(Tbench2Action(action_type="view", session_id="x"))
    assert not obs.success and not obs.done and "exec" in obs.error
    obs = env.step(Tbench2Action(action_type="write_file", file_path="../../escape.txt", content="no"))
    assert not obs.success and not (tmp_path / "escape.txt").exists()
    obs = env.step(Tbench2Action(action_type="close"))
    assert obs.done and obs.reward == 0.0  # close is graded like evaluate, so it always carries a reward


def test_default_action_is_exec():
    assert Tbench2Action(command="ls").action_type == "exec"
