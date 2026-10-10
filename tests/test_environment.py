import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WorkbenchAction  # noqa: E402
from server.sandbox import run_bash, truncate  # noqa: E402
from server.workbench_environment import MAX_STEPS, TASKS, WorkbenchEnvironment  # noqa: E402
from tests.test_downtime import solve  # noqa: E402
from tests.test_expenses import solve_expenses  # noqa: E402
from tests.test_routing import solve_routing  # noqa: E402

SOLVERS = {"downtime": (solve, "downtime.json"), "expenses": (solve_expenses, "expense_report.json"),
           "routing": (solve_routing, "plan.json")}


def test_sandbox_basics(tmp_path):
    r = run_bash("echo hi && exit 3", str(tmp_path))
    assert r.output.strip() == "hi" and r.exit_code == 3
    r = run_bash("python3 -c \"print('x' * 20000)\"", str(tmp_path))
    assert "output truncated: 20001 characters" in r.output and len(r.output) < 8400
    r = run_bash("sleep 5", str(tmp_path), timeout=1)
    assert r.exit_code == 124 and "timed out" in r.output
    assert len(truncate("abc", 2)) > 2  # adds a truncation note


def test_full_episode_with_reference_solution(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    for task_id in TASKS:
        solver, filename = SOLVERS[task_id.rsplit("-", 1)[0]]
        obs = env.reset(seed=11, task_id=task_id)
        assert filename in obs.output and obs.steps_left == MAX_STEPS and not obs.done
        obs = env.step(WorkbenchAction(type="run_bash", command="ls -R . | head -20 && wc -l data/*.csv"))
        assert "data" in obs.output and obs.exit_code == 0
        answer = json.dumps(solver(env._root))
        env.step(WorkbenchAction(type="run_bash", command=f"cat > output/{filename} <<'EOF'\n{answer}\nEOF"))
        obs = env.step(WorkbenchAction(type="submit", answer="done"))
        assert obs.done and obs.reward == 1.0, (task_id, obs.output)


def test_extra_action_fields_are_ignored():
    assert WorkbenchAction(**{"type": "submit", "answer": "ok", "done": True}).type == "submit"


def test_reset_wipes_previous_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    env.reset(seed=1, task_id="downtime-easy")
    env.step(WorkbenchAction(type="run_bash", command="echo leftover > scratch.txt"))
    env.reset(seed=2, task_id="downtime-easy")
    assert not (env._root / "scratch.txt").exists()


def test_running_out_of_steps_grades_and_ends(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    env.reset(seed=1, task_id="downtime-easy")
    for i in range(MAX_STEPS):
        obs = env.step(WorkbenchAction(type="run_bash", command="true"))
        assert obs.done == (i == MAX_STEPS - 1)
    assert obs.reward == 0.0 and "Out of steps" in obs.output
    assert env.step(WorkbenchAction(type="submit")).done
