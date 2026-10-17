import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WorkbenchAction  # noqa: E402
from server.sandbox import run_bash, truncate  # noqa: E402
from server.workbench_environment import (EFFICIENCY_BONUS, INPUTS_CHECK, MAX_STEPS, TASKS,  # noqa: E402
                                          WorkbenchEnvironment, reward_for)
from tests.test_downtime import solve  # noqa: E402
from tests.test_expenses import solve_expenses  # noqa: E402
from tests.test_routing import solve_routing  # noqa: E402
from server.families import pyrepo, repair  # noqa: E402
from tests.test_plant import solve_plant  # noqa: E402
from tests.test_reflow import solve_reflow  # noqa: E402
from tests.test_loans import solve_loans  # noqa: E402

SOLVERS = {"downtime": (solve, "downtime.json"), "expenses": (solve_expenses, "expense_report.json"),
           "routing": (solve_routing, "plan.json"), "reflow": (solve_reflow, "reflow_report.json"),
           "plant": (solve_plant, None), "loans": (solve_loans, "provisions.json")}


def test_sandbox_basics(tmp_path):
    r = run_bash("echo hi && exit 3", str(tmp_path))
    assert r.output.strip() == "hi" and r.exit_code == 3
    r = run_bash("python3 -c \"print('x' * 20000)\"", str(tmp_path))
    assert "output truncated: 20001 characters" in r.output and len(r.output) < 3400
    r = run_bash("python3 -c \"print('a' * 5000 + 'THE_END')\"", str(tmp_path))
    assert r.output.rstrip().endswith("THE_END")  # the tail survives truncation
    r = run_bash("sleep 5", str(tmp_path), timeout=1)
    assert r.exit_code == 124 and "timed out" in r.output
    assert len(truncate("abc", 2)) > 2  # adds a truncation note


def test_full_episode_with_reference_solution(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    for task_id in TASKS:
        family = task_id.rsplit("-", 1)[0]
        obs = env.reset(seed=11, task_id=task_id)
        assert obs.steps_left == MAX_STEPS and not obs.done
        obs = env.step(WorkbenchAction(type="run_bash", command="ls -R . | head -20"))
        assert obs.exit_code == 0
        if family == "pyrepo":  # put the upstream library files back
            meta = env._task.meta
            for e in meta["entries"]:
                src = (pyrepo.VENDOR / meta["lib"] / e["module"]).read_text()
                env.step(WorkbenchAction(type="run_bash",
                                         command=f"cat > repo/{meta['lib']}/{e['module']} <<'EOF'\n{src}EOF"))
        elif family == "repair":  # write the documented implementation over the buggy modules
            meta = env._task.meta
            for mod, src in repair.sources(meta["c"], set()).items():
                env.step(WorkbenchAction(type="run_bash",
                                         command=f"cat > project/src/{meta['pkg']}/{mod}.py <<'EOF'\n{src}\nEOF"))
        else:
            solver, filename = SOLVERS[family]
            if filename is None:  # plant: the file name depends on the level
                filename = "gains.json" if "gains.json" in env._task.instruction else "params.json"
            assert filename in env._task.instruction
            answer = json.dumps(solver(env._root))
            env.step(WorkbenchAction(type="run_bash", command=f"cat > output/{filename} <<'EOF'\n{answer}\nEOF"))
        obs = env.step(WorkbenchAction(type="submit", answer="done"))
        # full marks, plus the success-gated efficiency bonus for the steps left
        assert obs.done and obs.reward == round(1.0 - EFFICIENCY_BONUS * (1 - obs.steps_left / MAX_STEPS), 4), (task_id, obs.output)


def test_two_actions_left_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    obs = env.reset(seed=1, task_id="downtime-easy")
    assert '{"type": "run_bash"' in obs.output  # worked examples in the first observation
    for i in range(MAX_STEPS - 2):
        obs = env.step(WorkbenchAction(type="run_bash", command="true"))
    assert "2 actions left" in obs.output and obs.steps_left == 2


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


def test_reward_rules():
    checks = [("schema", True), ("a", True), ("b", False), ("c", False)]
    assert reward_for(checks, ("schema",), 6) == 0.25  # partial: half the share of checks
    assert reward_for(checks, ("schema", "b"), 6) == 0.0  # a protected check failed
    assert abs(reward_for([("a", True), ("b", True)], (), 6) - (1.0 - EFFICIENCY_BONUS / 2)) < 1e-9  # success, half the steps used
    assert abs(reward_for([("a", True), ("b", True)], (), 0) - (1.0 - EFFICIENCY_BONUS)) < 1e-9  # success always beats partial
    assert reward_for([("a", True), ("b", True)], (), MAX_STEPS) <= 1.0


def test_changing_an_input_file_zeroes_the_reward(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    env.reset(seed=3, task_id="expenses-easy")
    answer = json.dumps(solve_expenses(env._root))
    env.step(WorkbenchAction(type="run_bash", command=f"cat > output/expense_report.json <<'EOF'\n{answer}\nEOF"))
    env.step(WorkbenchAction(type="run_bash", command="echo x >> data/claims.csv && mkdir -p scratch && echo ok > scratch/notes"))
    obs = env.step(WorkbenchAction(type="submit", answer="done"))
    assert obs.reward == 0.0 and (INPUTS_CHECK, False) in obs.metadata["checks"]


def test_editing_tests_zeroes_pyrepo_but_untouched_code_scores_partially(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    env = WorkbenchEnvironment()
    env.reset(seed=5, task_id="pyrepo-bug1")
    obs = env.step(WorkbenchAction(type="submit", answer="done"))
    assert 0 < obs.reward < 0.5, obs.metadata["checks"]  # nothing fixed, nothing broken
    env.reset(seed=5, task_id="pyrepo-bug1")
    lib = env._task.meta["lib"]
    env.step(WorkbenchAction(type="run_bash", command=f"echo '# x' >> $(ls repo/{lib}/tests/*.py | head -1)"))
    obs = env.step(WorkbenchAction(type="submit", answer="done"))
    assert obs.reward == 0.0
