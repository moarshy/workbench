"""
Eval-rules desk: the organizers' interface, head-only truncation, xlsx/pdf inputs that keep
every task solvable, and the v9 reward (task score, gated clean-finish bonus, repeat penalty).
"""

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalrules_models import PostTrainAction  # noqa: E402
from server.evalrules_environment import (MAX_OUTPUT_CHARS, TASKS, TRUNCATION_MARK, EvalRulesEnvironment,  # noqa: E402
                                          head_only, reward_for)
from server.families import pyrepo, repair  # noqa: E402
from server.workbench_environment import FAMILIES  # noqa: E402
from tests.test_downtime import solve  # noqa: E402
from tests.test_expenses import solve_expenses  # noqa: E402
from tests.test_loans import solve_loans  # noqa: E402
from tests.test_reflow import solve_reflow  # noqa: E402

SOLVERS = {"downtime": (solve, "downtime.json"), "expenses": (solve_expenses, "expense_report.json"),
           "reflow": (solve_reflow, "reflow_report.json"), "loans": (solve_loans, "provisions.json")}


def env_at(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKBENCH_ROOT", str(tmp_path / "ws"))
    return EvalRulesEnvironment()


def bash(env, cmd):
    return env.step(PostTrainAction(type="run_bash", command=cmd))


def test_reward_rules():
    good = [("a", True), ("b", True)]
    assert reward_for(good, (), submitted=True, repeats=0) == 1.0
    assert reward_for(good, (), submitted=False, repeats=0) == 0.85          # died or finalized: no finish bonus
    half = [("schema", True), ("x", False)]
    assert reward_for(half, ("schema",), submitted=True, repeats=0) == 0.575  # 0.5 share: bonus applies
    low = [("schema", True), ("x", False), ("y", False)]
    assert reward_for(low, ("schema",), submitted=True, repeats=0) == round(0.85 / 3, 4)  # below 0.5: no bonus
    assert reward_for(half, ("x",), submitted=True, repeats=0) == 0.0         # protected check failed
    assert reward_for(good, (), submitted=True, repeats=10) == 0.8            # repeat penalty capped at 0.2
    assert reward_for([("a", False)], (), submitted=False, repeats=3) == 0.0  # clipped at 0
    assert reward_for([("s", False)], ("s",), submitted=False, repeats=0, turns=1) == 0.02  # survival: died on turn 2
    assert reward_for([("s", False)], ("s",), submitted=False, repeats=0, turns=9) == 0.1   # counts at most 5 turns
    assert reward_for(good, (), submitted=True, repeats=0, turns=5) == 1.0                  # clipped at 1


def test_head_only_truncation():
    assert head_only("x" * 10) == "x" * 10
    out = head_only("a" * 9000 + "TAIL")
    assert out.startswith("a" * MAX_OUTPUT_CHARS) and out.endswith(TRUNCATION_MARK) and "TAIL" not in out


def test_first_observation_is_the_instruction_alone(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    obs = env.reset(seed=1, task_id="loans-medium")
    assert obs.output == env._task.instruction and '{"type"' not in obs.output  # no worked examples
    assert ".xlsx" in obs.output and "loans.csv" not in obs.output and "credit_policy.pdf" in obs.output
    assert (env._root / "data" / "loans.xlsx").exists() and not (env._root / "data" / "loans.csv").exists()


def test_reference_solution_scores_full_on_every_task(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    for task_id in TASKS:
        family, level = task_id.rsplit("-", 1)
        env.reset(seed=13, task_id=task_id)
        if family == "pyrepo":
            meta = env._task.meta
            for e in meta["entries"]:
                shutil.copyfile(pyrepo.VENDOR / meta["lib"] / e["module"], env._root / "repo" / meta["lib"] / e["module"])
        elif family == "repair":
            meta = env._task.meta
            for mod, src in repair.sources(meta["c"], set()).items():
                (env._root / "project" / "src" / meta["pkg"] / f"{mod}.py").write_text(src + "\n")
        else:  # the same seed regenerates the same data in CSV/Markdown form; solve that copy
            solver, name = SOLVERS[family]
            plain = Path(tempfile.mkdtemp())
            FAMILIES[family].generate(random.Random(13), level, plain)
            assert name in env._task.instruction
            (env._root / "output" / name).write_text(json.dumps(solver(plain)))
        obs = env.step(PostTrainAction(type="submit", answer="done"))
        assert obs.done and obs.reward == 1.0, (task_id, obs.output, [c for c in obs.metadata["checks"] if not c[1]])


def test_finalize_and_repeats(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    env.reset(seed=2, task_id="expenses-easy")
    for _ in range(3):
        bash(env, "ls  /tmp >/dev/null; echo same")  # identical command and output: 2 repeats
    obs = env.step(PostTrainAction(type="finalize"))
    assert obs.done and obs.metadata["repeats"] == 2 and obs.metadata["submitted"] is False


def test_changing_an_xlsx_input_zeroes_the_reward(monkeypatch, tmp_path):
    env = env_at(monkeypatch, tmp_path)
    env.reset(seed=4, task_id="loans-medium")
    plain = Path(tempfile.mkdtemp())
    FAMILIES["loans"].generate(random.Random(4), "medium", plain)
    (env._root / "output" / "provisions.json").write_text(json.dumps(solve_loans(plain)))
    (env._root / "data" / "loans.xlsx").write_bytes(b"broken")
    obs = env.step(PostTrainAction(type="submit", answer="done"))
    assert obs.reward == 0.0


def test_the_model_can_read_the_converted_files(monkeypatch, tmp_path):
    """What an agent would do: open the workbook and PDF with the libraries the image provides."""
    env = env_at(monkeypatch, tmp_path)
    env.reset(seed=5, task_id="loans-medium")
    import openpyxl
    import pdfplumber
    rows = list(openpyxl.load_workbook(env._root / "data" / "payments.xlsx").active.iter_rows(values_only=True))
    assert rows[0] == ("loan_id", "due_date", "paid_date") and len(rows) > 500
    text = "".join(p.extract_text() or "" for p in pdfplumber.open(env._root / "policy" / "credit_policy.pdf").pages)
    assert "Days past due" in text and "above 90" in text
