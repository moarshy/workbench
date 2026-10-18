"""
Check a running eval-rules desk (v9) container the way the arena (and a cheating policy) would:
sandbox security, the agent's document libraries, head-only truncation, a reference-solved
episode per task (full reward), and the arena's example-action admission check.

Usage: python scripts/check_evalrules.py http://127.0.0.1:8210
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workbench.evalrules_client import EvalRulesEnv  # noqa: E402
from workbench.evalrules_models import PostTrainAction as A  # noqa: E402
from workbench.server.evalrules_environment import TASKS  # noqa: E402
from workbench.server.families import pyrepo, repair  # noqa: E402
from workbench.server.workbench_environment import FAMILIES  # noqa: E402
from tests.test_downtime import solve  # noqa: E402
from tests.test_expenses import solve_expenses  # noqa: E402
from tests.test_loans import solve_loans  # noqa: E402
from tests.test_reflow import solve_reflow  # noqa: E402

SOLVERS = {"downtime": (solve, "downtime.json"), "expenses": (solve_expenses, "expense_report.json"),
           "reflow": (solve_reflow, "reflow_report.json"), "loans": (solve_loans, "provisions.json")}
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8210"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))


def bash(env, cmd):
    return env.step(A(type="run_bash", command=cmd)).observation


def heredoc(path, text):
    return f"cat > {path} <<'WBEOF'\n{text}\nWBEOF"


with EvalRulesEnv(base_url=url).sync() as env:
    env.reset(task_id="loans-hard", seed=5)
    out = bash(env, "whoami; pwd").output
    check("commands run as the unprivileged agent user in /root", out.split() == ["agent", "/root"], out)
    out = bash(env, "ls /opt/workbench 2>&1; cat /proc/1/environ 2>&1 | head -c 80").output
    check("server code and environment are unreadable", out.count("Permission denied") == 2, out[:200])
    out = bash(env, "python3 -c 'import openpyxl, pdfplumber, pypdf, docx, pandas; print(\"libs\")'").output
    check("document libraries available to the agent", "libs" in out, out[-200:])
    obs = bash(env, "python3 -c \"print('a' * 9000 + 'TAIL')\"")
    check("output cut to the first 8,192 characters", obs.output.endswith("[benchflow output truncated]")
          and "TAIL" not in obs.output, obs.output[-80:])
    check("inputs are real xlsx and pdf files", "Microsoft Excel" in bash(env, "head -c 4 data/loans.xlsx | od -c | head -1; "
          "python3 -c \"import zipfile; print('Microsoft Excel' if zipfile.is_zipfile('data/loans.xlsx') else 'no')\"").output)

for task_id in TASKS:
    family, level = task_id.rsplit("-", 1)
    with EvalRulesEnv(base_url=url).sync() as env:
        env.reset(task_id=task_id, seed=21)
        if family == "pyrepo":
            local = pyrepo.generate(random.Random(21), level, Path(tempfile.mkdtemp()))
            lib = local.meta["lib"]
            for e in local.meta["entries"]:
                bash(env, heredoc(f"/root/repo/{lib}/{e['module']}", (pyrepo.VENDOR / lib / e["module"]).read_text()))
        elif family == "repair":
            local = repair.generate(random.Random(21), level, Path(tempfile.mkdtemp()))
            for m, src in repair.sources(local.meta["c"], set()).items():
                bash(env, heredoc(f"/root/project/src/{local.meta['pkg']}/{m}.py", src))
        else:  # the same seed regenerates the same data as CSV/Markdown locally
            solver, name = SOLVERS[family]
            plain = Path(tempfile.mkdtemp())
            FAMILIES[family].generate(random.Random(21), level, plain)
            bash(env, heredoc(f"/root/output/{name}", json.dumps(solver(plain))))
            shutil.rmtree(plain, ignore_errors=True)
        r = env.step(A(type="submit", answer="done"))
        check(f"{task_id}: reference solution inside the container gets reward 1.0", r.reward == 1.0,
              r.observation.output)
    with EvalRulesEnv(base_url=url).sync() as env:  # arena admission: example actions reach a terminal reward in [0, 1]
        env.reset(task_id=task_id)
        for a in [{"type": "run_bash", "command": "ls /root"}, {"type": "submit", "answer": "done"}]:
            r = env.step(A(**a))
        check(f"{task_id}: example actions end with done and reward in [0,1]", r.done and 0 <= r.reward <= 1, str(r.reward))
    with EvalRulesEnv(base_url=url).sync() as env:  # the finish_action path
        env.reset(task_id=task_id)
        r = env.step(A(type="finalize"))
        check(f"{task_id}: finalize ends with a reward in [0,1]", r.done and 0 <= r.reward <= 1, str(r.reward))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
