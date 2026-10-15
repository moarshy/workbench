"""
Admission-style checks for a running T-Desk container: identical tbench2 schema,
sandbox, a reference solution scoring 1 on every task, and the arena's example
action ({"action_type": "evaluate"}) ending every task with a reward in [0, 1].

Usage: python scripts/check_tdesk.py http://127.0.0.1:8030 [--expect-user agent|root]
"""
import json
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.server.families import pyrepo  # noqa: E402
from workbench.server.tdesk_environment import TASKS  # noqa: E402
from workbench.tdesk_client import TDeskEnv  # noqa: E402
from workbench.tdesk_models import Tbench2Action  # noqa: E402

url = sys.argv[1]
expect_user = sys.argv[sys.argv.index("--expect-user") + 1] if "--expect-user" in sys.argv else "agent"
ok = True
TESTS = Path(__file__).resolve().parents[1] / "tests"


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{str(detail)[:300]}]" if not cond else ""))


live = json.loads(urllib.request.urlopen(f"{url}/schema").read())
ref = json.load(open("/tmp/tb2/submission.json"))["schema"]
check("schema identical to the operators' tbench2 smoke submission",
      live["action"] == ref["action"] and live["observation"] == ref["observation"])

with TDeskEnv(base_url=url).sync() as env:
    r = env.reset(task_id="reflow-hard", seed=1)
    first = r.observation.instruction
    r = env.step(Tbench2Action(action_type="exec", command="whoami; pwd; ls /opt/workbench 2>&1 | head -2"))
    check(f"commands run as {expect_user} in /app", r.observation.output.split()[:2] == [expect_user, "/app"], r.observation.output)
    if expect_user == "agent":
        check("server code unreadable to the agent", "Permission denied" in r.observation.output, r.observation.output)
    check("instruction repeated in every observation", r.observation.instruction == first)
    r = env.step(Tbench2Action(action_type="exec", command="python3 -c \"print('y' * 5000)\""))
    check("output not truncated", len(r.observation.output.strip()) == 5000)

for task_id in TASKS:
    fam, lvl = task_id.rsplit("-", 1)
    local = __import__(f"workbench.server.families.{fam}", fromlist=["x"]).generate(random.Random(31), lvl, Path(tempfile.mkdtemp()))
    with TDeskEnv(base_url=url).sync() as env:
        env.reset(task_id=task_id, seed=31)
        if fam == "pyrepo":
            for e in local.meta["entries"]:
                env.step(Tbench2Action(action_type="write_file", file_path=f"repo/{local.meta['lib']}/{e['module']}",
                                       content=(pyrepo.VENDOR / local.meta["lib"] / e["module"]).read_text()))
        else:
            test_file, fn = {"expenses": ("test_expenses.py", "solve_expenses"), "routing": ("test_routing.py", "solve_routing"),
                             "reflow": ("test_reflow.py", "solve_reflow"), "plant": ("test_plant.py", "solve_plant")}[fam]
            src = (TESTS / test_file).read_text().split("def make(")[0]
            src = "\n".join(l for l in src.splitlines() if not l.startswith(("import pytest", "from server.")))
            out = {"expenses": "expense_report.json", "routing": "plan.json", "reflow": "reflow_report.json",
                   "plant": "gains.json" if lvl == "tune" else "params.json"}[fam]
            env.step(Tbench2Action(action_type="write_file", file_path="solve.py",
                                   content=src + f"\nws = Path('/app')\njson.dump({fn}(ws), open(ws / 'output' / '{out}', 'w'))\n"))
            env.step(Tbench2Action(action_type="exec", command="python3 /app/solve.py"))
        r = env.step(Tbench2Action(action_type="evaluate"))
        check(f"{task_id}: reference solution scores 1.0 via evaluate", r.reward == 1.0, r.observation.output[-300:])
    with TDeskEnv(base_url=url).sync() as env:
        env.reset(task_id=task_id)
        r = env.step(Tbench2Action(action_type="evaluate"))
        check(f"{task_id}: example action evaluate ends with reward in [0,1]", r.done and 0 <= r.reward <= 1, r.reward)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
