"""
Check a running Workbench container the way the arena (and a cheating policy) would:
security of the answer key, tools available, a full reference-solved episode per task,
and the arena's example-action admission check.

Usage: python scripts/check_container.py http://127.0.0.1:8000
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.client import WorkbenchEnv  # noqa: E402
from workbench.models import WorkbenchAction  # noqa: E402
from workbench.server.workbench_environment import TASKS  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))


def bash(env, cmd):
    r = env.step(WorkbenchAction(type="run_bash", command=cmd))
    return r.observation.output, r


with WorkbenchEnv(base_url=url).sync() as env:
    t = time.time()
    env.reset(task_id="downtime-hard", seed=5)
    check("reset of the biggest task takes < 30 s", time.time() - t < 30, f"{time.time() - t:.1f}s")
    out, _ = bash(env, "whoami; pwd")
    check("commands run as the unprivileged agent user in /app", out.split() == ["agent", "/app"], out)
    out, _ = bash(env, "ls /opt/workbench 2>&1; cat /opt/workbench/server/families/downtime.py 2>&1 | head -2")
    check("server code is unreadable to the agent", "Permission denied" in out and "def generate" not in out, out[:200])
    out, _ = bash(env, "ps aux | grep -c uvicorn; cat /proc/1/environ 2>&1 | head -c 100")
    check("agent cannot read the server's environment", "Permission denied" in out, out[:200])
    out, _ = bash(env, "python3 -c 'import pandas, numpy; print(pandas.__version__)'; jq --version; sqlite3 --version | cut -c1-6; rg --version | head -1")
    check("tools available: python3+pandas, jq, sqlite3, ripgrep", "2.2.3" in out and "jq-" in out and "ripgrep" in out, out)
    out, _ = bash(env, "curl -s -m 3 https://example.com >/dev/null; echo net=$?")
    print("info  outbound network from agent:", out.strip())

for task_id in TASKS:
    with WorkbenchEnv(base_url=url).sync() as env:
        env.reset(task_id=task_id, seed=21)
        # the reference solver from the tests, run inside the sandbox as the agent would
        family = task_id.rsplit("-", 1)[0]
        test_file, fn, out_name = {"downtime": ("test_downtime.py", "solve", "downtime.json"),
                                   "expenses": ("test_expenses.py", "solve_expenses", "expense_report.json"),
                                   "routing": ("test_routing.py", "solve_routing", "plan.json")}[family]
        solver = (Path(__file__).resolve().parents[1] / "tests" / test_file).read_text().split("def make(")[0]
        solver = "\n".join(l for l in solver.splitlines() if not l.startswith(("import pytest", "from server.")))
        script = solver + f"\nws = Path('/app')\njson.dump({fn}(ws), open(ws / 'output' / '{out_name}', 'w'))\nprint('written')"
        out, _ = bash(env, f"cat > /app/solve.py <<'PYEOF'\n{script}\nPYEOF\npython3 /app/solve.py")
        r = env.step(WorkbenchAction(type="submit", answer="done"))
        check(f"{task_id}: reference solution inside the container scores 1.0", r.reward == 1.0, f"{out[-300:]} | {r.observation.output}")

    with WorkbenchEnv(base_url=url).sync() as env:  # arena admission: example actions must reach a terminal reward in [0, 1]
        env.reset(task_id=task_id)
        for a in [{"type": "run_bash", "command": "ls /app"}, {"type": "submit", "answer": "done"}]:
            r = env.step(WorkbenchAction(**a))
        check(f"{task_id}: example actions end with done and reward in [0,1]", r.done and 0 <= r.reward <= 1, str(r.reward))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
