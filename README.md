# Workbench

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that trains **workspace habits**: explore a workspace too large to read, search it, catch the hidden caveat, compute with code rather than in your head, check the result, and write an exact output file. It was built for [OpenEnv Arena](https://openenvarena-arena.hf.space/).

**Why this design.** Base models already know enough about these domains. What breaks them in agentic tasks are habits:
- stopping once an answer looks plausible
- missing an amendment or erratum in another file
- doing arithmetic in their head
- overflowing the context window by printing whole files

Workbench rewards the opposite.

## Interface

Each reply is one JSON action. The interface is the same `run_bash` / `submit` shape as the PostTrain Arena OpenEnv adapter:

| action | effect |
|---|---|
| `{"type": "run_bash", "command": "..."}` | Runs the command in the workspace as an unprivileged user. Output is capped at 8,192 characters and each command times out after 120 s. |
| `{"type": "submit", "answer": "..."}` | Ends the task. The output files are graded. |

Tools inside the sandbox include `python3` with pandas and numpy, `grep`/`rg`, `jq`, `sqlite3` and coreutils. There is no network. The answer key lives only in the server process, and the server code can't be read from the sandbox. Tasks are graded automatically after 12 actions (20 in v2).

**Reward** = 0.5 × (share of checks passed) + 0.5 × (all checks passed). Partial credit gives GRPO a gradient, and the all-pass bonus pulls toward strict correctness.

## Task families

Every reset generates a fresh workspace with random data and randomly placed caveats.

| task | domain | what the model must do | the caveats |
|---|---|---|---|
| `downtime-{easy,medium,hard}` | industrial | Compute per-machine unplanned downtime for one shift from a per-minute sensor log (up to 52k rows) and a 600-line handbook. Write `downtime.json` | Micro-stop exclusion; superseded amendments; a limit revision; a **future-dated** revision to ignore; a night shift that crosses midnight |
| `expenses-{easy,medium,hard}` | office / finance | Audit one department's claims for one month (up to 3,000 multi-currency claims) against a travel policy. Write `expense_report.json` | **Mid-month** cap changes, applied by claim date; superseded caps; a city changing tier; corrected FX rates in a separate finance folder; a future-dated rule |
| `routing-{easy,medium,hard}` | math / formal reasoning (optimisation) | Plan two vans' routes with capacity and time windows. Write `plan.json`, which must be feasible and within 5% of the optimum | Dispatch notes: a van that can't serve a customer, a closed road between two stops, a cancelled order |
| `repair-{easy,medium,hard}` | software | Fix a 7-module Python billing package so it matches `docs/SPEC.md` and `docs/CHANGELOG.md`. Graded by hidden tests | Visible tests catch only one bug; the changelog overrides the spec (hard); a decoy that looks off-by-one but is specified (hard) |

Example: a `downtime-hard` workspace.

```
TASK.md
data/sensor_log.csv          51,841 rows
data/shifts.csv              night shift ends 06:00 the next day
handbook/maintenance_manual.md   607 lines; the limit is buried at a random line
handbook/amendments/         7 files: micro-stop rule 8 min (Jan) -> 5 min (Jul),
                             limit 45 -> 35 (Aug), 35 -> 80 (Oct, not yet in force), 3 unrelated
output/
```

## Calibration, v3 (12 actions per episode, Qwen3.8-27B, thinking off, 8 episodes per task)

| family | easy | medium | hard |
|---|---|---|---|
| downtime | 4/8 | 3/8 | 5/8 |
| expenses | 4/8 | 1/8 | 1/8 |
| routing | 7/8 | 5/8 | 4/8 |
| repair | 2/8 | 2/8 | 4/8 |

These numbers understate success. Of 45 failed replies, 11 were empty responses from the inference provider and 6 were JSON with raw newlines, which a lenient parser accepts. In v2, the arena's training reward (about 0.8) was well above our calibration.

## Calibration, v2 (20 actions per episode)

| family | easy | medium | hard |
|---|---|---|---|
| downtime | 7/8 | 5/8 | 4/8 |
| expenses | 4/8 | 4/8 | 3/8 |
| routing | 5/8 | 3/8 | 4/8 |

Counts are full-score episodes from `scripts/calibrate.py`. That script plays the episodes the way the arena trainer does: the system message is the action schema, observations are sent as JSON, and the first JSON object in each reply is taken as the action. Logs and transcripts are in `calibration/`.

Calibration showed that most failures were **format**, not reasoning:
- a sentence of reasoning before the JSON
- unescaped quotes from shell commands inside the JSON string
- Qwen's native `<tool_call>` format leaking out

With strict one-JSON-object parsing, medium and hard drop from about 5/8 to about 2/8. That is a habit RL can fix, and it carries over to any JSON-action environment.

## Verification

- **Independent reference solvers** (`tests/`) read only the files the model can see. They score 1.0 on every level and seed tested, for example 180/180 downtime workspaces.
- **Caveats matter:** solvers that ignore the amendments, corrections or notes always fail medium and hard.
- **Sandbox:** `scripts/check_container.py` checks that the agent runs as `agent` in `/app`, cannot read the server code or environment, has the expected tools, solves every task inside the container, and that the arena's example actions finish every task.
- **A bug the tests caught:** two amendments sharing an effective date once overwrote each other, which made 2 of 60 hard tasks unsolvable. There is now a regression test for it.

## Run it

```sh
uv sync && python -m pytest tests                    # 36 tests
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build -f server/Dockerfile -t workbench .
docker run --rm -p 8011:8000 workbench
python scripts/check_container.py http://127.0.0.1:8011
OPENROUTER_API_KEY=... python scripts/calibrate.py --episodes 8 --lenient --ports 8011
```

## Layout

```
models.py                       run_bash / submit action, observation
server/sandbox.py               command runner: agent user, rlimits, output cap, timeout
server/workbench_environment.py reset / step / grading
server/families/                downtime.py, expenses.py, routing.py (+ common.py)
tests/                          reference solvers and environment tests
scripts/                        calibrate.py, check_container.py
```

Built with OpenEnv pinned at `86a180ed` (the arena's revision). Earlier entry: [Stepwise Math](https://github.com/moarshy/stepwise-math), whose lessons led to this design.
