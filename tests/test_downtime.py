"""
The downtime family must be solvable from the visible files alone, and the
caveats must matter. The reference solver below reads only the workspace,
never the generator's ground truth.
"""

import csv
import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import downtime  # noqa: E402

SHIFT_HOURS = {"day": (6, 14), "evening": (14, 22), "night": (22, 30)}


def solve(ws: Path, use_amendments: bool = True) -> dict:
    task = (ws / "TASK.md").read_text()
    shift, day = re.search(r"for the (\w+) shift of (\d{4}-\d{2}-\d{2})", task).groups()
    target = datetime.fromisoformat(day)

    limit = int(re.search(r"more than (\d+) minutes of unplanned downtime",
                          (ws / "handbook" / "maintenance_manual.md").read_text()).group(1))
    micro = None
    if use_amendments:
        limit_rules, micro_rules = [], []
        for f in sorted((ws / "handbook" / "amendments").iterdir()):
            text = f.read_text()
            eff = datetime.fromisoformat(re.search(r"Effective date: (\S+)", text).group(1))
            if eff > target:
                continue  # not yet in force
            if m := re.search(r"limit is changed from \d+ to (\d+) minutes", text):
                limit_rules.append((eff, int(m.group(1))))
            if m := re.search(r"Stops shorter than (\d+) minutes are classified as micro-stops", text):
                micro_rules.append((eff, int(m.group(1))))
        if limit_rules:
            limit = max(limit_rules)[1]
        if micro_rules:
            micro = max(micro_rules)[1]

    h0, h1 = SHIFT_HOURS[shift]
    lo, hi = target + timedelta(hours=h0), target + timedelta(hours=h1)
    series = {}
    with open(ws / "data" / "sensor_log.csv") as f:
        for row in csv.DictReader(f):
            series.setdefault(row["machine_id"], []).append((datetime.fromisoformat(row["timestamp"]), row["state"]))
    out = {}
    for m, rows in series.items():
        total, run, run_start = 0, 0, None
        for i, (ts, s) in enumerate(rows + [(None, "END")]):
            if s == "DOWN":
                if run == 0:
                    run_start = ts
                run += 1
            else:
                if run and lo <= run_start < hi and (micro is None or run >= micro):
                    total += run
                run = 0
        out[m] = total
    return {"limit_min": limit, "downtime_by_machine": out,
            "over_limit": sorted(m for m, v in out.items() if v > limit)}


def make(tmp_path, level, seed):
    ws = tmp_path / f"{level}-{seed}"
    ws.mkdir()
    return ws, downtime.generate(random.Random(seed), level, ws)


def score(task, ws):
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks)


@pytest.mark.parametrize("level", list(downtime.LEVELS))
def test_reference_solver_gets_full_marks(tmp_path, level):
    for seed in range(25):
        ws, task = make(tmp_path, level, seed)
        (ws / "output" / "downtime.json").write_text(json.dumps(solve(ws)))
        passed, total = score(task, ws)
        assert passed == total, (level, seed, task.meta, [c for c in task.verify(ws) if not c[1]])


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_ignoring_amendments_fails(tmp_path, level):
    for seed in range(8):
        ws, task = make(tmp_path, level, seed)
        (ws / "output" / "downtime.json").write_text(json.dumps(solve(ws, use_amendments=False)))
        passed, total = score(task, ws)
        assert passed < total, (level, seed, task.meta["caveats"])


def test_every_amendment_is_written(tmp_path):
    for level in downtime.LEVELS:
        for seed in range(60):  # seeds 10 and 21 once lost an amendment to a filename clash
            ws, task = make(tmp_path, level, seed)
            assert len(list((ws / "handbook" / "amendments").iterdir())) == task.meta["n_amendments"]


def test_missing_or_broken_output_scores_zero(tmp_path):
    ws, task = make(tmp_path, "easy", 0)
    assert score(task, ws)[0] == 0
    (ws / "output" / "downtime.json").write_text("{not json")
    assert score(task, ws)[0] == 0


def test_answers_are_nontrivial_and_seeded(tmp_path):
    for level in downtime.LEVELS:
        for seed in range(5):
            ws, task = make(tmp_path, level, seed)
            over, machines = task.meta["over"], task.meta["machines"]
            assert 0 < len(over) < len(machines)
            ws2 = tmp_path / f"again-{level}-{seed}"
            ws2.mkdir()
            again = downtime.generate(random.Random(seed), level, ws2)
            assert again.meta["truth"] == task.meta["truth"]


def test_hard_has_all_traps(tmp_path):
    ws, task = make(tmp_path, "hard", 3)
    assert set(task.meta["caveats"]) == {"micro", "limit", "superseded", "future"}
    assert len(list((ws / "handbook" / "amendments").iterdir())) >= 7
    assert sum(1 for _ in open(ws / "data" / "sensor_log.csv")) > 50_000  # far too big to read in a context window
