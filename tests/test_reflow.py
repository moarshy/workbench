"""Reflow audit: solvable from the visible files alone; the traps matter."""

import csv
import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import reflow  # noqa: E402


def solve_reflow(ws: Path, careful: bool = True) -> dict:
    hb = (ws / "docs" / "handbook.md").read_text()
    hi = float(re.search(r"first exceeds (\d+) degrees C", hb).group(1))
    lo = float(re.search(r"at or above (\d+) degrees C", hb).group(1))
    liq = float(re.search(r"Liquidus of \S+: ([\d.]+) C", hb).group(1))
    limit = float(re.search(r"Maximum preheat ramp rate: ([\d.]+) C/s", hb).group(1))
    tal_lo, tal_hi = map(float, re.search(r"TAL must be between (\d+) and (\d+) s", hb).groups())
    peak_min = float(re.search(r"at least ([\d.]+) C", hb).group(1))
    coldest = "the coldest one" in hb
    rev = ws / "docs" / "handbook_revision_B.md"
    if careful and rev.exists():
        limit = float(re.search(r"lowered to ([\d.]+) C/s", rev.read_text()).group(1))
    units = {r["run_id"]: r["temp_unit"] for r in csv.DictReader(open(ws / "data" / "runs.csv"))}
    refs = {}
    if (ws / "data" / "boards.csv").exists():
        refs = {r["run_id"]: r["tc_id"] for r in csv.DictReader(open(ws / "data" / "boards.csv")) if r["role"] == "reference"}
    series = {}
    for r in csv.DictReader(open(ws / "data" / "thermocouples.csv")):
        v = float(r["temp"])
        if careful and units[r["run_id"]] == "F":
            v = (v - 32) * 5 / 9
        series.setdefault(r["run_id"], {}).setdefault(r["tc_id"], []).append((float(r["t_s"]), v))
    out = {}
    for run in units:
        if run not in series:
            out[run] = None
            continue
        tc = min(series[run], key=lambda t: (max(v for _, v in series[run][t]), t)) if coldest else refs[run]
        s = series[run][tc]
        dt = s[1][0] - s[0][0]
        ramps = []
        for (_, a), (_, b) in zip(s, s[1:]):
            if a > hi or b > hi:
                break  # preheat ends when the board first exceeds the upper temperature
            if a >= lo and b >= lo:
                ramps.append((b - a) / dt)
        tal = 0.0
        for (_, a), (_, b) in zip(s, s[1:]):
            if a >= liq and b >= liq:
                tal += dt
            elif a < liq <= b or b < liq <= a:
                f = (liq - a) / (b - a)
                tal += dt * ((1 - f) if b >= liq else f)
        out[run] = dict(tc_id=tc, max_preheat_ramp=round(max(ramps), 2), tal_s=round(tal, 2), peak_c=round(max(v for _, v in s), 2))
    task = (ws / "TASK.md").read_text()
    if '"violating_runs"' in task:
        return {"ramp_limit_c_per_s": limit, "max_preheat_ramp": {r: v["max_preheat_ramp"] for r, v in out.items()},
                "violating_runs": sorted(r for r, v in out.items() if v["max_preheat_ramp"] > limit)}
    runs, failing = {}, []
    for r, v in out.items():
        if v is None:
            runs[r] = {"tc_id": None, "max_preheat_ramp": None, "tal_s": None, "peak_c": None, "status": "non-compliant"}
            failing.append(r)
            continue
        ok = v["max_preheat_ramp"] <= limit and tal_lo <= v["tal_s"] <= tal_hi and v["peak_c"] >= peak_min
        runs[r] = {**v, "status": "compliant" if ok else "non-compliant"}
        if not ok:
            failing.append(r)
    return {"ramp_limit_c_per_s": limit, "runs": runs, "failing_runs": sorted(failing)}


def make(tmp_path, level, seed):
    ws = tmp_path / f"rf-{level}-{seed}"
    ws.mkdir()
    return ws, reflow.generate(random.Random(seed), level, ws)


def score(task, ws, answer):
    (ws / "output" / "reflow_report.json").write_text(json.dumps(answer))
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level", list(reflow.LEVELS))
def test_reference_solver_gets_full_marks(tmp_path, level):
    for seed in range(40):
        ws, task = make(tmp_path, level, seed)
        passed, total, checks = score(task, ws, solve_reflow(ws))
        assert passed == total, (level, seed, [c for c in checks if not c[1]])


def test_ignoring_revision_and_fahrenheit_fails_hard(tmp_path):
    for seed in range(20):
        ws, task = make(tmp_path, "hard", seed)
        passed, total, _ = score(task, ws, solve_reflow(ws, careful=False))
        assert passed < total, seed


def test_answers_are_nontrivial(tmp_path):
    for level in reflow.LEVELS:
        mixed = 0
        for seed in range(30):
            ws, task = make(tmp_path, level, seed)
            ans = solve_reflow(ws)
            bad = ans.get("violating_runs", ans.get("failing_runs"))
            mixed += 0 < len(bad) < len(task.meta["truth"])
        assert mixed >= 20, (level, mixed)  # most instances have some failing and some passing runs
