"""
The delivery plan must be solvable from the visible files, the notes must matter,
and the verifier must reject infeasible or poor plans.
"""

import csv
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import routing  # noqa: E402


def solve_routing(ws: Path, use_notes: bool = True) -> dict:
    cfg = json.loads((ws / "data" / "vans.json").read_text())
    depot = (cfg["depot"]["lat"], cfg["depot"]["lon"])
    speed, service = cfg["speed_kmh"], cfg["service_minutes_per_stop"]
    start = int(cfg["departure"][:2]) * 60 + int(cfg["departure"][3:])
    caps = {v: d["capacity_kg"] for v, d in cfg["vans"].items()}
    mins = lambda s: int(s[:2]) * 60 + int(s[3:])  # noqa: E731
    orders = {r["order_id"]: (float(r["lat"]), float(r["lon"]), int(r["demand_kg"]), mins(r["window_start"]), mins(r["window_end"]))
              for r in csv.DictReader(open(ws / "data" / "orders.csv"))}
    notes = (ws / "data" / "ops_notes.md").read_text() if use_notes else ""
    only_v1 = set(re.findall(r"(C\d+) must be served by V1", notes))
    closed = {frozenset(p) for p in re.findall(r"directly between (C\d+) and (C\d+)", notes)}
    dropped = set(re.findall(r"Order (C\d+) was cancelled", notes))

    def dist(a, b):
        p1, l1, p2, l2 = map(math.radians, (*a, *b))
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
        return 2 * 6371 * math.asin(math.sqrt(h))

    def cost(route, van):
        if sum(orders[o][2] for o in route) > caps[van] or (van == "V2" and only_v1 & set(route)):
            return None
        if any(frozenset(p) in closed for p in zip(route, route[1:])):
            return None
        t, km, here = start, 0.0, depot
        for o in route:
            leg = dist(here, orders[o][:2])
            km, t = km + leg, max(t + leg / speed * 60, orders[o][3])
            if t > orders[o][4]:
                return None
            t, here = t + service, orders[o][:2]
        return km + dist(here, depot)

    def best(stops, van):
        cands = [(c, p) for p in itertools.permutations(stops) if (c := cost(p, van)) is not None]
        return min(cands) if cands else None

    todo = [o for o in orders if o not in dropped]
    top = None
    for k in range(len(todo) + 1):
        for a in itertools.combinations(todo, k):
            b = tuple(o for o in todo if o not in a)
            ra, rb = best(a, "V1"), best(b, "V2")
            if ra and rb and (top is None or ra[0] + rb[0] < top[0]):
                top = (ra[0] + rb[0], list(ra[1]), list(rb[1]))
    return {"routes": {"V1": top[1], "V2": top[2]}, "total_km": round(top[0], 2)}


def make(tmp_path, level, seed):
    ws = tmp_path / f"rt-{level}-{seed}"
    ws.mkdir()
    return ws, routing.generate(random.Random(seed), level, ws)


def score(task, ws, plan):
    (ws / "output" / "plan.json").write_text(json.dumps(plan))
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level", list(routing.LEVELS))
def test_reference_solver_gets_full_marks(tmp_path, level):
    for seed in range(6):
        ws, task = make(tmp_path, level, seed)
        passed, total, checks = score(task, ws, solve_routing(ws))
        assert passed == total, (level, seed, task.meta["notes"], [c for c in checks if not c[1]])


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_ignoring_notes_fails(tmp_path, level):
    for seed in range(6):
        ws, task = make(tmp_path, level, seed)
        passed, total, _ = score(task, ws, solve_routing(ws, use_notes=False))
        assert passed < total, (level, seed, task.meta["notes"])


def test_verifier_rejects_bad_plans(tmp_path):
    ws, task = make(tmp_path, "medium", 1)
    good = solve_routing(ws)
    passed, total, _ = score(task, ws, {**good, "total_km": good["total_km"] + 5})
    assert passed == total - 1  # only the distance claim is wrong
    routes = good["routes"]
    one_van = {"routes": {"V1": routes["V1"] + routes["V2"], "V2": []}, "total_km": good["total_km"]}
    _, _, checks = score(task, ws, one_van)  # every order on one van exceeds its capacity
    assert dict(checks)["routes respect capacity, time windows and the dispatch notes"] is False
    missing = {"routes": {"V1": routes["V1"][1:], "V2": routes["V2"]}, "total_km": good["total_km"]}
    assert score(task, ws, missing)[0] <= 2
    assert score(task, ws, {})[0] == 1  # only "is a JSON object" holds for {}
