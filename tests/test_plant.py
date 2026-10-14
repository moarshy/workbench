"""plant-tune: identifiable from experiments alone; the tuning rule meets the spec; bad answers fail."""

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import plant  # noqa: E402


def load_plant(ws: Path):
    spec = importlib.util.spec_from_file_location(f"plant_{ws.name}", ws / "plant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve_plant(ws: Path) -> dict:
    p = load_plant(ws)
    u = 50.0
    y = p.simulate([u] * 1500, seed=1)
    amb = p.AMBIENT
    final = sum(y[-100:]) / 100
    K = (final - amb) / u
    thr = amb + 4 * p.NOISE  # first rise clearly above the measurement noise
    theta = next(i for i, v in enumerate(y) if v > thr)
    t63 = next(i for i, v in enumerate(y) if v >= amb + 0.632 * (final - amb))
    tau = t63 - theta + 1
    out = {"K": K, "tau_s": tau, "theta_s": theta}
    spec = (ws / "SPEC.md").read_text()
    if "gains.json" in (ws / "TASK.md").read_text():
        lam = float(re.search(r"lambda = ([\d.]+) \* tau", spec).group(1)) * tau
        kp = tau / (K * (lam + theta))
        out.update(kp=kp, ki=kp / tau)
    return out


def make(tmp_path, level, seed):
    ws = tmp_path / f"pl-{level}-{seed}"
    ws.mkdir()
    return ws, plant.generate(random.Random(seed), level, ws)


def score(task, ws, ans, name):
    (ws / "output" / name).write_text(json.dumps(ans))
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level,name", [("identify", "params.json"), ("tune", "gains.json")])
def test_step_test_solver_gets_full_marks(tmp_path, level, name):
    for seed in range(25):
        ws, task = make(tmp_path, level, seed)
        passed, total, checks = score(task, ws, solve_plant(ws), name)
        assert passed == total, (level, seed, task.meta, [c for c in checks if not c[1]])


def test_true_gains_meet_the_acceptance_criteria():
    for seed in range(60):
        m = plant.generate(random.Random(seed), "tune", Path(__import__("tempfile").mkdtemp())).meta
        series = plant._sim_closed(m["k"], m["tau"], m["theta"], m["ambient"], m["noise"], m["kp"], m["ki"],
                                   m["setpoint"], m["settle_limit"] + 200, seed=seed)
        ov, st = plant._metrics(series, m["ambient"], m["setpoint"])
        assert ov <= 0.05 and st is not None and st <= m["settle_limit"], (seed, ov, st, m)


def test_bad_answers_fail(tmp_path):
    ws, task = make(tmp_path, "tune", 3)
    good = solve_plant(ws)
    assert score(task, ws, {**good, "kp": good["kp"] * 3}, "gains.json")[0] < score(task, ws, good, "gains.json")[1]
    assert score(task, ws, {**good, "K": good["K"] * 1.3}, "gains.json")[0] < score(task, ws, good, "gains.json")[1]
    assert score(task, ws, {}, "gains.json")[0] == 1


def test_parameters_are_not_in_plain_text(tmp_path):
    ws, task = make(tmp_path, "identify", 2)
    src = (ws / "plant.py").read_text()
    assert str(task.meta["tau"]) not in src and f'"k": {task.meta["k"]}' not in src
