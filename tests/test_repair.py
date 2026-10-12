"""
The repair family: the documented fix scores full marks, partial fixes get partial
credit, "fixing" the decoy is penalised, and every injected bug is detectable.
"""

import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import repair  # noqa: E402


def make(tmp_path, level, seed):
    ws = tmp_path / f"rep-{level}-{seed}"
    ws.mkdir()
    return ws, repair.generate(random.Random(seed), level, ws)


def write_sources(ws, task, bugs):
    pkg = ws / "project" / "src" / task.meta["pkg"]
    for mod, src in repair.sources(task.meta["c"], set(bugs)).items():
        (pkg / f"{mod}.py").write_text(src)


def score(task, ws):
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level", list(repair.LEVELS))
def test_documented_fix_gets_full_marks(tmp_path, level):
    for seed in range(15):
        ws, task = make(tmp_path, level, seed)
        assert score(task, ws)[0] < score(task, ws)[1]  # broken as generated
        write_sources(ws, task, [])
        passed, total, checks = score(task, ws)
        assert passed == total, (level, seed, task.meta["bugs"], [c for c in checks if not c[1]])


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_fixing_only_the_visible_bug_is_partial(tmp_path, level):
    for seed in range(10):
        ws, task = make(tmp_path, level, seed)
        write_sources(ws, task, [b for b in task.meta["bugs"] if b != task.meta["visible"]])
        passed, total, _ = score(task, ws)
        assert 0 < passed < total


def test_breaking_the_decoy_is_penalised(tmp_path):
    for seed in range(5):
        ws, task = make(tmp_path, "hard", seed)
        write_sources(ws, task, ["days_between"])  # "fix" the inclusive +1 that looks off-by-one
        _, _, checks = score(task, ws)
        assert dict(checks)["days_between kept as specified (inclusive count)"] is False


def test_visible_tests_expose_exactly_the_reported_bug(tmp_path):
    for level in repair.LEVELS:
        for seed in range(10):
            ws, task = make(tmp_path, level, seed)
            out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                                 cwd=ws / "project", capture_output=True, text=True).stdout
            failed = {line.split("::")[1].split(" ")[0] for line in out.splitlines() if line.startswith("FAILED")}
            assert failed and all(f"test_{task.meta['visible']}_" in name for name in failed), (level, seed, out[-500:])


def test_broken_package_scores_low(tmp_path):
    ws, task = make(tmp_path, "medium", 3)
    (ws / "project" / "src" / task.meta["pkg"] / "money.py").write_text("raise ImportError('broken')\n")
    passed, total, _ = score(task, ws)
    assert passed <= 1
