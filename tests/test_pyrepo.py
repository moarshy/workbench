"""
pyrepo: the upstream code scores full marks, the generated task is broken, partial
fixes get partial credit, the second bug is hidden from the visible suite, and
editing tests or adding conftest files does not help.
"""

import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import pyrepo  # noqa: E402

VENDOR = pyrepo.VENDOR


def make(tmp_path, level, seed):
    ws = tmp_path / f"py-{level}-{seed}"
    ws.mkdir()
    return ws, pyrepo.generate(random.Random(seed), level, ws)


def restore(ws, task, entries):
    """Put the upstream version of the files touched by `entries` back."""
    lib = task.meta["lib"]
    for e in entries:
        shutil.copyfile(VENDOR / lib / e["module"], ws / "repo" / lib / e["module"])


def score(task, ws):
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level", list(pyrepo.LEVELS))
def test_upstream_code_gets_full_marks_and_task_starts_broken(tmp_path, level):
    for seed in range(4):
        ws, task = make(tmp_path, level, seed)
        passed, total, checks = score(task, ws)
        assert passed < total, (level, seed, checks)
        restore(ws, task, task.meta["entries"])
        passed, total, checks = score(task, ws)
        assert passed == total, (level, seed, task.meta["entries"][0]["func"], [c for c in checks if not c[1]])


def test_fixing_only_the_visible_bug_is_partial(tmp_path):
    for seed in range(3):
        ws, task = make(tmp_path, "bug2", seed)
        restore(ws, task, task.meta["entries"][:1])
        if task.meta["entries"][0]["module"] == task.meta["entries"][1]["module"]:
            continue  # same file: restoring it fixes both; nothing partial to test
        passed, total, _ = score(task, ws)
        assert 0 < passed < total


def test_second_bug_is_invisible_to_the_visible_suite(tmp_path):
    for seed in range(3):
        ws, task = make(tmp_path, "bug2", seed)
        lib = task.meta["lib"]
        e1, e2 = task.meta["entries"]
        restore(ws, task, [e1])
        if e1["module"] == e2["module"]:
            continue
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ws / "repo" / lib,
                             capture_output=True, text=True).stdout
        assert " passed" in out and " failed" not in out, (seed, e2["func"], out[-400:])  # bug 2 still present, yet green


def test_editing_tests_or_adding_conftest_does_not_help(tmp_path):
    ws, task = make(tmp_path, "bug1", 7)
    lib = task.meta["lib"]
    tests = ws / "repo" / lib / "tests"
    for f in tests.glob("test_*.py"):
        f.write_text("def test_ok():\n    assert True\n")
    (tests / "conftest.py").write_text("import pytest\n@pytest.fixture(autouse=True)\ndef x():\n    pass\n")
    (ws / "repo" / lib / "pytest.ini").write_text("[pytest]\naddopts = -k nothing\n")
    passed, total, _ = score(task, ws)
    assert passed < total


def test_instruction_names_a_real_failing_test(tmp_path):
    ws, task = make(tmp_path, "bug1", 3)
    lib = task.meta["lib"]
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ws / "repo" / lib,
                         capture_output=True, text=True).stdout
    assert " failed" in out
    assert task.meta["named"].split("::")[1].split("[")[0] in out or task.meta["named"] in task.instruction
