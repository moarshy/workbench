"""
Software family: fix or complete a REAL Python library (toolz, boltons, tabulate, inflection).

Bugs come from a catalogue of single-token mutations (comparison, boolean,
arithmetic, off-by-one) that were validated against the library's own test
suite when the catalogue was built (scripts/build_catalog.py): each one breaks
1..12 test functions. Per level:

  bug1       one bug; the visible test suite shows it failing; one failing test is named
  bug2       two bugs in the same library; the visible suite shows only the first;
             the second arrives as a user bug report naming the misbehaving function
  implement  a function body is replaced by NotImplementedError; its tests are removed
             from the visible suite; the docstring is the spec

Grading copies only the agent's edits to the library's own .py files into a
fresh copy with the pristine, complete test suite (tests, conftest and config
written by the agent are never used) and runs it as the sandbox user.
"""

import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from .common import Check, TaskInstance

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "vendor"
# entries whose "failures" are whole test modules failing to import are not localised bugs; skip them
CATALOG = [e for e in json.loads((HERE / "pyrepo_catalog.json").read_text())
           if not any(t.startswith("::") for t in e["failing"])]
PKG = {"toolz": "toolz", "boltons": "boltons", "tabulate": "tabulate", "inflection": "inflection"}
LEVELS = {"bug1": dict(kind="mutation", n=1), "bug2": dict(kind="mutation", n=2), "implement": dict(kind="implement", n=1)}


def _func_name(test_id: str) -> str:
    """'tests.test_x.TestC::test_y[p]' -> ('tests/test_x.py', 'TestC', 'test_y')."""
    return re.sub(r"\[.*\]$", "", test_id)


def _remove_tests(test_file: Path, names: set) -> None:
    """Delete test functions / methods (with decorators) whose 'Class.func' or 'func' is in names."""
    src = test_file.read_text()
    tree = ast.parse(src)
    cut = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for f in node.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"{node.name}.{f.name}" in names:
                    cut.append(f)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names and node.col_offset == 0:
            cut.append(node)
    lines = src.splitlines(keepends=True)
    for f in sorted(cut, key=lambda f: -f.lineno):
        start = min([f.lineno] + [d.lineno for d in f.decorator_list]) - 1
        indent = re.match(r"\s*", lines[f.lineno - 1]).group(0)
        lines[start:f.end_lineno] = [f"{indent}pass  # test removed\n"] if indent else []
    test_file.write_text("".join(lines))


def _split(test_id: str):
    """classname 'tests.test_mod.TestC' + name -> (file 'tests/test_mod.py', key 'TestC.name' or 'name')."""
    cls, name = test_id.split("::")
    name = re.sub(r"\[.*\]$", "", name)
    parts = cls.split(".")
    mod_idx = max(i for i, p in enumerate(parts) if p.startswith("test"))
    file = "/".join(parts[:mod_idx + 1]) + ".py"
    key = ".".join(parts[mod_idx + 1:] + [name])
    return file, key


def _apply(entry: dict, libdir: Path) -> None:
    path = libdir / entry["module"]
    lines = path.read_text().splitlines(keepends=True)
    if entry["kind"] == "mutation":
        assert lines[entry["line"] - 1].rstrip("\n") == entry["before"]
        lines[entry["line"] - 1] = entry["after"] + "\n"
        path.write_text("".join(lines))
    else:
        body = lines[entry["body_start"] - 1]
        indent = body[:len(body) - len(body.lstrip())] or "    "
        stub = f'{indent}raise NotImplementedError("TODO: implement {entry["func"]} as described in its docstring")\n'
        path.write_text("".join(lines[:entry["body_start"] - 1] + [stub] + lines[entry["end"]:]))


def _pick(rng: random.Random, cfg: dict):
    pool = [e for e in CATALOG if e["kind"] == cfg["kind"]]
    first = rng.choice(pool)
    if cfg["n"] == 1:
        return [first]
    # second bug: same library, different function and different line, disjoint failing tests
    second = [e for e in pool if e["lib"] == first["lib"] and e["func"] != first["func"]
              and not ({_func_name(t) for t in e["failing"]} & {_func_name(t) for t in first["failing"]})]
    return [first, rng.choice(second)] if second else _pick(rng, cfg)


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    entries = _pick(rng, cfg)
    lib = entries[0]["lib"]
    repo = root / "repo"
    shutil.copytree(VENDOR / lib, repo / lib, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "report.xml"))
    libdir = repo / lib
    for e in entries:
        _apply(e, libdir)
    (libdir / "pytest.ini").write_text("[pytest]\npythonpath = .\naddopts = -p no:cacheprovider\n")
    (root / "output").mkdir(exist_ok=True)

    # visible tests: hide the tests that would reveal the second bug, or the blanked function's tests
    hidden = entries[1:] if level == "bug2" else (entries if level == "implement" else [])
    by_file = {}
    for e in hidden:
        for t in e["failing"]:
            f, key = _split(t)
            by_file.setdefault(f, set()).add(key)
    for f, keys in by_file.items():
        _remove_tests(libdir / f, keys)

    first = entries[0]
    named = sorted(first["failing"])[0] if level != "implement" else None
    pkg = PKG[lib]
    run = f"cd {libdir} && python3 -m pytest -q tests"
    if level == "bug1":
        body = (f"A recent change introduced a bug in the `{lib}` library at {libdir}. "
                f"For example, the test `{named}` now fails.\n\nFind the cause and fix it in the library code.")
    elif level == "bug2":
        body = (f"A recent change introduced bugs in the `{lib}` library at {libdir}. "
                f"For example, the test `{named}` now fails.\n\n"
                f"Separately, a user reported (issue #{rng.randint(300, 999)}): \"`{entries[1]['func']}` in "
                f"`{entries[1]['module']}` returns wrong results for some inputs since the last release.\" "
                f"No existing test catches that one.\n\nFix both problems in the library code.")
    else:
        body = (f"The function `{first['func']}` in `{libdir}/{first['module']}` is not implemented yet: it raises "
                f"NotImplementedError. Implement it so it behaves as its docstring describes, consistent with the "
                f"rest of the `{lib}` library.")
    instruction = f"""# Fix the {lib} library

{body}

- Run the visible tests with `{run}`. They do not cover everything: hidden tests will run the library's complete test suite.
- Change only the library's own code under {libdir}/{pkg}. Do not edit or add tests; edited tests are ignored when grading.
- Do not break anything that currently works.

Call submit when you are done.
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        failed, collected_ok = _grade(ws / "repo" / lib, lib)
        checks: List[Check] = [("library imports and the full test suite runs", collected_ok)]
        for i, e in enumerate(entries, 1):
            what = "implemented" if e["kind"] == "implement" else "fixed"
            checks.append((f"{what}: {e['func']} ({e['module']}) passes its tests",
                           collected_ok and not (set(e["failing"]) & failed)))
        checks.append(("complete test suite passes (no regressions)", collected_ok and not failed))
        return checks

    return TaskInstance(instruction, verify, meta=dict(lib=lib, entries=entries, named=named))


def _grade(agent_lib: Path, lib: str):
    """Run the pristine full suite against the agent's library code. Returns (failed test ids, collected_ok)."""
    try:
        from ..sandbox import _agent_identity, _limits
    except ImportError:
        from server.sandbox import _agent_identity, _limits
    tmp = Path(tempfile.mkdtemp(prefix="grade-"))
    try:
        dest = tmp / lib
        shutil.copytree(VENDOR / lib, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "report.xml"))
        pkg = PKG[lib]
        for src in (VENDOR / lib / pkg).rglob("*.py"):  # only files that exist upstream; never tests or config
            rel = src.relative_to(VENDOR / lib)
            mine = agent_lib / rel
            if mine.is_file() and not mine.is_symlink():
                shutil.copyfile(mine, dest / rel)
        (dest / "pytest.ini").write_text("[pytest]\npythonpath = .\n")
        ident = _agent_identity()
        if ident:
            for dirpath, dirnames, filenames in os.walk(tmp):
                for n in [dirpath, *(os.path.join(dirpath, x) for x in dirnames + filenames)]:
                    os.chown(n, *ident)
        xml = dest / f"report-{os.urandom(6).hex()}.xml"
        kwargs = dict(cwd=dest, capture_output=True, timeout=300,
                      env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(tmp)})
        if ident:
            kwargs.update(user=ident[0], group=ident[1], extra_groups=[], preexec_fn=_limits)
        # In the container (server runs as root) the system python3 has pytest, whether or not we could switch
        # users; on a developer machine use the current interpreter.
        python = "python3" if os.geteuid() == 0 else sys.executable
        try:
            subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", "tests"], **kwargs)
        except subprocess.TimeoutExpired:
            return set(), False
        if not xml.exists():
            return set(), False
        root = ET.parse(xml).getroot()
        failed, errors = set(), 0
        for suite in root.iter("testsuite"):
            errors += int(suite.get("errors", 0))
        for case in root.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed.add(f"{case.get('classname')}::{case.get('name')}")
        total = sum(1 for _ in root.iter("testcase"))
        return failed, total > 0 and not (errors and not failed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
