"""
Build the pyrepo catalogue: real bugs (single-token mutations) and blankable
functions in the vendored libraries, each validated against the library's own
test suite. Run once locally; the JSON is committed and shipped in the image.

A mutation is kept when at least 2 test cases, from 1..12 distinct test functions, fail with it
(detectable but localised; parametrised cases count once).
A function is "implementable" when blanking its body fails 2..25 tests and it
has a docstring to work from.

Usage: python scripts/build_catalog.py [--per-lib 80] [--workers 8]
"""

import argparse
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "server" / "vendor"
OUT = ROOT / "server" / "families" / "pyrepo_catalog.json"
LIBS = {"toolz": "toolz", "boltons": "boltons", "tabulate": "tabulate", "inflection": "inflection"}
SKIP_MODULES = {"__init__.py", "compatibility.py", "_signatures.py", "_version.py", "sandbox"}

CMP_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
BIN_SWAP = {"+": "-", "-": "+"}
BOOL_SWAP = {"and": "or", "or": "and"}


def run_suite(libdir: Path) -> set:
    """Failing test ids of the library's suite (or {'<collection>'} if it doesn't even collect)."""
    xml = libdir / "report.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x" if False else "-q",
                    f"--junitxml={xml}", "tests"], cwd=libdir, capture_output=True, timeout=300,
                   env={**os.environ, "PYTHONPATH": str(libdir)})
    if not xml.exists():
        return {"<collection>"}
    failed = set()
    for case in ET.parse(xml).getroot().iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(f"{case.get('classname')}::{case.get('name')}")
    for suite in ET.parse(xml).getroot().iter("testsuite"):
        if int(suite.get("errors", 0)) and not failed:
            failed.add("<collection>")
    return failed


def mutation_sites(src: str):
    """(func_qualname, lineno, before_line, after_line, kind) for single-line mutations inside functions."""
    tree = ast.parse(src)
    lines = src.splitlines()
    sites = []

    def walk(node, qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_func(child, f"{qual}{child.name}")
                walk(child, f"{qual}{child.name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{qual}{child.name}.")

    def visit_func(fn, qual):
        for n in ast.walk(fn):
            if getattr(n, "lineno", None) is None or n.lineno != getattr(n, "end_lineno", None):
                continue
            line = lines[n.lineno - 1]
            if isinstance(n, ast.Compare) and len(n.ops) == 1:
                seg = line[n.left.end_col_offset:n.comparators[0].col_offset]
                for a, b in CMP_SWAP.items():
                    if re.fullmatch(rf"\s*{re.escape(a)}\s*", seg):
                        new = line[:n.left.end_col_offset] + seg.replace(a, b) + line[n.comparators[0].col_offset:]
                        sites.append((qual, n.lineno, line, new, f"compare {a} -> {b}"))
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
                seg = line[n.left.end_col_offset:n.right.col_offset]
                a = "+" if isinstance(n.op, ast.Add) else "-"
                if re.fullmatch(rf"\s*{re.escape(a)}\s*", seg) and not isinstance(n.left, ast.Constant) or isinstance(n.left, ast.Constant) and isinstance(n.left.value, (int, float)):
                    if re.fullmatch(rf"\s*{re.escape(a)}\s*", seg):
                        new = line[:n.left.end_col_offset] + seg.replace(a, BIN_SWAP[a]) + line[n.right.col_offset:]
                        sites.append((qual, n.lineno, line, new, f"arith {a} -> {BIN_SWAP[a]}"))
            elif isinstance(n, ast.BoolOp) and len(n.values) == 2:
                seg = line[n.values[0].end_col_offset:n.values[1].col_offset]
                a = "and" if isinstance(n.op, ast.And) else "or"
                if re.fullmatch(rf"\s*{a}\s*", seg):
                    new = line[:n.values[0].end_col_offset] + seg.replace(a, BOOL_SWAP[a]) + line[n.values[1].col_offset:]
                    sites.append((qual, n.lineno, line, new, f"bool {a} -> {BOOL_SWAP[a]}"))
            elif isinstance(n, ast.Constant) and type(n.value) is int and n.value in (0, 1, 2):
                text = line[n.col_offset:n.end_col_offset]
                if text == str(n.value):
                    for v in {n.value + 1, n.value - 1} - {-1}:
                        new = line[:n.col_offset] + str(v) + line[n.end_col_offset:]
                        sites.append((qual, n.lineno, line, new, f"constant {n.value} -> {v}"))

    walk(tree, "")
    return [s for s in sites if s[2] != s[3] and not s[2].lstrip().startswith(("#", "assert", "raise"))]


def implement_sites(src: str):
    """(func_qualname, start_line, end_line, body_start_line) for top-level functions with docstrings."""
    tree = ast.parse(src)
    out = []
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and ast.get_docstring(fn) and len(fn.body) >= 2:
            doc = fn.body[0]
            n_body = fn.end_lineno - doc.end_lineno
            if 3 <= n_body <= 30 and not fn.name.startswith("_"):
                out.append((fn.name, fn.lineno, fn.end_lineno, doc.end_lineno + 1))
    return out


def blank_body(src: str, site) -> str:
    name, start, end, body_start = site
    lines = src.splitlines(keepends=True)
    body_line = lines[body_start - 1]
    indent = body_line[:len(body_line) - len(body_line.lstrip())] or "    "
    stub = f'{indent}raise NotImplementedError("TODO: implement {name} as described in its docstring")\n'
    return "".join(lines[:body_start - 1] + [stub] + lines[end:])


def try_candidate(job):
    lib, rel, kind, payload = job
    tmp = Path(tempfile.mkdtemp(prefix="cat-"))
    try:
        libdir = tmp / lib
        shutil.copytree(VENDOR / lib, libdir)
        path = libdir / rel
        src = path.read_text()
        if kind == "mutation":
            qual, lineno, before, after, desc = payload
            lines = src.splitlines(keepends=True)
            if lines[lineno - 1].rstrip("\n") != before:
                return None
            lines[lineno - 1] = after + "\n"
            path.write_text("".join(lines))
        else:
            path.write_text(blank_body(src, payload))
        failed = run_suite(libdir)
        funcs = {re.sub(r"\[.*\]$", "", f) for f in failed}  # parametrised cases count once
        if "<collection>" in failed or any(f.startswith("::") for f in failed) \
                or not (2 <= len(failed)) or not (1 <= len(funcs) <= 12):
            return None
        files = {f.split("::")[0] for f in failed}
        if len(files) > 2:
            return None
        if kind == "mutation":
            qual, lineno, before, after, desc = payload
            return dict(kind="mutation", lib=lib, module=rel, func=qual, line=lineno, before=before, after=after,
                        desc=desc, failing=sorted(failed))
        name, start, end, body_start = payload
        return dict(kind="implement", lib=lib, module=rel, func=name, start=start, end=end, body_start=body_start,
                    failing=sorted(failed))
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lib", type=int, default=80, help="mutation candidates tried per library")
    ap.add_argument("--impl-per-lib", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(0)
    jobs = []
    for lib, pkg in LIBS.items():
        base = run_suite(VENDOR / lib)
        assert not base, f"{lib} baseline has failures: {sorted(base)[:5]}"
        muts, impls = [], []
        for path in sorted((VENDOR / lib / pkg).rglob("*.py")):
            rel = str(path.relative_to(VENDOR / lib))
            src = path.read_text()
            tiny_init = path.name == "__init__.py" and src.count("\n") < 60
            if tiny_init or (path.name in SKIP_MODULES - {"__init__.py"}) or "sandbox" in rel:
                continue
            muts += [(lib, rel, "mutation", s) for s in mutation_sites(src)]
            impls += [(lib, rel, "implement", s) for s in implement_sites(src)]
        rng.shuffle(muts)
        rng.shuffle(impls)
        jobs += muts[:args.per_lib] + impls[:args.impl_per_lib]
        print(f"{lib}: {len(muts)} mutation sites, {len(impls)} implementable functions", flush=True)
    kept = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(try_candidate, jobs), 1):
            if res:
                kept.append(res)
            if i % 25 == 0:
                print(f"  tried {i}/{len(jobs)}, kept {len(kept)}", flush=True)
    for lib in LIBS:
        for v in ("mutation", "implement"):
            print(f"{lib:10s} {v:9s} {sum(1 for k in kept if k['lib'] == lib and k['kind'] == v)}")
    OUT.write_text(json.dumps(kept, indent=1))
    print(f"wrote {len(kept)} entries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
