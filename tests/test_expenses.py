"""
The expense audit must be solvable from the visible files alone, and the caveats
must matter. The reference solver reads only the workspace.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import expenses  # noqa: E402


def solve_expenses(ws: Path, use_updates: bool = True) -> dict:
    task = (ws / "TASK.md").read_text()
    dept, lo, hi = re.search(r"audit of the (\w+) department's .*?\((\S+) to (\S+)\)", task).groups()
    policy = (ws / "policy" / "travel_policy.md").read_text()
    tiers = {}
    for t in "AB":
        for city in re.search(rf"Tier {t} cities: (.+)\.", policy).group(1).split(", "):
            tiers[city] = t
    caps = {}
    for cat, a, b in re.findall(r"\| (\w+) \| (\d+) \| (\d+) \|", policy):
        caps[(cat, "A")], caps[(cat, "B")] = int(a), int(b)
    receipt = int(re.search(r"Receipts are required for any claim above (\d+) EUR", policy).group(1))

    amendments = []  # (effective date, kind, key, value)
    if use_updates:
        for f in (ws / "policy" / "amendments").iterdir():
            text = f.read_text()
            eff = re.search(r"Effective date: (\S+)", text).group(1)
            if m := re.search(r"The (\w+) cap for tier (\w) cities is changed to (\d+) EUR", text):
                amendments.append((eff, "cap", (m.group(1), m.group(2)), int(m.group(3))))
            if m := re.search(r"(.+) is reclassified as a tier (\w) city", text):
                amendments.append((eff, "tier", m.group(1).strip(), m.group(2)))
            if m := re.search(r"Receipts are required for any claim above (\d+) EUR", text):
                amendments.append((eff, "receipt", None, int(m.group(1))))

    def rule(kind, key, base, when):
        vals = [(e, v) for e, k, kk, v in amendments if k == kind and kk == key and e <= when]
        return max(vals)[1] if vals else base

    rates = {}
    with open(ws / "finance" / "fx_rates.csv") as f:
        for r in csv.DictReader(f):
            rates[(r["date"], r["currency"])] = float(r["eur_per_unit"])
    if use_updates:
        for d, cur, r in re.findall(r"\| (\d{4}-\d{2}-\d{2}) \| (\w+) \| ([\d.]+) \|",
                                    (ws / "finance" / "fx_corrections.md").read_text()):
            rates[(d, cur)] = float(r)

    depts = {r["employee_id"]: r["department"] for r in csv.DictReader(open(ws / "data" / "employees.csv"))}
    violations, by_emp = [], {}
    for c in csv.DictReader(open(ws / "data" / "claims.csv")):
        if depts[c["employee_id"]] != dept or not (lo <= c["date"] <= hi):
            continue
        d = c["date"]
        eur = round(float(c["amount"]) * (1.0 if c["currency"] == "EUR" else rates[(d, c["currency"])]), 2)
        if c["category"] == "flights":
            bad, pay = not c["approval_id"], (eur if c["approval_id"] else 0.0)
        else:
            tier = rule("tier", c["city"], tiers[c["city"]], d)
            cap = rule("cap", (c["category"], tier), caps[(c["category"], tier)], d)
            if eur > rule("receipt", None, receipt, d) and c["receipt"] != "yes":
                bad, pay = True, 0.0
            else:
                bad, pay = eur > cap, min(eur, cap)
        if bad:
            violations.append(c["claim_id"])
        by_emp[c["employee_id"]] = round(by_emp.get(c["employee_id"], 0.0) + pay, 2)
    return {"violations": sorted(violations), "by_employee": by_emp, "reimbursable_eur": round(sum(by_emp.values()), 2)}


def make(tmp_path, level, seed):
    ws = tmp_path / f"exp-{level}-{seed}"
    ws.mkdir()
    return ws, expenses.generate(random.Random(seed), level, ws)


def score(task, ws):
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks)


@pytest.mark.parametrize("level", list(expenses.LEVELS))
def test_reference_solver_gets_full_marks(tmp_path, level):
    for seed in range(20):
        ws, task = make(tmp_path, level, seed)
        (ws / "output" / "expense_report.json").write_text(json.dumps(solve_expenses(ws)))
        passed, total = score(task, ws)
        assert passed == total, (level, seed, task.meta["caveats"], [c for c in task.verify(ws) if not c[1]])


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_ignoring_amendments_and_corrections_fails(tmp_path, level):
    for seed in range(20):
        ws, task = make(tmp_path, level, seed)
        (ws / "output" / "expense_report.json").write_text(json.dumps(solve_expenses(ws, use_updates=False)))
        passed, total = score(task, ws)
        assert passed < total, (level, seed, task.meta["caveats"], task.meta["affected"])


def test_every_amendment_is_written_and_scope_is_nontrivial(tmp_path):
    for level in expenses.LEVELS:
        for seed in range(30):
            ws, task = make(tmp_path, level, seed)
            assert len(list((ws / "policy" / "amendments").iterdir())) == task.meta["n_amendments"]
            assert task.meta["violations"] and task.meta["n_in_scope"] >= 20


def test_missing_output_scores_zero(tmp_path):
    ws, task = make(tmp_path, "easy", 0)
    assert score(task, ws)[0] == 0


def test_hard_size(tmp_path):
    ws, task = make(tmp_path, "hard", 1)
    assert set(task.meta["caveats"]) == {"cap", "cap_superseded", "tier", "fx", "future"}
    assert sum(1 for _ in open(ws / "data" / "claims.csv")) > 3000
