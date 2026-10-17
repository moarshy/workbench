"""
The loan audit must be solvable from the visible files alone, and each trap
(memos, the false premise, payments after the reporting date) must matter.
The reference solver reads only the workspace.
"""

import csv
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.families import loans  # noqa: E402


def solve_loans(ws: Path, use_memos=True, trust_request=False, use_after=True) -> dict:
    task = (ws / "TASK.md").read_text()
    report = re.search(r"provisions as at (\d{4}-\d{2}-\d{2})", task).group(1)
    named = re.findall(r"L-\d{4}", task.split("were written off")[0])
    policy = (ws / "policy" / "credit_policy.md").read_text()
    rates = {int(s): float(p) for s, p in re.findall(r"\| (\d) \| ([\d.]+) \|", policy)}
    stage2_from = 31
    returned = set()
    if use_memos:
        memos = []
        for f in (ws / "memos").iterdir():
            text = f.read_text()
            memos.append((re.search(r"Effective date: (\S+)", text).group(1), text))
        for eff, text in sorted(memos):
            if eff > report:
                continue
            if m := re.search(r"The stage (\d) provision rate is changed to ([\d.]+) percent", text):
                rates[int(m.group(1))] = float(m.group(2))
            if m := re.search(r"Stage 2 starts at (\d+) DPD", text):
                stage2_from = int(m.group(1))
            if "payments_returned.csv" in text:
                returned = {(r["loan_id"], r["due_date"]) for r in csv.DictReader(open(ws / "data" / "payments_returned.csv"))}
    written = {r["loan_id"] for r in csv.DictReader(open(ws / "data" / "writeoffs.csv"))}
    fx = {r["currency"]: float(r["eur_per_unit"]) for r in csv.DictReader(open(ws / "finance" / "fx_rates.csv"))
          if r["date"] == report}
    oldest = {}
    for r in csv.DictReader(open(ws / "data" / "payments.csv")):
        paid = r["paid_date"]
        unpaid = not paid or (use_after and paid > report) or (r["loan_id"], r["due_date"]) in returned
        if unpaid and (r["loan_id"] not in oldest or r["due_date"] < oldest[r["loan_id"]]):
            oldest[r["loan_id"]] = r["due_date"]
    excluded = written | (set(named) if trust_request else set())
    s2, s3, total = [], [], 0.0
    for ln in csv.DictReader(open(ws / "data" / "loans.csv")):
        lid = ln["loan_id"]
        if lid in excluded:
            continue
        dpd = (date.fromisoformat(report) - date.fromisoformat(oldest[lid])).days if lid in oldest else 0
        stage = 3 if dpd > 90 else (2 if dpd >= stage2_from else 1)
        (s2 if stage == 2 else s3 if stage == 3 else []).append(lid)
        eur = round(float(ln["outstanding_balance"]) * fx.get(ln["currency"], 1.0), 2)
        total += round(eur * rates[stage] / 100, 2)
    disc = [] if trust_request else sorted(x for x in named if x not in written)
    return {"stage2_loans": sorted(s2), "stage3_loans": sorted(s3), "total_provision_eur": round(total, 2),
            "discrepancies": disc}


def make(tmp_path, level, seed):
    ws = tmp_path / f"loans-{level}-{seed}"
    ws.mkdir()
    return ws, loans.generate(random.Random(seed), level, ws)


def score(task, ws, answer):
    (ws / "output" / "provisions.json").write_text(json.dumps(answer))
    checks = task.verify(ws)
    return sum(ok for _, ok in checks), len(checks), checks


@pytest.mark.parametrize("level", list(loans.LEVELS))
def test_reference_solver_gets_full_marks(tmp_path, level):
    for seed in range(20):
        ws, task = make(tmp_path, level, seed)
        passed, total, checks = score(task, ws, solve_loans(ws))
        assert passed == total, (level, seed, task.meta["caveats"], [c for c in checks if not c[1]])


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_ignoring_memos_fails(tmp_path, level):
    for seed in range(20):
        ws, task = make(tmp_path, level, seed)
        passed, total, _ = score(task, ws, solve_loans(ws, use_memos=False))
        assert passed < total, (level, seed, task.meta["caveats"])


@pytest.mark.parametrize("level", list(loans.LEVELS))
def test_trusting_a_false_premise_fails(tmp_path, level):
    n_false = 0
    for seed in range(20):
        ws, task = make(tmp_path, level, seed)
        passed, total, _ = score(task, ws, solve_loans(ws, trust_request=True))
        if task.meta["named_false"]:
            n_false += 1
            assert passed <= total - 2, (level, seed)  # wrong discrepancies and a wrong stage list or total
        else:
            assert passed == total  # the claim was true: trusting it is right
    assert 10 <= n_false <= 19  # mostly false, sometimes true


def test_payments_after_the_reporting_date_matter(tmp_path):
    for seed in range(10):
        ws, task = make(tmp_path, "hard", seed)
        passed, total, _ = score(task, ws, solve_loans(ws, use_after=False))
        assert passed < total, seed


def test_missing_or_malformed_output_scores_zero(tmp_path):
    ws, task = make(tmp_path, "easy", 0)
    assert sum(ok for _, ok in task.verify(ws)) == 0
    (ws / "output" / "provisions.json").write_text("[1, 2]")
    assert sum(ok for _, ok in task.verify(ws)) == 0


def test_sizes(tmp_path):
    ws, task = make(tmp_path, "hard", 1)
    assert set(task.meta["caveats"]) >= {"superseded", "future", "after", "fx"}
    assert sum(1 for _ in open(ws / "data" / "payments.csv")) > 2000
