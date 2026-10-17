"""
Finance family: month-end loan-loss provisions.

The model gets a lender's loan book, the installment and payment history, a
write-off register, a long credit policy and a folder of dated risk-committee
memos. It must stage every loan by days past due (DPD) on the reporting date and
compute the total provision in EUR, in /app/output/provisions.json.

Traps:
- late caveat: a memo changes a provision rate from its effective date (on hard,
  a later memo supersedes it), or declares some payments returned by the bank
  (those installments count as unpaid);
- false premise: the request says three loans were written off, but (usually)
  not all of them are in the write-off register. The records win, and the
  contradicted IDs go under "discrepancies";
- hard also has a future-dated memo to ignore, payments recorded after the
  reporting date (unpaid on that date) and balances in several currencies.
"""

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from .common import Check, SCHEMA_CHECKS, TaskInstance, load_json_object, num_eq

LEVELS = {
    "easy": dict(loans=80, memo_caveats=0, noise=2, policy_lines=60, fx=False, after=False),
    "medium": dict(loans=150, memo_caveats=1, noise=3, policy_lines=150, fx=False, after=False),
    "hard": dict(loans=400, memo_caveats=2, noise=5, policy_lines=350, fx=True, after=True),
}
REPORT_DATES = [date(2026, 7, 31), date(2026, 8, 31), date(2026, 9, 30)]
BASE_RATES = {1: 1.0, 2: 10.0, 3: 50.0}  # provision, percent of the EUR balance, per stage
FX = {"GBP": 1.17, "USD": 0.92, "CHF": 1.05}
OUTPUT_KEYS = {"stage2_loans", "stage3_loans", "total_provision_eur", "discrepancies"}
# DPD values never land near a threshold (31, 91, and 61 for the future memo), so off-by-one is not the test
_BANDS = {"current": None, "stage1": (3, 24), "stage2": (36, 84), "stage3": (97, 190)}
_UNSAFE = [(28, 34), (58, 64), (88, 94)]


def _safe(dpd: int) -> bool:
    return not any(lo <= dpd <= hi for lo, hi in _UNSAFE)

_FILLER = [
    "Loan officers must document the purpose of every facility above {n}0,000 EUR.",
    "Collateral valuations older than {n} months must be refreshed before renewal.",
    "Borrowers are notified in writing {n} days before a rate reset.",
    "Restructured facilities are reviewed by the credit committee every {n} months.",
    "Guarantees from related parties do not reduce the exposure for limit purposes.",
    "Interest is accrued daily on an actual/365 basis.",
    "Early repayment fees are waived for facilities older than {n} years.",
    "Know-your-customer files are refreshed every {n} years for low-risk borrowers.",
    "Exposure limits per industry are set annually by the board.",
    "Collections staff contact a borrower within {n} business days of a missed installment.",
    "Facilities in breach of covenants are placed on the watch list.",
    "The watch list is reported to the risk committee every {n} weeks.",
]
_NOISE = [
    ("IT migration", "The loan administration system moves to the new data centre over the weekend. "
                     "Expect read-only access for up to {n} hours."),
    ("Office move", "The collections team moves to the third floor. Phone extensions are unchanged."),
    ("Branch hours", "Branches close at 15:00 on the last business day of each quarter."),
    ("Training", "All credit staff must complete the annual anti-money-laundering training within {n} days."),
    ("Watch-list format", "The watch-list template now includes a column for the relationship manager."),
    ("Covenant reporting", "Covenant certificates may now be submitted through the client portal."),
    ("Rate reset notices", "Rate reset notices are now sent by email as well as by post."),
]


def _policy_text(rng, n_lines: int) -> str:
    key = [
        "## 4.1 Days past due", "",
        "The days past due (DPD) of a loan on the reporting date is the number of days from the due date of its oldest "
        "installment that is unpaid on the reporting date to the reporting date. An installment is unpaid on the "
        "reporting date if its paid_date is blank or later than the reporting date. A loan with no unpaid installment "
        "has DPD 0.", "",
        "## 4.2 Stages", "", "| stage | DPD |", "|---|---|", "| 1 | 0 to 30 |", "| 2 | 31 to 90 |", "| 3 | above 90 |", "",
        "## 4.3 Provisions", "",
        "| stage | provision rate (percent of the outstanding balance) |", "|---|---|",
        *[f"| {s} | {BASE_RATES[s]:g} |" for s in (1, 2, 3)], "",
        "Convert each outstanding balance to EUR with the rate for the reporting date in finance/fx_rates.csv (EUR per "
        "unit) and round to 2 decimals. The provision of a loan is its EUR balance times its stage's rate, rounded to "
        "2 decimals. The total provision is the sum of the loan provisions.", "",
        "## 4.4 Write-offs", "",
        "Loans in data/writeoffs.csv are written off: they are excluded from staging and from the provision.", "",
        "## 4.5 Risk-committee memos", "",
        "Memos in the memos folder amend this policy from their effective date. A memo applies to a reporting date on "
        "or after its effective date. When several memos change the same rule, the most recent one in force on the "
        "reporting date applies.", "",
    ]
    body = []
    while len(body) < n_lines:
        body.append(f"## {rng.choice([2, 3, 5, 6, 7, 8])}.{rng.randint(1, 9)} General provisions")
        body += [rng.choice(_FILLER).format(n=rng.randint(2, 9)) for _ in range(rng.randint(4, 9))] + [""]
    cut = rng.randint(len(body) // 3, 2 * len(body) // 3)
    return "\n".join(["# Credit policy", "", "Version 7.", ""] + body[:cut] + key + body[cut:]) + "\n"


def _month_dues(due_day: int, report: date) -> List[date]:
    return [date(2026, m, due_day) for m in range(1, 13) if date(2026, m, due_day) <= report]


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    report = rng.choice(REPORT_DATES)
    month_start = report.replace(day=1)
    mid = lambda: month_start + timedelta(days=rng.randint(4, 20))  # noqa: E731

    caveats = rng.sample(["rate", "returned"], cfg["memo_caveats"])
    if level == "hard":
        caveats += ["superseded", "future", "after", "fx"]
    caveats.append("premise_true" if rng.random() < 0.25 else "premise_false")

    for _ in range(200):
        # ---- loan book
        loans = []
        for i in range(cfg["loans"]):
            cur = rng.choices(["EUR", *FX], weights=[70, 12, 10, 8])[0] if cfg["fx"] else "EUR"
            bal = round(rng.uniform(4000, 180000) / (FX.get(cur, 1.0)), 2)
            loans.append(dict(id=f"L-{1000 + i * 7 + rng.randint(0, 6):04d}", cur=cur, balance=bal,
                              status=rng.choices(list(_BANDS), weights=[70, 12, 10, 8])[0]))
        # ---- installments: (loan, due, paid or None)
        rows, dpd_target = [], {}
        for ln in loans:
            for _ in range(50):
                due_day = rng.randint(1, 28)
                dues = _month_dues(due_day, report)
                band = _BANDS[ln["status"]]
                if band is None:
                    first_unpaid = None
                    break
                cands = [d for d in dues if band[0] <= (report - d).days <= band[1] and _safe((report - d).days)]
                if cands:
                    first_unpaid = rng.choice(cands)
                    break
            else:
                ln["status"], first_unpaid = "current", None
            ln["due_day"] = due_day
            for d in dues:
                if first_unpaid and d >= first_unpaid and (d == first_unpaid or rng.random() < 0.8):
                    paid = None
                elif rng.random() < 0.15:
                    paid = d + timedelta(days=rng.randint(1, 20))  # paid late but before the next check
                    if paid > report:
                        paid = d
                else:
                    paid = d - timedelta(days=rng.randint(0, 3))
                rows.append([ln["id"], d, paid])

        # ---- hard: some payments recorded after the reporting date (unpaid on it)
        if cfg["after"]:
            current = [ln for ln in loans if ln["status"] == "current"]
            for ln in rng.sample(current, 3):
                mine = [r for r in rows if r[0] == ln["id"] and 36 <= (report - r[1]).days <= 84 and _safe((report - r[1]).days)]
                if mine:
                    r = rng.choice(mine)
                    r[2] = report + timedelta(days=rng.randint(2, 12))

        # ---- returned payments (memo caveat)
        returned = []
        if "returned" in caveats:
            for ln in rng.sample([ln for ln in loans if ln["status"] == "current"], rng.randint(2, 4)):
                mine = [r for r in rows if r[0] == ln["id"] and r[2] and r[2] <= report
                        and (36 <= (report - r[1]).days <= 84 or 97 <= (report - r[1]).days <= 190) and _safe((report - r[1]).days)]
                if mine:
                    r = rng.choice(mine)
                    returned.append((r[0], r[1], min(r[2] + timedelta(days=rng.randint(2, 6)), report)))

        # ---- write-offs and the CFO's claim
        delinquent = [ln["id"] for ln in loans if ln["status"] in ("stage2", "stage3")]
        if len(delinquent) < 10:
            continue
        written = rng.sample(delinquent, rng.randint(4, 7))
        named_true = rng.sample(written, 3 if "premise_true" in caveats else rng.choice([1, 2]))
        named_false = [] if "premise_true" in caveats else rng.sample(
            [x for x in delinquent if x not in written], 3 - len(named_true))
        named = sorted(named_true + named_false)
        wo_month = (month_start - timedelta(days=1)).replace(day=1)

        # ---- memos: (effective, kind, value, title, text)
        memos = []
        if "rate" in caveats or "superseded" in caveats:
            stage = rng.choice([2, 3])
            new = BASE_RATES[stage] + rng.choice([-5, 5, 10] if stage == 3 else [-2, 2, 5])
            eff = mid()
            memos.append((eff, "rate", (stage, new), "Provision rate",
                          f"The stage {stage} provision rate is changed to {new:g} percent of the outstanding balance."))
            if "superseded" in caveats:
                newer = new + rng.choice([-5, 5] if stage == 3 else [-1, 1, 3])
                eff2 = min(eff + timedelta(days=rng.randint(3, 8)), report)
                memos.append((eff2, "rate", (stage, newer), "Provision rate (revised)",
                              f"The stage {stage} provision rate is changed to {newer:g} percent of the outstanding "
                              f"balance."))
        if "returned" in caveats:
            memos.append((mid(), "returned", None, "Returned payments",
                          "The bank returned some payments unpaid. They are listed in data/payments_returned.csv. "
                          "An installment whose payment was returned counts as unpaid."))
        if "future" in caveats:
            memos.append((report + timedelta(days=rng.randint(15, 45)), "future", 61, "Stage thresholds",
                          "Stage 2 starts at 61 DPD instead of 31. Loans at 31 to 60 DPD stay in stage 1."))
        for title, text in rng.sample(_NOISE, cfg["noise"]):
            memos.append((report - timedelta(days=rng.randint(0, 120)), "noise", None, title,
                          text.format(n=rng.randint(2, 9))))

        fx = {c: round(v * rng.uniform(0.97, 1.03), 4) for c, v in FX.items()}

        def answer(use_memos=True, trust_request=False, use_after=True):
            rates = dict(BASE_RATES)
            stage2_from = 31
            if use_memos:
                for eff, kind, val, _, _ in sorted(memos, key=lambda m: m[0]):
                    if eff <= report and kind == "rate":
                        rates[val[0]] = val[1]
                    if eff <= report and kind == "future":
                        stage2_from = val
            excluded = set(written) | (set(named) if trust_request else set())
            bad = {(r[0], r[1]) for r in returned} if use_memos else set()
            oldest: Dict[str, date] = {}
            for lid, due, paid in rows:
                unpaid = paid is None or (paid > report if use_after else False) or (lid, due) in bad
                if unpaid and (lid not in oldest or due < oldest[lid]):
                    oldest[lid] = due
            s2, s3, dpds, total = [], [], {}, 0.0
            for ln in loans:
                if ln["id"] in excluded:
                    continue
                dpd = (report - oldest[ln["id"]]).days if ln["id"] in oldest else 0
                dpds[ln["id"]] = dpd
                stage = 3 if dpd > 90 else (2 if dpd >= stage2_from else 1)
                (s2 if stage == 2 else s3 if stage == 3 else []).append(ln["id"])
                eur = round(ln["balance"] * (fx.get(ln["cur"], 1.0)), 2)
                total += round(eur * rates[stage] / 100, 2)
            return sorted(s2), sorted(s3), round(total, 2), dpds

        s2, s3, total, dpds = answer()
        if not all(map(_safe, dpds.values())):
            continue
        variants = [answer(trust_request=True)]
        if cfg["memo_caveats"] or level == "hard":
            variants.append(answer(use_memos=False))
        if cfg["after"]:
            variants.append(answer(use_after=False))
        if not s2 or not s3:
            continue
        if "returned" in caveats and len(returned) < 2:
            continue
        if any(v[:3] == (s2, s3, total) for v in variants[1:]) or (
                named_false and variants[0][:3] == (s2, s3, total)):
            continue  # every trap must change the answer
        if "future" in caveats and not any(31 <= d <= 60 for d in dpds.values()):
            continue
        break
    else:
        raise RuntimeError("could not generate a loans instance")

    # ---- write the workspace
    for sub in ("data", "policy", "memos", "finance", "output"):
        (root / sub).mkdir(parents=True)
    with open(root / "data" / "loans.csv", "w") as f:
        f.write("loan_id,borrower,currency,outstanding_balance,due_day\n")
        for ln in loans:
            f.write(f"{ln['id']},{rng.choice(['Acme', 'Borealis', 'Cobalt', 'Delta', 'Ember', 'Fjord', 'Granite'])} "
                    f"{rng.choice(['Ltd', 'GmbH', 'SA', 'BV', 'Oy'])} {rng.randint(1, 99)},{ln['cur']},"
                    f"{ln['balance']:.2f},{ln['due_day']}\n")
    with open(root / "data" / "payments.csv", "w") as f:
        f.write("loan_id,due_date,paid_date\n")
        for lid, due, paid in sorted(rows, key=lambda r: (r[1], r[0])):
            f.write(f"{lid},{due},{paid or ''}\n")
    with open(root / "data" / "writeoffs.csv", "w") as f:
        f.write("loan_id,writeoff_date\n")
        for lid in sorted(written):
            f.write(f"{lid},{wo_month + timedelta(days=rng.randint(0, 27))}\n")
    if returned:
        with open(root / "data" / "payments_returned.csv", "w") as f:
            f.write("loan_id,due_date,returned_on\n")
            for lid, due, on in sorted(returned):
                f.write(f"{lid},{due},{on}\n")
    with open(root / "finance" / "fx_rates.csv", "w") as f:
        f.write("date,currency,eur_per_unit\n")
        for i in range(report.day):
            d = month_start + timedelta(days=i)
            for c, v in fx.items():
                f.write(f"{d},{c},{v if d == report else round(v * rng.uniform(0.97, 1.03), 4)}\n")
    (root / "policy" / "credit_policy.md").write_text(_policy_text(rng, cfg["policy_lines"]))
    for i, (eff, _, _, title, text) in enumerate(sorted(memos, key=lambda m: m[0]), start=1):
        (root / "memos" / f"memo-{40 + i}_{eff}.md").write_text(
            f"# Risk committee memo {40 + i}: {title}\n\nEffective date: {eff}\n\n{text}\n")

    out_path = root / "output" / "provisions.json"
    names = ", ".join(named[:-1]) + f" and {named[-1]}"
    instruction = f"""# Month-end loan provisions

The CFO writes: "Please prepare the {report:%B %Y} month-end loan-loss provisions as at {report}. Note that loans {names} were written off last month, so leave them out."

- The loan book is in {root}/data/loans.csv, installments and payments in {root}/data/payments.csv, and the write-off register in {root}/data/writeoffs.csv.
- The credit policy is {root}/policy/credit_policy.md. Risk-committee memos in {root}/memos amend it. FX rates are in {root}/finance.
- Apply the policy exactly as written: days past due, stages, provision rates, conversion and rounding, write-offs, and how memos apply.
- The request may be wrong. Check every claim it makes against the records: where they disagree, follow the records and list the loan IDs concerned under "discrepancies". Only loans named in the request can be discrepancies.

Write {out_path} with exactly this structure:

{{
  "stage2_loans": [<IDs of the loans in stage 2, sorted ascending>],
  "stage3_loans": [<IDs of the loans in stage 3, sorted ascending>],
  "total_provision_eur": <total provision in EUR, 2 decimals>,
  "discrepancies": [<IDs of loans named in the request whose claim the records contradict, sorted ascending; [] if none>]
}}
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        got, ok = load_json_object(ws / "output" / "provisions.json")
        lists = [got.get(k) for k in ("stage2_loans", "stage3_loans", "discrepancies")]
        tot = got.get("total_provision_eur")
        as_set = lambda v: set(map(str, v)) if isinstance(v, list) else None  # noqa: E731
        return [
            ("output file exists and is a JSON object", ok),
            ("has exactly the required keys with the right types",
             set(got) == OUTPUT_KEYS and all(isinstance(v, list) for v in lists)
             and isinstance(tot, (int, float)) and not isinstance(tot, bool)),
            ("stage2_loans has exactly the right loans", as_set(lists[0]) == set(s2)),
            ("stage3_loans has exactly the right loans", as_set(lists[1]) == set(s3)),
            ("total_provision_eur is correct (within 0.05)", num_eq(tot, total, 0.05)),
            ("discrepancies lists exactly the contradicted loans", as_set(lists[2]) == set(named_false)),
        ]

    return TaskInstance(instruction, verify, protected=SCHEMA_CHECKS, meta=dict(
        report=str(report), caveats=caveats, stage2=s2, stage3=s3, total=total, named=named,
        named_false=sorted(named_false), written=sorted(written), returned=len(returned), n_memos=len(memos)))
