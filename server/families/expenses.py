"""
Office family: expense audit.

The model gets thousands of expense claims in several currencies, daily FX
rates, a long travel policy (city tiers, per-category caps, receipt and flight
rules), dated policy amendments and a finance folder with FX corrections. It
must list the claims of one department and month that break policy and the
reimbursable EUR per employee, in /app/output/expense_report.json.

Caveats: a cap that changes mid-month (claims before the date keep the old cap,
and on hard a second amendment supersedes the first), a city that changes tier
mid-month, corrected FX rates, and on hard a future-dated rule to ignore.
"""

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from .common import Check, TaskInstance, load_json_object, num_eq

CITIES = {  # city: (currency, tier in the base policy)
    "Lisbon": ("EUR", "B"), "Madrid": ("EUR", "B"), "Berlin": ("EUR", "A"), "Paris": ("EUR", "A"),
    "London": ("GBP", "A"), "Manchester": ("GBP", "B"), "Zurich": ("CHF", "A"), "Geneva": ("CHF", "A"),
    "New York": ("USD", "A"), "Austin": ("USD", "B"), "Tokyo": ("JPY", "A"), "Osaka": ("JPY", "B"),
}
EUR_ONLY = ["Lisbon", "Madrid", "Berlin", "Paris"]
BASE_CAPS = {("meals", "A"): 60, ("meals", "B"): 40, ("lodging", "A"): 220, ("lodging", "B"): 140,
             ("taxi", "A"): 50, ("taxi", "B"): 50}
BASE_RATES = {"GBP": 1.17, "CHF": 1.05, "USD": 0.92, "JPY": 0.0062}
DEPARTMENTS = ["Sales", "Research", "Operations", "Finance", "Marketing"]
LEVELS = {
    "easy": dict(months=2, departments=2, employees=10, claims=260, eur_only=True, noise=1),
    "medium": dict(months=3, departments=4, employees=28, claims=1400, eur_only=False, noise=2),
    "hard": dict(months=3, departments=5, employees=40, claims=3000, eur_only=False, noise=3),
}
MONTHS = [(2026, 7), (2026, 8), (2026, 9)]

_FILLER = [
    "Travel must be booked through the company travel portal unless the portal has no availability.",
    "Economy class is the default for flights shorter than {n} hours.",
    "Claims must be submitted within {n} days of the end of the trip.",
    "Alcohol is not reimbursable unless it is part of an approved client dinner.",
    "Lost receipts may be replaced by a signed declaration only for amounts below the receipt threshold.",
    "Personal travel combined with a business trip is reimbursed only for the business portion.",
    "Mileage for private cars is reimbursed at the rate published by the finance team.",
    "Tips are reimbursable up to {n} percent of the bill where tipping is customary.",
    "Conference fees are booked as training, not as travel, and follow the training policy.",
    "Laundry is reimbursable for trips longer than {n} nights.",
    "Visa fees and vaccinations required for travel are reimbursed in full.",
    "Upgrades paid with personal loyalty points are not reimbursed.",
]
_NOISE = [
    "The travel portal now also supports rail bookings in Germany and France.",
    "Claims for client gifts move to the procurement system and are no longer part of travel expenses.",
    "Employees may book ride-sharing services under the taxi category.",
    "The deadline for submitting claims is shortened from 60 to 45 days.",
    "Mileage claims require the start and end address of each journey.",
]


def _policy_text(rng, receipt: int, n_lines: int) -> str:
    tier_a = ", ".join(sorted(c for c, (_, t) in CITIES.items() if t == "A"))
    tier_b = ", ".join(sorted(c for c, (_, t) in CITIES.items() if t == "B"))
    key = [
        "## 3.1 City tiers", "", f"Tier A cities: {tier_a}.", f"Tier B cities: {tier_b}.", "",
        "## 3.2 Caps per claim (EUR)", "", "| category | tier A | tier B |", "|---|---|---|",
        *[f"| {c} | {BASE_CAPS[(c, 'A')]} | {BASE_CAPS[(c, 'B')]} |" for c in ("meals", "lodging", "taxi")], "",
        "A claim above its cap breaks policy; it is reimbursed up to the cap.", "",
        "## 3.3 Receipts", "", f"Receipts are required for any claim above {receipt} EUR. "
        "A claim that needs a receipt and has none breaks policy and is reimbursed at 0.", "",
        "## 3.4 Flights", "", "Flights have no cap but need a pre-approval ID. A flight claim without one breaks "
        "policy and is reimbursed at 0.", "",
        "## 3.5 Currency", "", "Convert each claim to EUR with the rate for the claim's date in the finance team's "
        "daily rate table (EUR per unit of the claim currency), and round the converted amount to 2 decimals "
        "before applying any rule.", "",
        "## 3.6 Amendments", "", "Amendments apply to claims dated on or after their effective date. When several "
        "amendments change the same rule, the most recent one in force on the claim date applies.", "",
    ]
    body = []
    while len(body) < n_lines:
        body.append(f"## {rng.randint(4, 12)}.{rng.randint(1, 9)} General provisions")
        body += [rng.choice(_FILLER).format(n=rng.randint(3, 14)) for _ in range(rng.randint(4, 10))] + [""]
    cut = rng.randint(len(body) // 3, 2 * len(body) // 3)
    return "\n".join(["# Travel and expense policy", "", "Version 12.", ""] + body[:cut] + key + body[cut:]) + "\n"


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    months = MONTHS[3 - cfg["months"]:]
    first, last = date(*months[0], 1), (date(2026, 10, 1) - timedelta(days=1))
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    ty, tm = rng.choice(months[-2:]) if len(months) > 1 else months[0]
    mstart = date(ty, tm, 1)
    mend = date(ty + (tm == 12), tm % 12 + 1, 1) - timedelta(days=1)
    mid = lambda: mstart + timedelta(days=rng.randint(8, 20))  # noqa: E731

    depts = DEPARTMENTS[:cfg["departments"]]
    dept = rng.choice(depts)
    employees = {f"E{i:02d}": depts[i % len(depts)] for i in range(1, cfg["employees"] + 1)}
    cities = EUR_ONLY if cfg["eur_only"] else list(CITIES)
    receipt = rng.choice([25, 30, 40])

    # ---- rules: base values plus dated amendments
    caveats = []
    if level == "medium":
        caveats = rng.sample(["cap", "tier", "fx"], 2)
    elif level == "hard":
        caveats = ["cap", "cap_superseded", "tier", "fx", "future"]
    amendments = []  # (effective, kind, key, value, text)
    if "cap" in caveats:
        cat, tier = rng.choice([("meals", "A"), ("meals", "B"), ("lodging", "A"), ("lodging", "B")])
        new = BASE_CAPS[(cat, tier)] + rng.choice([-15, -10, 10, 15, 20])
        eff = mid()
        amendments.append((eff, "cap", (cat, tier), new,
                           f"The {cat} cap for tier {tier} cities is changed to {new} EUR."))
        if "cap_superseded" in caveats:
            newer = new + rng.choice([-10, 5, 10])
            amendments.append((min(eff + timedelta(days=rng.randint(3, 7)), mend), "cap", (cat, tier), newer,
                               f"The {cat} cap for tier {tier} cities is changed to {newer} EUR."))
    if "tier" in caveats:
        city = rng.choice([c for c in cities if c not in EUR_ONLY] or cities)
        new_tier = "B" if CITIES[city][1] == "A" else "A"
        amendments.append((mid(), "tier", city, new_tier, f"{city} is reclassified as a tier {new_tier} city."))
    if "future" in caveats:
        amendments.append((date(2026, 11, rng.randint(1, 28)), "receipt", None, receipt + 25,
                           f"Receipts are required for any claim above {receipt + 25} EUR."))
    for text in rng.sample(_NOISE, cfg["noise"]):
        amendments.append((rng.choice(days), "noise", None, None, text))

    def rule(kind, key, base, when):
        vals = [(e, v) for e, k, kk, v, _ in amendments if k == kind and kk == key and e <= when]
        return max(vals, key=lambda x: x[0])[1] if vals else base

    # ---- FX: published daily rates, plus corrections in the finance notes
    published = {(d, c): round(BASE_RATES[c] * rng.uniform(0.985, 1.015), 6 if c == "JPY" else 4)
                 for d in days for c in BASE_RATES}
    corrections = {}
    if "fx" in caveats:
        for _ in range(2 if level == "hard" else 1):
            c = rng.choice([cur for cur in BASE_RATES if any(CITIES[x][0] == cur for x in cities)])
            d = mstart + timedelta(days=rng.randint(2, 26))
            corrections[(d, c)] = round(published[(d, c)] * rng.choice([0.9, 0.93, 1.07, 1.1]), 6 if c == "JPY" else 4)
    noise_correction = (first + timedelta(days=3), "GBP")  # a correction outside the target month (distractor)
    corrections_all = {**corrections, noise_correction: round(published[noise_correction] * 1.05, 4)}

    def rate(d, cur, correct=True):
        if cur == "EUR":
            return 1.0
        return corrections_all.get((d, cur), published[(d, cur)]) if correct else published[(d, cur)]

    def evaluate(c, use_rules=True):
        """(violates, reimbursable EUR) for one claim under the full rules, or ignoring amendments/corrections."""
        d = c["date"]
        eur = round(c["amount"] * rate(d, c["currency"], use_rules), 2)
        r_thr = rule("receipt", None, receipt, d) if use_rules else receipt
        if c["category"] == "flights":
            ok = bool(c["approval_id"])
            return (not ok, eur if ok else 0.0, eur, None)
        tier = rule("tier", c["city"], CITIES[c["city"]][1], d) if use_rules else CITIES[c["city"]][1]
        cap = rule("cap", (c["category"], tier), BASE_CAPS[(c["category"], tier)], d) if use_rules \
            else BASE_CAPS[(c["category"], tier)]
        if eur > r_thr and not c["receipt"]:
            return (True, 0.0, eur, cap)
        return (eur > cap, min(eur, cap), eur, cap)

    # ---- claims: retry until the answer is non-trivial and the caveats change it
    for _ in range(100):
        claims = []
        for i in range(cfg["claims"]):
            while True:
                cat = rng.choices(["meals", "lodging", "taxi", "flights"], weights=[40, 25, 25, 10])[0]
                city = rng.choice(cities)
                cur = CITIES[city][0]
                d = rng.choice(days)
                base = 300 if cat == "flights" else BASE_CAPS[(cat, CITIES[city][1])]
                eur_target = base * rng.uniform(0.45, 1.45)
                amount = round(eur_target / BASE_RATES.get(cur, 1.0), 0 if cur == "JPY" else 2)
                c = dict(claim_id=f"C{i + 1:05d}", employee_id=rng.choice(list(employees)), date=d, city=city,
                         category=cat, amount=amount, currency=cur, receipt=rng.random() < 0.85,
                         approval_id=f"AP-{rng.randint(1000, 9999)}" if cat == "flights" and rng.random() < 0.8 else "")
                # keep every converted amount well away from any threshold it is compared with
                margins = []
                for use in (True, False):
                    _, _, eur, cap = evaluate(c, use)
                    margins += [abs(eur - cap)] if cap is not None else []
                    margins += [abs(eur - rule("receipt", None, receipt, d)), abs(eur - receipt)]
                if min(margins) > 0.15:
                    break
            claims.append(c)
        in_scope = [c for c in claims if employees[c["employee_id"]] == dept and mstart <= c["date"] <= mend]
        truth = {c["claim_id"]: evaluate(c) for c in in_scope}
        naive = {c["claim_id"]: evaluate(c, False) for c in in_scope}
        violations = sorted(k for k, v in truth.items() if v[0])
        by_emp: Dict[str, float] = {}
        for c in in_scope:
            by_emp[c["employee_id"]] = round(by_emp.get(c["employee_id"], 0.0) + truth[c["claim_id"]][1], 2)
        affected = [k for k in truth if truth[k][:2] != naive[k][:2]]
        if violations and len(in_scope) >= 20 and (not caveats or len(affected) >= 2):
            break
    total = round(sum(by_emp.values()), 2)

    # ---- write the workspace
    for sub in ("data", "policy/amendments", "finance", "output"):
        (root / sub).mkdir(parents=True)
    with open(root / "data" / "claims.csv", "w") as f:
        f.write("claim_id,employee_id,date,city,category,amount,currency,receipt,approval_id\n")
        for c in claims:
            amt = f"{c['amount']:.0f}" if c["currency"] == "JPY" else f"{c['amount']:.2f}"
            f.write(f"{c['claim_id']},{c['employee_id']},{c['date']},{c['city']},{c['category']},{amt},"
                    f"{c['currency']},{'yes' if c['receipt'] else 'no'},{c['approval_id']}\n")
    with open(root / "data" / "employees.csv", "w") as f:
        f.write("employee_id,name,department\n")
        for e, dp in employees.items():
            f.write(f"{e},{rng.choice(['Ana', 'Ben', 'Chen', 'Dara', 'Eli', 'Femi', 'Gus', 'Hana'])} "
                    f"{rng.choice(['Silva', 'Novak', 'Okafor', 'Lind', 'Mori', 'Rossi'])},{dp}\n")
    with open(root / "finance" / "fx_rates.csv", "w") as f:
        f.write("date,currency,eur_per_unit\n")
        for d in days:
            for cur in BASE_RATES:
                f.write(f"{d},{cur},{published[(d, cur)]}\n")
    lines = ["# FX rate corrections", "", "Rates in fx_rates.csv that were published wrongly. Use the corrected "
             "rate for these dates.", "", "| date | currency | corrected eur_per_unit |", "|---|---|---|"]
    lines += [f"| {d} | {cur} | {r} |" for (d, cur), r in sorted(corrections_all.items())]
    (root / "finance" / "fx_corrections.md").write_text("\n".join(lines) + "\n")
    (root / "finance" / "README.md").write_text("Finance team shared folder: daily FX rates and corrections.\n")
    (root / "policy" / "travel_policy.md").write_text(_policy_text(rng, receipt, 140 if level == "easy" else 380))
    for i, (eff, kind, _, _, text) in enumerate(sorted(amendments, key=lambda a: a[0]), start=1):
        (root / "policy" / "amendments" / f"amendment-{20 + i}_{eff}.md").write_text(
            f"# Policy amendment {20 + i}\n\nEffective date: {eff}\n\n{text}\n")

    out_path = root / "output" / "expense_report.json"
    instruction = f"""# Expense audit

Finance needs an audit of the {dept} department's expense claims dated in {mstart:%B %Y} ({mstart} to {mend}).

- Claims are in {root}/data/claims.csv; employees and departments in {root}/data/employees.csv.
- The travel and expense policy is in {root}/policy, including amendments. The finance team's folder {root}/finance has the daily FX rates and their notes.
- Apply the policy exactly as written: city tiers, caps, receipt and flight rules, currency conversion and rounding, and how amendments apply.

Write {out_path} with exactly this structure:

{{
  "violations": [<IDs of the department's claims in that month that break policy, sorted ascending>],
  "by_employee": {{"E01": <total reimbursable EUR for that employee's claims in scope, 2 decimals>, ...}},
  "reimbursable_eur": <total reimbursable EUR for all claims in scope, 2 decimals>
}}

Include in by_employee every employee of the department who has at least one claim in that month.
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        got, ok = load_json_object(ws / "output" / "expense_report.json")
        vio, be, tot = got.get("violations"), got.get("by_employee"), got.get("reimbursable_eur")
        return [
            ("output file exists and is a JSON object", ok),
            ("has exactly the required keys with the right types",
             set(got) == {"violations", "by_employee", "reimbursable_eur"} and isinstance(vio, list)
             and isinstance(be, dict) and isinstance(tot, (int, float))),
            ("violations has exactly the right claims", isinstance(vio, list) and set(map(str, vio)) == set(violations)),
            ("violations is sorted ascending", isinstance(vio, list) and len(vio) > 0 and vio == sorted(map(str, vio))),
            ("by_employee covers exactly the department's employees with claims in the month",
             isinstance(be, dict) and set(be) == set(by_emp)),
            ("every employee's reimbursable EUR is correct (within 0.02)",
             isinstance(be, dict) and set(be) == set(by_emp) and all(num_eq(be[e], v, 0.02) for e, v in by_emp.items())),
            ("reimbursable_eur total is correct (within 0.05)", num_eq(tot, total, 0.05)),
            ("reimbursable_eur equals the sum of by_employee (within 0.02)",
             isinstance(be, dict) and all(isinstance(v, (int, float)) for v in be.values())
             and num_eq(tot, sum(be.values()), 0.02)),
        ]

    return TaskInstance(instruction, verify, meta=dict(
        dept=dept, month=f"{mstart:%Y-%m}", caveats=caveats, violations=violations, by_employee=by_emp,
        total=total, n_in_scope=len(in_scope), n_amendments=len(amendments),
        affected=sum(1 for k in truth if truth[k][:2] != naive[k][:2])))
