"""
Software family: repair a small Python billing package so it matches its spec.

The workspace is a multi-module package (money, pricing, invoices, dates, fees,
units, reports) with injected bugs, a visible test suite that catches exactly
one of them, docs/SPEC.md and docs/CHANGELOG.md. The other bugs are found only
by reading the spec and the changelog, which can override the spec (hard level).
Hard also has a decoy: code that looks off-by-one but is what the spec asks for.

Grading runs hidden cases in a subprocess as the agent user and compares the
JSON results with expected values computed here; the expected values never
reach the sandbox, so code that inspects the harness still has to be correct.
"""

import json
import os
import random
import secrets
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from .common import Check, TaskInstance

LEVELS = {"easy": dict(bugs=1, changelog=False, decoy=False),
          "medium": dict(bugs=2, changelog=False, decoy=False),
          "hard": dict(bugs=3, changelog=True, decoy=True)}
PACKAGES = ["ledgerkit", "billcore", "invoicer", "accrual", "tallyho"]
FUNCS = ["round_div", "discount_cents", "tax_cents", "invoice_total", "days_between", "month_bucket",
         "late_fee", "parse_qty", "top_customers"]
MODULE_OF = {"round_div": "money", "discount_cents": "pricing", "tax_cents": "pricing", "invoice_total": "invoices",
             "days_between": "dates", "month_bucket": "dates", "late_fee": "fees", "parse_qty": "units",
             "top_customers": "reports"}


# --------------------------------------------------------------- source templates
# Each correct snippet has a buggy variant. Bugs are the kinds real code review finds:
# floor vs half-up, > vs >=, wrong scale, wrong order of operations, off-by-one,
# missing zero padding, floor vs ceil, case sensitivity, unstable tie-break.

def sources(c: dict, bugs: set) -> Dict[str, str]:
    b = lambda name: name in bugs  # noqa: E731
    money = f'''"""Money helpers. All amounts are integer cents."""


def round_div(numer: int, denom: int) -> int:
    """Divide two non-negative integers and round half up (SPEC 2.1)."""
{"    return numer // denom" if b("round_div") else "    q, r = divmod(numer, denom)" + chr(10) + "    return q + (1 if 2 * r >= denom else 0)"}
'''
    pricing = f'''"""Discounts and tax."""

from .money import round_div

DISCOUNT_THRESHOLD_CENTS = {c["threshold"]}


def discount_cents(subtotal: int, pct: int) -> int:
    """Percentage discount on the subtotal, only above the threshold (SPEC 3.1)."""
    if subtotal {">" if b("discount_cents") else ">="} DISCOUNT_THRESHOLD_CENTS:
        return round_div(subtotal * pct, 100)
    return 0


def tax_cents(amount: int, rate_bps: int) -> int:
    """Tax on an amount; the rate is in basis points (SPEC 3.2)."""
    return round_div(amount * rate_bps, {1000 if b("tax_cents") else 10000})
'''
    if b("invoice_total"):
        body = ('    after_discount = subtotal - discount_cents(subtotal, pct)\n'
                '    taxed = after_discount + tax_cents(after_discount, rate_bps)\n'
                '    return max(taxed - coupon_cents, 0)')
    else:
        body = ('    after_discount = subtotal - discount_cents(subtotal, pct)\n'
                '    after_coupon = max(after_discount - coupon_cents, 0)\n'
                '    return after_coupon + tax_cents(after_coupon, rate_bps)')
    invoices = f'''"""Invoice totals."""

from .pricing import discount_cents, tax_cents


def invoice_total(lines, pct: int, coupon_cents: int, rate_bps: int) -> int:
    """lines: list of (quantity, unit_price_cents). See SPEC 4.1 for the order of operations."""
    subtotal = sum(q * p for q, p in lines)
{body}
'''
    dates = f'''"""Date helpers. Dates are ISO strings (YYYY-MM-DD)."""

from datetime import date


def days_between(start: str, end: str) -> int:
    """Number of days from start to end (SPEC 5.1)."""
    return (date.fromisoformat(end) - date.fromisoformat(start)).days{"" if b("days_between") else " + 1"}


def month_bucket(day: str) -> str:
    """Reporting month of a date (SPEC 5.2)."""
    d = date.fromisoformat(day)
    return {'f"{d.year}-{d.month}"' if b("month_bucket") else 'f"{d.year:04d}-{d.month:02d}"'}
'''
    fees = f'''"""Late fees."""

from .money import round_div

LATE_FEE_BPS = {c["late_bps"]}
PERIOD_DAYS = {c["period"]}


def late_fee(amount_cents: int, days_late: int) -> int:
    """Late fee for an overdue amount (SPEC 6.1)."""
    if days_late <= 0:
        return 0
    periods = {"days_late // PERIOD_DAYS" if b("late_fee") else "-(-days_late // PERIOD_DAYS)"}
    return round_div(amount_cents * LATE_FEE_BPS * periods, 10000)
'''
    units = f'''"""Quantity parsing."""

UNITS = {json.dumps(c["units"])}


def parse_qty(text: str) -> int:
    """Parse '3 dozen' style quantities into pieces (SPEC 7.1)."""
    number, unit = text.split()
    return int(number) * UNITS[unit{"" if b("parse_qty") else ".lower()"}]
'''
    key = "lambda kv: -kv[1]" if b("top_customers") else "lambda kv: (-kv[1], kv[0])"
    reports = f'''"""Customer reports."""


def top_customers(invoices, n: int) -> list:
    """invoices: list of (customer, total_cents). See SPEC 8.1."""
    totals = {{}}
    for customer, total in invoices:
        totals[customer] = totals.get(customer, 0) + total
    ranked = sorted(totals.items(), key={key})
    return [customer for customer, _ in ranked[:n]]
'''
    return {"money": money, "pricing": pricing, "invoices": invoices, "dates": dates, "fees": fees,
            "units": units, "reports": reports}


# --------------------------------------------------------------- reference behaviour


def ref(name: str, c: dict, args, bugs=frozenset()):
    """Documented behaviour, or the behaviour of the buggy package when `bugs` is given."""
    def round_div(n, d):
        if "round_div" in bugs:
            return n // d
        q, r = divmod(n, d)
        return q + (1 if 2 * r >= d else 0)

    def discount(sub, pct):
        hit = sub > c["threshold"] if "discount_cents" in bugs else sub >= c["threshold"]
        return round_div(sub * pct, 100) if hit else 0

    def tax(a, bps):
        return round_div(a * bps, 1000 if "tax_cents" in bugs else 10000)

    if name == "round_div":
        return round_div(*args)
    if name == "discount_cents":
        return discount(*args)
    if name == "tax_cents":
        return tax(*args)
    if name == "invoice_total":
        lines, pct, coupon, bps = args
        sub = sum(q * p for q, p in lines)
        after = sub - discount(sub, pct)
        if "invoice_total" in bugs:
            return max(after + tax(after, bps) - coupon, 0)
        after = max(after - coupon, 0)
        return after + tax(after, bps)
    if name == "days_between":
        return (date.fromisoformat(args[1]) - date.fromisoformat(args[0])).days + (0 if "days_between" in bugs else 1)
    if name == "month_bucket":
        d = date.fromisoformat(args[0])
        return f"{d.year}-{d.month}" if "month_bucket" in bugs else f"{d.year:04d}-{d.month:02d}"
    if name == "late_fee":
        amt, days = args
        if days <= 0:
            return 0
        periods = days // c["period"] if "late_fee" in bugs else -(-days // c["period"])
        return round_div(amt * c["late_bps"] * periods, 10000)
    if name == "parse_qty":
        n, u = args[0].split()
        units = c["units"]
        if "parse_qty" in bugs and u not in units:
            return "error: KeyError"
        return int(n) * units[u if "parse_qty" in bugs else u.lower()]
    if name == "top_customers":
        inv, n = args
        tot = {}
        for cu, t in inv:
            tot[cu] = tot.get(cu, 0) + t
        key = (lambda kv: -kv[1]) if "top_customers" in bugs else (lambda kv: (-kv[1], kv[0]))
        return [cu for cu, _ in sorted(tot.items(), key=key)[:n]]
    raise KeyError(name)


def cases(rng: random.Random, c: dict) -> Dict[str, list]:
    """Hidden inputs per function, including the edge cases each bug breaks."""
    thr = c["threshold"]
    names = ["Acme", "Birch", "Cobalt", "Delta", "Ember", "Fjord"]
    d0 = date(2026, 1, 1)
    out = {
        "round_div": [(d * k + r, d) for d, k, r in [(100, 3, 50), (100, 2, 49), (7, 5, 4), (7, 5, 3), (10000, 9, 5000)]]
        + [(rng.randint(0, 10 ** 6), rng.randint(1, 999)) for _ in range(12)],
        "discount_cents": [(thr, 10), (thr - 1, 10), (thr + 1, 15)] + [(rng.randint(thr // 2, thr * 2), rng.randint(1, 30)) for _ in range(10)],
        "tax_cents": [(rng.randint(100, 10 ** 6), rng.choice([700, 825, 1950, 2100])) for _ in range(10)],
        "invoice_total": [([(2, thr)], 10, 1500, 825), ([(1, thr // 2)], 0, 500, 2000)]
        + [([(rng.randint(1, 5), rng.randint(200, thr)) for _ in range(rng.randint(1, 4))],
                           rng.randint(0, 20), rng.choice([0, 500, 1500]), rng.choice([0, 825, 2000])) for _ in range(12)],
        "days_between": [(str(d0 + timedelta(days=a)), str(d0 + timedelta(days=a + b)))
                         for a, b in [(0, 0), (3, 10), (40, 31)] + [(rng.randint(0, 300), rng.randint(0, 90)) for _ in range(6)]],
        "month_bucket": [(str(d0 + timedelta(days=rng.randint(0, 364))),) for _ in range(8)] + [("2026-03-09",), ("2026-11-30",)],
        "late_fee": [(rng.randint(1000, 10 ** 6), days) for days in (0, 1, c["period"], c["period"] + 1, 2 * c["period"], 3, 95)],
        "parse_qty": [(f"{rng.randint(1, 40)} {u}",) for u in list(c["units"]) + [u.capitalize() for u in c["units"]] + [u.upper() for u in c["units"]]],
        "top_customers": [([(rng.choice(names), t) for t in (500, 500, 300, 300, 900)], 3),
                          ([("Ember", 100), ("Birch", 100), ("Acme", 100), ("Delta", 50)], 2)]
        + [([(rng.choice(names), rng.randint(1, 50) * 100) for _ in range(12)], rng.randint(1, 4)) for _ in range(4)],
    }
    return {k: [list(a) for a in v] for k, v in out.items()}


# --------------------------------------------------------------- docs


def spec_text(c: dict, changelog_threshold: bool) -> str:
    rule = ">" if changelog_threshold else ">="
    word = "strictly above" if changelog_threshold else "at or above"
    return f"""# {c['pkg']} specification

All money amounts are integer cents. Functions must not mutate their arguments.

## 2. Money
2.1 `round_div(numer, denom)` divides and rounds half up: 249/100 -> 2, 250/100 -> 3.

## 3. Pricing
3.1 `discount_cents(subtotal, pct)` returns pct percent of the subtotal (rounded with round_div) when the subtotal is {word} DISCOUNT_THRESHOLD_CENTS ({c['threshold']}), i.e. `subtotal {rule} threshold`; otherwise 0.
3.2 `tax_cents(amount, rate_bps)` returns amount * rate_bps / 10000 rounded with round_div (825 bps = 8.25%).

## 4. Invoices
4.1 `invoice_total(lines, pct, coupon_cents, rate_bps)`: subtotal = sum(quantity * unit price); subtract the percentage discount; then subtract the coupon (never below 0); then add tax on the remaining amount.

## 5. Dates
5.1 `days_between(start, end)` counts calendar days **inclusive of both ends**: the same day gives 1.
5.2 `month_bucket(day)` returns the reporting month as zero-padded `YYYY-MM` (e.g. 2026-03).

## 6. Fees
6.1 `late_fee(amount_cents, days_late)` charges LATE_FEE_BPS ({c['late_bps']} bps) of the amount for every started period of {c['period']} days overdue (1 day late = one period); 0 if not overdue.

## 7. Units
7.1 `parse_qty(text)` parses "<integer> <unit>" with units {', '.join(c['units'])}; unit names are case-insensitive.

## 8. Reports
8.1 `top_customers(invoices, n)` sums totals per customer and returns the n customers with the highest totals; ties are broken by customer name ascending.
"""


def changelog_text(c: dict, threshold_change: bool) -> str:
    lines = [f"# {c['pkg']} changelog", "",
             "## 2.4.0", "- Reports: tie-breaking documented in SPEC 8.1.", "- Internal: type hints.", ""]
    if threshold_change:
        lines[2:2] = ["## 2.5.0 (latest)",
                      "- **Behaviour change:** the discount threshold is now inclusive: a subtotal exactly equal to "
                      "DISCOUNT_THRESHOLD_CENTS gets the discount. This supersedes SPEC 3.1.", ""]
    lines += ["## 2.3.1", "- Fixed: units are matched case-insensitively.", "",
              "## 2.0.0", "- Money moved to integer cents everywhere.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------- generator


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    c = dict(pkg=rng.choice(PACKAGES), threshold=rng.choice([5000, 7500, 10000, 12500]),
             late_bps=rng.choice([100, 150, 200, 250]), period=rng.choice([14, 30]),
             units={"pcs": 1, "pair": 2, "dozen": 12, rng.choice(["case", "box", "crate"]): rng.choice([6, 24, 48])})
    decoy = "days_between" if cfg["decoy"] else None
    pool = [f for f in FUNCS if f != decoy]
    if cfg["changelog"]:  # the threshold bug matches the (outdated) spec; only the changelog reveals it
        bugs = ["discount_cents"] + rng.sample([f for f in pool if f != "discount_cents"], cfg["bugs"] - 1)
    else:
        bugs = rng.sample(pool, cfg["bugs"])
    visible = rng.choice([f for f in bugs if f != "discount_cents" or not cfg["changelog"]])
    hidden = cases(rng, c)
    expected = {f: [ref(f, c, a) for a in hidden[f]] for f in FUNCS}
    buggy = {f: [ref(f, c, a, frozenset(bugs)) for a in hidden[f]] for f in FUNCS}
    for f in bugs:  # each injected bug must be visible to the hidden cases
        assert buggy[f] != expected[f], f"hidden cases do not detect the {f} bug"

    proj = root / "project"
    pkg = proj / "src" / c["pkg"]
    for d in (pkg, proj / "tests", proj / "docs", proj / "scripts", root / "output"):
        d.mkdir(parents=True, exist_ok=True)
    for mod, src in sources(c, set(bugs)).items():
        (pkg / f"{mod}.py").write_text(src)
    (pkg / "__init__.py").write_text(f'"""{c["pkg"]}: billing helpers."""\n__version__ = "2.5.0"\n')
    (proj / "docs" / "SPEC.md").write_text(spec_text(c, cfg["changelog"]))
    (proj / "docs" / "CHANGELOG.md").write_text(changelog_text(c, cfg["changelog"]))
    (proj / "README.md").write_text(f"# {c['pkg']}\n\nBilling helpers used by the invoicing service.\n\n"
                                    f"Run the tests: `cd {proj} && python3 -m pytest -q`\n\nSee docs/SPEC.md and docs/CHANGELOG.md.\n")
    (proj / "pytest.ini").write_text("[pytest]\npythonpath = src\n")
    (proj / "scripts" / "export_invoices.py").write_text("# one-off export script kept for reference; not part of the package\n"
                                                         "import csv, sys\nprint('usage: export_invoices.py OUT.csv')\n")

    # visible tests: a few that pass on the buggy package, plus ones that expose the reported bug
    vis_cases = {f: [a for a, w, g in zip(hidden[f], expected[f], buggy[f]) if w == g][:2] for f in FUNCS if f not in bugs}
    vis_cases = {f: cs for f, cs in vis_cases.items() if cs}
    reveal = [a for a, w, g in zip(hidden[visible], expected[visible], buggy[visible]) if w != g][:2]
    test_lines = ["import pytest", "", f"from {c['pkg']} import money, pricing, invoices, dates, fees, units, reports", "", ""]
    for f, cs in list(vis_cases.items())[:4] + [(visible, reveal)]:
        mod = MODULE_OF[f]
        for i, a in enumerate(cs):
            want = ref(f, c, a)
            test_lines += [f"def test_{f}_{i}():", f"    assert {mod}.{f}(*{a!r}) == {want!r}", "", ""]
    (proj / "tests" / "test_basic.py").write_text("\n".join(test_lines))

    instruction = f"""# Fix the {c['pkg']} package

Customers report wrong amounts from the invoicing service. The cause is in the Python package at {proj}/src/{c['pkg']}.

- The required behaviour is in {proj}/docs/SPEC.md and {proj}/docs/CHANGELOG.md (the latest changelog entry wins where they differ).
- The visible tests in {proj}/tests cover only part of the behaviour. Run them with `cd {proj} && python3 -m pytest -q`.
- Fix every function that does not match the documented behaviour. Do not change function names, signatures or module layout, and do not break functions that are already correct.

Hidden tests will check every function in the package against the documentation. Call submit when you are done.
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        got = _run_hidden(ws / "project" / "src", c["pkg"], hidden)
        ok = {f: got.get(f) == expected[f] for f in FUNCS}
        checks: List[Check] = [("package imports and every function runs", got.get("_import_ok") is True)]
        checks += [(f"fixed: {f} matches the documentation", ok[f]) for f in bugs]
        others = [f for f in FUNCS if f not in bugs and f != decoy]
        checks.append(("no regressions in the functions that were already correct", all(ok[f] for f in others)))
        if decoy:
            checks.append((f"{decoy} kept as specified (inclusive count)", ok[decoy]))
        return checks

    protected = ("package imports and every function runs", "no regressions in the functions that were already correct")
    if decoy:
        protected += (f"{decoy} kept as specified (inclusive count)",)
    return TaskInstance(instruction, verify, protected=protected, editable=("project/src/*",), meta=dict(pkg=c["pkg"], bugs=bugs, visible=visible, decoy=decoy, c=c,
                                                       expected=expected, hidden=hidden))


# --------------------------------------------------------------- hidden test harness


_HARNESS = r'''
import json, sys, importlib
sys.path.insert(0, SRC)
out = {"_import_ok": True}
mods = {}
try:
    for m in ("money", "pricing", "invoices", "dates", "fees", "units", "reports"):
        mods[m] = importlib.import_module(PKG + "." + m)
except Exception as e:
    out["_import_ok"] = False
for f, (mod, cases) in CASES.items():
    res = []
    for args in cases:
        try:
            v = getattr(mods[mod], f)(*args)
            res.append(v if isinstance(v, (int, str, list)) and not isinstance(v, bool) else repr(v))
        except Exception as e:
            res.append("error: " + type(e).__name__)
    out[f] = res
sys.stdout.write(NONCE + json.dumps(out, default=repr) + NONCE)
'''


def _run_hidden(src: Path, pkg: str, hidden: dict) -> dict:
    """Run every hidden case in a fresh interpreter as the agent; return {function: [results]}."""
    try:
        from ..sandbox import _agent_identity, _limits
    except ImportError:
        from server.sandbox import _agent_identity, _limits
    nonce = secrets.token_hex(16)
    payload = {f: (MODULE_OF[f], cs) for f, cs in hidden.items()}
    script = f"SRC = {str(src)!r}\nPKG = {pkg!r}\nNONCE = {nonce!r}\nCASES = {json.dumps(payload)}\n" + _HARNESS
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    os.chmod(path, 0o644)
    kwargs = dict(capture_output=True, timeout=60, cwd=str(src), env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
    ident = _agent_identity()
    if ident:
        kwargs.update(user=ident[0], group=ident[1], extra_groups=[], preexec_fn=_limits)
    try:
        p = subprocess.run(["python3", "-I", path], **kwargs)
        parts = p.stdout.decode("utf-8", "replace").split(nonce)
        return json.loads(parts[-2]) if len(parts) >= 3 else {}
    except Exception:
        return {}
    finally:
        os.unlink(path)
