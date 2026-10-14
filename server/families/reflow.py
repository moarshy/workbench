"""
Industrial family: reflow soldering audit (short handbook + thermocouple data -> exact JSON).

Modelled on the "handbook + sensor data + exact report" shape of SkillsBench's
manufacturing tasks, kept small: a 40-120 line handbook, a few hundred rows of
thermocouple data, and one output file. The traps are the ones such tasks use:
definitions that live only in the handbook, a revision that supersedes a limit,
a run logged in Fahrenheit, thermocouple selection rules, linear interpolation at
threshold crossings, missing data -> null, sorting and rounding.

Levels
  easy    max preheat ramp rate per run (+ which runs violate the ramp limit)
  hard    ramp + time above liquidus + peak temperature per run, compliance status,
          with a handbook revision and a Fahrenheit run and a run with no data
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

from .common import Check, TaskInstance, load_json_object, num_eq

LEVELS = {"easy": dict(runs=4, revision=False, fahrenheit=False, missing=False, report="ramp"),
          "hard": dict(runs=6, revision=True, fahrenheit=True, missing=True, report="full")}
FILLER = [
    "Operators must wear ESD wrist straps when handling boards.",
    "Clean the conveyor rails weekly with isopropyl alcohol.",
    "Nitrogen flow is set by process engineering and must not be changed on the line.",
    "Log every profile change in the line book with the board family and the recipe ID.",
    "Thermocouples are attached with high-temperature tape or solder, never with adhesive alone.",
    "The oven must be stable for 20 minutes after a recipe change before production boards run.",
    "Board families are listed in the MES; use the family code printed on the panel.",
]


def _profile(rng, peak: float, liquidus: float, dt: int, ramp_rate: float) -> List[float]:
    """Piecewise reflow curve sampled every dt s: preheat ramp, soak, reflow rise, peak hold, cool-down."""
    temps, t = [], 25.0 + rng.uniform(-2, 2)
    soak_from, soak_to = rng.uniform(140, 160), rng.uniform(170, 185)
    while t < soak_from:                                       # preheat at about ramp_rate C/s
        t += ramp_rate * dt * rng.uniform(0.85, 1.05)
        temps.append(t)
    while t < soak_to:                                         # soak: slow rise
        t += rng.uniform(0.2, 0.5) * dt
        temps.append(t)
    while t < peak:                                            # reflow rise
        t += rng.uniform(1.0, 1.8) * dt
        temps.append(min(t, peak))
    for _ in range(max(1, int(rng.uniform(12, 40) / dt))):   # hold near the peak
        temps.append(peak - rng.uniform(0, 1.5))
    t = temps[-1]
    while t > 80:                                              # cool-down
        t -= rng.uniform(1.8, 3.2) * dt
        temps.append(t)
    return [round(v, 2) for v in temps]


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    alloy = rng.choice([("SAC305", 217.0), ("Sn63Pb37", 183.0), ("SN100C", 227.0)])
    liquidus = alloy[1]
    preheat_lo, preheat_hi = rng.choice([(50, 150), (60, 160), (40, 140)])
    ramp_limit = rng.choice([2.0, 2.5, 3.0])
    tal_min, tal_max = rng.choice([(45, 90), (30, 75), (60, 120)])
    peak_min = liquidus + rng.choice([15, 20, 25])
    rev_ramp = round(ramp_limit - 0.5, 1) if cfg["revision"] else None
    tc_rule = rng.choice(["coldest", "the thermocouple listed as 'reference' in boards.csv"])
    dt = rng.choice([1, 2])
    run_ids = [f"R{100 + i:03d}" for i in range(cfg["runs"])]
    fahrenheit = rng.choice(run_ids) if cfg["fahrenheit"] else None
    missing = rng.choice([r for r in run_ids if r != fahrenheit]) if cfg["missing"] else None

    # ---- data (regenerated until the instance is non-trivial: some runs pass and some fail)
    for _attempt in range(100):
        data, refs, truth = _make_data(rng, run_ids, missing, peak_min, liquidus, dt, cfg, preheat_lo, preheat_hi,
                                       rev_ramp if cfg["revision"] else ramp_limit, tc_rule, tal_min, tal_max)
        bad = [r for r in run_ids if truth[r] is None or not _ok(truth[r], rev_ramp if cfg["revision"] else ramp_limit,
                                                                  tal_min, tal_max, peak_min, cfg)]
        if 0 < len(bad) < len(run_ids):
            break
    limit = rev_ramp if cfg["revision"] else ramp_limit

    # ---- workspace
    (root / "data").mkdir(parents=True)
    (root / "output").mkdir()
    _write_data(root, rng, data, refs, run_ids, fahrenheit, missing, dt, alloy, tc_rule)
    _write_docs(root, rng, alloy, liquidus, preheat_lo, preheat_hi, ramp_limit, tal_min, tal_max, peak_min,
                tc_rule, cfg, rev_ramp)
    return _finish(root, cfg, run_ids, truth, limit, tal_min, tal_max, peak_min, liquidus, tc_rule, fahrenheit,
                   missing, dt, preheat_lo, preheat_hi)


def _ok(t, limit, tal_min, tal_max, peak_min, cfg):
    if cfg["report"] == "ramp":
        return t["ramp"] <= limit
    return t["ramp"] <= limit and tal_min <= t["tal"] <= tal_max and t["peak"] >= peak_min


def _make_data(rng, run_ids, missing, peak_min, liquidus, dt, cfg, preheat_lo, preheat_hi, limit, tc_rule, tal_min, tal_max):
    data: Dict[str, Dict[str, List[float]]] = {}
    refs = {}
    for r in run_ids:
        if r == missing:
            continue
        tcs = {}
        run_rate = limit * rng.uniform(0.75, 1.3)  # each run's preheat rate sits near the limit
        for tc in ("TC1", "TC2", "TC3"):
            tcs[tc] = _profile(rng, peak_min + rng.uniform(-8, 14), liquidus, dt, run_rate * rng.uniform(0.95, 1.05))
        data[r] = tcs
        refs[r] = rng.choice(["TC1", "TC2", "TC3"])

    def ramp(series):
        best = None
        for a, b in zip(series, series[1:]):
            if preheat_lo <= a <= preheat_hi and preheat_lo <= b <= preheat_hi:
                v = (b - a) / dt
                best = v if best is None else max(best, v)
        return best

    def tal(series):
        """Seconds above liquidus with linear interpolation at the crossings."""
        total = 0.0
        for a, b in zip(series, series[1:]):
            if a >= liquidus and b >= liquidus:
                total += dt
            elif a < liquidus <= b or b < liquidus <= a:
                frac = (liquidus - a) / (b - a)
                total += dt * ((1 - frac) if b >= liquidus else frac)
        return total

    def chosen_tc(r):
        if tc_rule == "coldest":
            return min(data[r], key=lambda tc: (max(data[r][tc]), tc))
        return refs[r]

    truth = {}
    for r in run_ids:
        if r not in data:
            truth[r] = None
            continue
        tc = chosen_tc(r)
        s = data[r][tc]
        rp = ramp(s)
        truth[r] = dict(tc=tc, ramp=round(rp if rp is not None else 0.0, 2), tal=round(tal(s), 2), peak=round(max(s), 2))
    return data, refs, truth


def _write_data(root, rng, data, refs, run_ids, fahrenheit, missing, dt, alloy, tc_rule):
    with open(root / "data" / "thermocouples.csv", "w") as f:
        f.write("run_id,tc_id,t_s,temp\n")
        for r, tcs in data.items():
            for tc, s in tcs.items():
                for i, v in enumerate(s):
                    temp = v * 9 / 5 + 32 if r == fahrenheit else v
                    f.write(f"{r},{tc},{i * dt},{temp:.2f}\n")
    with open(root / "data" / "runs.csv", "w") as f:
        f.write("run_id,board_family,alloy,temp_unit,notes\n")
        for r in run_ids:
            unit = "F" if r == fahrenheit else "C"
            note = "thermocouple logger failed; no data" if r == missing else ""
            f.write(f"{r},BF-{rng.randint(10, 99)},{alloy[0]},{unit},{note}\n")
    if tc_rule != "coldest":
        with open(root / "data" / "boards.csv", "w") as f:
            f.write("run_id,tc_id,role\n")
            for r in data:
                for tc in ("TC1", "TC2", "TC3"):
                    f.write(f"{r},{tc},{'reference' if tc == refs[r] else 'auxiliary'}\n")


def _write_docs(root, rng, alloy, liquidus, preheat_lo, preheat_hi, ramp_limit, tal_min, tal_max, peak_min,
                tc_rule, cfg, rev_ramp):
    key = [
        "## 4. Profile definitions",
        f"- Preheat region: board temperature between {preheat_lo} and {preheat_hi} degrees C (both samples of a step inside the region).",
        f"- Ramp rate: temperature change between consecutive samples divided by the sampling interval, in C/s. The run's ramp rate is the maximum over the preheat region.",
        f"- Liquidus of {alloy[0]}: {liquidus:g} C. Time above liquidus (TAL): seconds spent at or above liquidus; at a crossing, interpolate linearly between the two samples.",
        f"- Peak temperature: the maximum sample of the run.",
        f"- Which thermocouple to use: {'the coldest one, i.e. the thermocouple with the lowest peak temperature' if tc_rule == 'coldest' else tc_rule}.",
        "",
        "## 5. Limits",
        f"- Maximum preheat ramp rate: {ramp_limit:g} C/s.",
        f"- TAL must be between {tal_min} and {tal_max} s (inclusive).",
        f"- Peak temperature must be at least {peak_min:g} C.",
        "- A run with no thermocouple data cannot be verified and is treated as failing.",
    ]
    body = ["# Reflow oven process handbook (line 3)", "", "## 1. General"] + rng.sample(FILLER, 4) + [""] + key
    body += ["", "## 6. Maintenance"] + rng.sample(FILLER, 3) + ["", "## 7. Units", "All temperatures in this handbook are degrees Celsius. Some loggers record Fahrenheit; see runs.csv."]
    (root / "docs").mkdir()
    (root / "docs" / "handbook.md").write_text("\n".join(body) + "\n")
    if cfg["revision"]:
        (root / "docs" / "handbook_revision_B.md").write_text(
            f"# Handbook revision B (supersedes revision A where they differ)\n\n"
            f"- Section 5: the maximum preheat ramp rate is lowered to {rev_ramp:g} C/s.\n"
            f"- Section 1: hearing protection is no longer required next to the wave solder machine.\n")


def _finish(root, cfg, run_ids, truth, limit, tal_min, tal_max, peak_min, liquidus, tc_rule, fahrenheit, missing,
            dt, preheat_lo, preheat_hi):
    out = root / "output" / "reflow_report.json"
    if cfg["report"] == "ramp":
        schema = ('{\n  "ramp_limit_c_per_s": <the limit that applies>,\n  "max_preheat_ramp": {"R100": <C/s, 2 decimals>, ...},\n'
                  '  "violating_runs": [<run IDs whose ramp exceeds the limit, sorted>]\n}')
    else:
        schema = ('{\n  "ramp_limit_c_per_s": <the limit that applies>,\n  "runs": {\n    "R100": {"tc_id": "TC2", "max_preheat_ramp": <2 dp>, '
                  '"tal_s": <2 dp>, "peak_c": <2 dp>, "status": "compliant" or "non-compliant"},\n    ...\n  },\n'
                  '  "failing_runs": [<run IDs that are non-compliant, sorted>]\n}\nUse null for every value of a run that has no data.')
    instruction = f"""# Reflow profile audit

Process engineering needs an audit of today's reflow oven runs on line 3.

- The process handbook is in {root}/docs (read every document there; definitions and limits come only from the handbook).
- Thermocouple readings are in {root}/data/thermocouples.csv; run details are in {root}/data/runs.csv.
- Use the thermocouple the handbook prescribes for each run. Report temperatures in degrees C.

Write {out} with exactly this structure:

{schema}

Round to 2 decimals. Do not guess values: derive everything from the files.
"""
    (root / "TASK.md").write_text(instruction)

    def compliant(t):
        if t is None:
            return False
        return t["ramp"] <= limit and tal_min <= t["tal"] <= tal_max and t["peak"] >= peak_min

    def verify(ws: Path) -> List[Check]:
        got, ok = load_json_object(ws / "output" / "reflow_report.json")
        checks: List[Check] = [("output file exists and is a JSON object", ok),
                               ("ramp_limit_c_per_s is the limit in force", num_eq(got.get("ramp_limit_c_per_s"), limit, 1e-6))]
        if cfg["report"] == "ramp":
            mr = got.get("max_preheat_ramp")
            vr = got.get("violating_runs")
            want = sorted(r for r in run_ids if truth[r] and truth[r]["ramp"] > limit)
            checks += [
                ("max_preheat_ramp covers exactly the runs", isinstance(mr, dict) and set(mr) == set(run_ids)),
                ("every run's max preheat ramp is correct (within 0.01)",
                 isinstance(mr, dict) and all(num_eq(mr.get(r), truth[r]["ramp"], 0.011) for r in run_ids)),
                ("violating_runs is exactly right and sorted", isinstance(vr, list) and vr == want),
            ]
        else:
            runs = got.get("runs") if isinstance(got.get("runs"), dict) else {}
            fr = got.get("failing_runs")
            have = [r for r in run_ids if truth[r]]
            checks += [
                ("runs covers exactly the run IDs", set(runs) == set(run_ids)),
                ("the prescribed thermocouple is used for every run",
                 all(isinstance(runs.get(r), dict) and runs[r].get("tc_id") == truth[r]["tc"] for r in have)),
                ("ramp rates correct (within 0.01)", all(isinstance(runs.get(r), dict) and num_eq(runs[r].get("max_preheat_ramp"), truth[r]["ramp"], 0.011) for r in have)),
                ("TAL correct (within 0.05 s)", all(isinstance(runs.get(r), dict) and num_eq(runs[r].get("tal_s"), truth[r]["tal"], 0.051) for r in have)),
                ("peak temperatures correct in C (within 0.01)", all(isinstance(runs.get(r), dict) and num_eq(runs[r].get("peak_c"), truth[r]["peak"], 0.011) for r in have)),
                ("the run without data has null values", all(isinstance(runs.get(r), dict) and all(runs[r].get(k) is None for k in ("max_preheat_ramp", "tal_s", "peak_c")) for r in run_ids if truth[r] is None)),
                ("failing_runs is exactly right and sorted", isinstance(fr, list) and fr == sorted(r for r in run_ids if not compliant(truth[r]))),
            ]
        return checks

    return TaskInstance(instruction, verify, meta=dict(truth=truth, limit=limit, liquidus=liquidus, tc_rule=tc_rule,
                                                       fahrenheit=fahrenheit, missing=missing, dt=dt,
                                                       preheat=(preheat_lo, preheat_hi), tal_range=(tal_min, tal_max),
                                                       peak_min=peak_min))
