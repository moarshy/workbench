"""
Industrial family: line downtime report.

The model gets a per-minute sensor log (too big to read), a long maintenance
handbook and a folder of amendments, and must write /app/output/downtime.json.
Hidden caveats live in the amendments: a micro-stop exclusion (sometimes
superseded by a newer one) and a revised per-shift limit (sometimes with a
future-dated revision that must be ignored). The night shift crosses midnight.

Ground truth is computed from the generated timeline; it never touches disk.
"""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

from .common import Check, TaskInstance, num_eq as _num_eq

SHIFTS = {"day": (6, 14), "evening": (14, 22), "night": (22, 30)}  # hours from midnight; night ends 06:00 next day
LEVELS = {
    "lite": dict(machines=3, days=2, handbook_lines=12, noise_amendments=1),
    "easy": dict(machines=6, days=2, handbook_lines=160, noise_amendments=1),
    "medium": dict(machines=9, days=3, handbook_lines=360, noise_amendments=2),
    "hard": dict(machines=12, days=3, handbook_lines=600, noise_amendments=3),
}

# ------------------------------------------------------------------ handbook text

_FILLER = [
    "Inspect the conveyor belt tension at the start of every {n}th shift and record it in the line book.",
    "Lubricate the drive chain with grade {g} oil; do not mix lubricant grades.",
    "Operators must wear hearing protection when the line runs above {n} parts per minute.",
    "Replace nozzle filters every {n} operating hours or when the pressure drop exceeds {p} kPa.",
    "The reflow zone thermocouples are calibrated quarterly by the metrology team.",
    "Spare parts are requested through the CMMS using the machine ID as reference.",
    "Log every alarm code in the shift handover sheet, including alarms that clear on their own.",
    "Clean the vision camera lens with lint-free wipes only; solvents damage the coating.",
    "Torque values for the gripper mounts are listed in appendix {a}.",
    "Before a changeover the operator verifies that the recipe ID matches the work order.",
    "Coolant concentration should stay between {p} and {q} percent.",
    "Any guard interlock fault must be reported to the shift lead before restart.",
    "Vibration readings above {n} mm/s for more than ten minutes trigger a bearing inspection.",
    "Temperatures are recorded in degrees Celsius at one-minute resolution.",
    "Preventive maintenance windows are scheduled by planning and appear as IDLE in the logs.",
    "Calibration certificates are stored in the quality folder for at least {n} years.",
]
_SECTIONS = ["Safety", "Lubrication", "Changeover procedure", "Alarm handling", "Calibration", "Spare parts",
             "Cleaning", "Quality checks", "Shift handover", "Electrical isolation", "Pneumatics", "Vision system"]


def _filler_line(rng):
    return rng.choice(_FILLER).format(n=rng.randint(2, 12), g=rng.choice(["ISO VG 68", "ISO VG 100", "EP-2"]),
                                      p=rng.randint(3, 9), q=rng.randint(10, 15), a=rng.choice("ABCDEF"))


def _handbook(rng, n_lines: int, key_lines: List[str]) -> str:
    lines = ["# Line A maintenance handbook", "", "Revision 7. Applies to all machines on production line A.", ""]
    body = []
    while len(body) < n_lines:
        body.append(f"## {rng.randint(1, 14)}.{rng.randint(1, 9)} {rng.choice(_SECTIONS)}")
        body += [_filler_line(rng) for _ in range(rng.randint(4, 12))]
        body.append("")
    for key in key_lines:  # bury each key rule somewhere in the middle of the document
        body.insert(rng.randint(len(body) // 4, 3 * len(body) // 4), key)
    return "\n".join(lines + body) + "\n"


def _amendment(number: int, effective: date, text: str) -> str:
    return (f"# Handbook amendment {number}\n\nEffective date: {effective.isoformat()}\n\n{text}\n\n"
            f"Approved by plant engineering.\n")


_NOISE_AMENDMENTS = [
    "Hearing protection is now mandatory in the whole of hall 2, not only next to the presses.",
    "The lubricant supplier changed; the new product code for ISO VG 68 is LX-680.",
    "Shift handover sheets are now filled in electronically in the CMMS.",
    "Vision camera lenses may also be cleaned with isopropyl alcohol wipes.",
    "Spare part requests above 500 EUR need approval from the line manager.",
]


# ------------------------------------------------------------------ timeline


def _timeline(rng, minutes: int, micro_max: int) -> List[str]:
    """Per-minute states for one machine; DOWN stops never cross a shift boundary."""
    boundaries = sorted({d * 1440 + h * 60 for d in range(minutes // 1440 + 1) for h in (6, 14, 22)})
    states: List[str] = []
    while len(states) < minutes:
        states += ["RUN"] * rng.randint(20, 160)
        kind = rng.choices(["micro", "down", "setup", "idle"], weights=[5, 3, 2, 2])[0]
        if kind in ("micro", "down"):
            length = rng.randint(1, micro_max) if kind == "micro" else rng.randint(micro_max + 2, 75)
            t = len(states)
            nxt = next((b for b in boundaries if b > t), minutes)
            if t + length > nxt:  # move the stop to just after the boundary
                states += ["RUN"] * (nxt - t)
            states += ["DOWN"] * length
        else:
            states += ["SETUP" if kind == "setup" else "IDLE"] * (rng.randint(10, 30) if kind == "setup" else rng.randint(15, 60))
    return states[:minutes]


def _down_runs(states: List[str], lo: int, hi: int):
    """Lengths of contiguous DOWN stops that start inside [lo, hi)."""
    runs, t = [], lo
    while t < hi:
        if states[t] == "DOWN" and (t == 0 or states[t - 1] != "DOWN"):
            e = t
            while e < len(states) and states[e] == "DOWN":
                e += 1
            runs.append(e - t)
            t = e
        else:
            t += 1
    return runs


# ------------------------------------------------------------------ generator


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    start = date(2026, 9, rng.randint(7, 20))
    days = cfg["days"]
    machines = [f"M{i:02d}" for i in range(1, cfg["machines"] + 1)]
    shift = rng.choice(list(SHIFTS))
    target = start + timedelta(days=rng.randint(0, days - 2))  # leaves room for the night shift's next morning

    # rules: base limit, plus level-dependent caveats in the amendments
    base_limit = rng.choice([30, 40, 45, 50, 60])
    limit, micro = base_limit, None  # micro = "stops shorter than micro minutes do not count"
    amendments = []  # (effective date, text)
    caveats = []
    if level == "lite":
        caveats = [rng.choice(["micro", "limit"])]
    elif level == "medium":
        caveats = ["micro", "limit"]
    elif level == "hard":
        caveats = ["micro", "limit", "superseded", "future"]
    if "micro" in caveats:
        micro = rng.choice([3, 4, 5])
        amendments.append((date(2026, rng.randint(6, 8), rng.randint(1, 28)),
                           f"Stops shorter than {micro} minutes are classified as micro-stops and are excluded "
                           f"from unplanned downtime. Micro-stops are still logged as DOWN by the sensors."))
    if "superseded" in caveats:
        old = rng.choice([m for m in (2, 6, 8) if m != micro])
        amendments.append((date(2026, rng.randint(1, 4), rng.randint(1, 28)),
                           f"Stops shorter than {old} minutes are classified as micro-stops and are excluded "
                           f"from unplanned downtime. This rule is replaced by any later micro-stop amendment."))
    if "limit" in caveats:
        limit = rng.choice([v for v in (35, 40, 45, 55, 60, 70) if v != base_limit])
        amendments.append((date(2026, rng.randint(5, 8), rng.randint(1, 28)),
                           f"The per-shift unplanned downtime limit is changed from {base_limit} to {limit} minutes."))
    if "future" in caveats:
        future = rng.choice([v for v in (25, 65, 80) if v not in (limit, base_limit)])
        amendments.append((date(2026, rng.randint(10, 12), rng.randint(1, 28)),
                           f"The per-shift unplanned downtime limit is changed from {limit} to {future} minutes."))
    for text in rng.sample(_NOISE_AMENDMENTS, cfg["noise_amendments"]):
        amendments.append((date(2026, rng.randint(1, 8), rng.randint(1, 28)), text))

    # timeline: retry until the answer is non-trivial (some machines over the limit, some not)
    minutes = days * 1440
    micro_max = (micro or 4) + 1  # micro-length stops straddle the threshold, including exactly-at-threshold ones
    h0, h1 = SHIFTS[shift]
    lo = (target - start).days * 1440 + h0 * 60
    hi = (target - start).days * 1440 + h1 * 60
    for _ in range(200):
        timelines = {m: _timeline(rng, minutes, micro_max) for m in machines}
        truth = {m: sum(r for r in _down_runs(timelines[m], lo, hi) if micro is None or r >= micro) for m in machines}
        naive = {m: sum(_down_runs(timelines[m], lo, hi)) for m in machines}
        over = sorted(m for m in machines if truth[m] > limit)
        if 0 < len(over) < len(machines) and (micro is None or naive != truth):
            break

    # ---- write the workspace
    (root / "data").mkdir(parents=True)
    (root / "handbook" / "amendments").mkdir(parents=True)
    (root / "output").mkdir()
    t0 = datetime(start.year, start.month, start.day)
    with open(root / "data" / "sensor_log.csv", "w") as f:
        f.write("timestamp,machine_id,state,temp_c,vibration_mm_s\n")
        for t in range(minutes):
            ts = (t0 + timedelta(minutes=t)).strftime("%Y-%m-%d %H:%M")
            for m in machines:
                s = timelines[m][t]
                temp = 0 if s == "DOWN" else rng.uniform(38, 62)
                f.write(f"{ts},{m},{s},{temp:.1f},{rng.uniform(0.5, 4.5) if s == 'RUN' else 0.2:.2f}\n")
    (root / "data" / "shifts.csv").write_text(
        "shift,start,end,notes\nday,06:00,14:00,\nevening,14:00,22:00,\n"
        "night,22:00,06:00,ends on the next calendar day; a night shift belongs to the date on which it starts\n")
    (root / "data" / "machines.json").write_text(json.dumps(
        [{"id": m, "line": "A", "rated_parts_per_min": rng.randint(40, 120)} for m in machines], indent=1))
    key_lines = [
        "Unplanned downtime is the time a machine spends in state DOWN. Time in SETUP or IDLE is planned and is not downtime.",
        f"Any machine with more than {base_limit} minutes of unplanned downtime in a single shift must be reported "
        f"(this is the per-shift downtime limit).",
    ]
    (root / "handbook" / "maintenance_manual.md").write_text(_handbook(rng, cfg["handbook_lines"], key_lines))
    rng.shuffle(amendments)
    for i, (eff, text) in enumerate(sorted(amendments, key=lambda a: a[0]), start=1):
        # the number keeps names unique when two amendments share an effective date
        (root / "handbook" / "amendments" / f"amendment-{100 + i}_{eff.isoformat()}.md").write_text(_amendment(100 + i, eff, text))

    out_path = root / "output" / "downtime.json"
    instruction = f"""# Line downtime report

Plant engineering needs the unplanned downtime of production line A for the {shift} shift of {target.isoformat()}.

- The plant documentation is in {root}/handbook, including amendments. Use it for the definition of unplanned downtime, the per-shift downtime limit that applies on that date, and any rules about which stops count.
- Sensor data is in {root}/data/sensor_log.csv: one row per machine per minute; `state` is the machine state during that minute.
- Shift times are in {root}/data/shifts.csv.

Write {out_path} with exactly this structure:

{{
  "limit_min": <the per-shift unplanned downtime limit that applies on {target.isoformat()}>,
  "downtime_by_machine": {{"M01": <unplanned downtime minutes>, ...}},
  "over_limit": [<IDs of machines whose downtime is strictly greater than the limit, sorted ascending>]
}}

Include every machine that appears in the sensor log. Do not guess values: derive everything from the files.
"""
    if level == "lite":  # T-Desk budget-fit tier: short, since the instruction repeats every turn
        instruction = f"""# Line downtime
Report unplanned downtime per machine for the {shift} shift of {target.isoformat()}. Rules (definition, the limit in force on that date, which stops count) are in {root}/handbook, including amendments. Data: {root}/data.
Write {out_path}: {{"limit_min": <limit>, "downtime_by_machine": {{"M01": <minutes>, ...}}, "over_limit": [<sorted IDs strictly over the limit>]}}
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        checks: List[Check] = []
        try:
            got = json.loads((ws / "output" / "downtime.json").read_text())
            ok = isinstance(got, dict)
        except Exception:
            got, ok = {}, False
        checks.append(("output file exists and is a JSON object", ok))
        if not ok:
            got = {}
        dbm = got.get("downtime_by_machine")
        ovl = got.get("over_limit")
        checks.append(("has exactly the required keys with the right types",
                       set(got) == {"limit_min", "downtime_by_machine", "over_limit"}
                       and isinstance(got.get("limit_min"), (int, float)) and isinstance(dbm, dict) and isinstance(ovl, list)))
        checks.append(("limit_min is the limit in force on the date", _num_eq(got.get("limit_min"), limit)))
        checks.append(("downtime_by_machine covers exactly the machines in the log", isinstance(dbm, dict) and set(dbm) == set(machines)))
        checks.append(("every machine's downtime minutes are correct",
                       isinstance(dbm, dict) and all(_num_eq(dbm.get(m), truth[m]) for m in machines)))
        checks.append(("over_limit has exactly the right machines", isinstance(ovl, list) and set(map(str, ovl)) == set(over)))
        checks.append(("over_limit is a sorted list of machine IDs",
                       isinstance(ovl, list) and len(ovl) > 0 and all(isinstance(x, str) for x in ovl) and ovl == sorted(ovl)))
        return checks

    return TaskInstance(instruction, verify, meta=dict(
        shift=shift, date=target.isoformat(), limit=limit, base_limit=base_limit, micro=micro,
        caveats=caveats, truth=truth, naive=naive, over=over, machines=machines, n_amendments=len(amendments)))
