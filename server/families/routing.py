"""
Math / formal reasoning family: depot delivery plan (a small vehicle routing problem).

The model gets orders with coordinates, demands and time windows, a vans file,
and operations notes with extra constraints (a van that cannot serve some
customers, a road closure that forbids a pair, a dropped order). It must write a
feasible plan to /app/output/plan.json whose total distance is within 5% of the
optimum. Instances are small enough that the generator finds the exact optimum by
exhaustive search; the model needs code (haversine, feasibility checks, search),
not mental arithmetic.
"""

import itertools
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import Check, TaskInstance, load_json_object, num_eq

LEVELS = {
    "easy": dict(orders=6, notes=0, windows=False),
    "medium": dict(orders=8, notes=2, windows=True),
    "hard": dict(orders=9, notes=3, windows=True),
}
SPEED_KMH = 40.0
SERVICE_MIN = 10
DAY_START = 8 * 60  # vans leave the depot at 08:00


def haversine(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def route_ok(route, van, P) -> Tuple[bool, float]:
    """Feasibility of one van's route and its distance (depot -> stops -> depot)."""
    if sum(P["orders"][o]["demand_kg"] for o in route) > P["vans"][van]["capacity_kg"]:
        return False, 0.0
    t, km, here = DAY_START, 0.0, P["depot"]
    for o in route:
        if o in P["forbidden_van"].get(van, ()):
            return False, 0.0
        nxt = P["orders"][o]
        leg = haversine(here, (nxt["lat"], nxt["lon"]))
        km += leg
        t += leg / SPEED_KMH * 60
        if P["windows"]:
            t = max(t, nxt["window_start"])
            if t > nxt["window_end"]:
                return False, 0.0
        t += SERVICE_MIN
        here = (nxt["lat"], nxt["lon"])
    for a, b in zip(route, route[1:]):
        if frozenset((a, b)) in P["closed_pairs"]:
            return False, 0.0
    km += haversine(here, P["depot"])
    return True, km


def best_route(stops, van, P) -> Optional[Tuple[float, tuple]]:
    best = None
    for perm in itertools.permutations(stops):
        ok, km = route_ok(perm, van, P)
        if ok and (best is None or km < best[0]):
            best = (km, perm)
    return best


def optimum(P) -> Optional[Tuple[float, Dict[str, list]]]:
    """Exact optimum over every split of the orders between the vans."""
    orders = [o for o in P["orders"] if o not in P["dropped"]]
    vans = list(P["vans"])
    cache, best = {}, None
    for mask in range(2 ** len(orders)):
        a = tuple(o for i, o in enumerate(orders) if mask >> i & 1)
        b = tuple(o for i, o in enumerate(orders) if not mask >> i & 1)
        res = []
        for van, stops in ((vans[0], a), (vans[1], b)):
            key = (van, stops)
            if key not in cache:
                cache[key] = best_route(stops, van, P) if stops else (0.0, ())
            res.append(cache[key])
        if None in res:
            continue
        total = res[0][0] + res[1][0]
        if best is None or total < best[0]:
            best = (total, {vans[0]: list(res[0][1]), vans[1]: list(res[1][1])})
    return best


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    for _ in range(500):
        depot = (round(rng.uniform(52.30, 52.42), 5), round(rng.uniform(4.80, 5.00), 5))
        ids = [f"C{i:02d}" for i in range(1, cfg["orders"] + 1)]
        orders = {}
        for o in ids:
            ws = DAY_START + rng.randint(0, 150)
            orders[o] = dict(lat=round(depot[0] + rng.uniform(-0.12, 0.12), 5),
                             lon=round(depot[1] + rng.uniform(-0.2, 0.2), 5),
                             demand_kg=rng.randint(40, 220), window_start=ws, window_end=ws + rng.randint(90, 240))
        total_demand = sum(o["demand_kg"] for o in orders.values())
        cap = int(total_demand * rng.uniform(0.6, 0.75)) // 10 * 10
        vans = {"V1": {"capacity_kg": cap}, "V2": {"capacity_kg": cap}}
        P = dict(depot=depot, orders=orders, vans=vans, windows=cfg["windows"], forbidden_van={},
                 closed_pairs=set(), dropped=set())
        notes, kinds = [], rng.sample(["forbidden", "closed", "dropped"], cfg["notes"])
        if "forbidden" in kinds:
            o = rng.choice(ids)
            P["forbidden_van"]["V2"] = {o}
            notes.append(f"Van V2 is too high for the loading bay of customer {o}; {o} must be served by V1.")
        if "closed" in kinds:
            a, b = rng.sample(ids, 2)
            P["closed_pairs"].add(frozenset((a, b)))
            notes.append(f"Road works: no van may drive directly between {a} and {b} (in either direction).")
        if "dropped" in kinds:
            o = rng.choice([x for x in ids if x not in P["forbidden_van"].get("V2", ())])
            P["dropped"].add(o)
            notes.append(f"Order {o} was cancelled by the customer this morning; do not serve it.")
        opt = optimum(P)
        if opt is None:
            continue
        # the notes must matter: the optimum while ignoring them must break one of them
        if notes:
            P_naive = {**P, "forbidden_van": {}, "closed_pairs": set(), "dropped": set()}
            naive = optimum(P_naive)
            if naive is None or abs(naive[0] - opt[0]) < 0.5:
                continue
        break

    opt_km, opt_routes = opt
    served = [o for o in ids if o not in P["dropped"]]

    # ---- write the workspace
    for sub in ("data", "output"):
        (root / sub).mkdir(parents=True)

    def hhmm(m):
        return f"{m // 60:02d}:{m % 60:02d}"
    with open(root / "data" / "orders.csv", "w") as f:
        f.write("order_id,lat,lon,demand_kg,window_start,window_end\n")
        for o in ids:
            d = orders[o]
            ws_, we_ = (hhmm(d["window_start"]), hhmm(d["window_end"])) if cfg["windows"] else ("08:00", "20:00")
            f.write(f"{o},{d['lat']},{d['lon']},{d['demand_kg']},{ws_},{we_}\n")
    (root / "data" / "vans.json").write_text(json.dumps(
        {"depot": {"lat": depot[0], "lon": depot[1]}, "departure": "08:00", "speed_kmh": SPEED_KMH,
         "service_minutes_per_stop": SERVICE_MIN, "vans": vans}, indent=1))
    ops = ["# Dispatch notes for today", "",
           "- Vans start and end at the depot and leave at the departure time in vans.json.",
           "- A van may arrive early and wait for a window to open; it may not arrive after a window closes.",
           *[f"- {n}" for n in notes], "- Fuel cards were renewed; no action needed.", ""]
    (root / "data" / "ops_notes.md").write_text("\n".join(ops))

    out_path = root / "output" / "plan.json"
    windows_txt = "time windows (a van may wait for a window to open), " if cfg["windows"] else ""
    instruction = f"""# Depot delivery plan

Plan today's deliveries for two vans, V1 and V2.

- Orders are in {root}/data/orders.csv; the depot, vans, speed and service time are in {root}/data/vans.json.
- Read {root}/data/ops_notes.md: every note there is a binding constraint.
- Distances are great-circle distances on a sphere of radius 6371 km between the coordinates. Travel time = distance / speed; each stop takes the service time.
- Respect van capacity, {windows_txt}and every note. Each remaining order is served exactly once.
- Minimise the total distance driven by both vans, including the return to the depot.

Write {out_path} with exactly this structure:

{{
  "routes": {{"V1": [<order IDs in visiting order>], "V2": [<order IDs in visiting order>]}},
  "total_km": <total distance of your plan in km, 2 decimals>
}}

A plan is accepted if it is feasible and its total distance is within 5% of the best possible plan.
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        got, ok = load_json_object(ws / "output" / "plan.json")
        routes, tot = got.get("routes"), got.get("total_km")
        shape = (set(got) == {"routes", "total_km"} and isinstance(routes, dict) and set(routes) == {"V1", "V2"}
                 and all(isinstance(r, list) and all(isinstance(x, str) for x in r) for r in routes.values())
                 and isinstance(tot, (int, float)))
        visits = [o for r in routes.values() for o in r] if shape else []
        each_once = shape and sorted(visits) == sorted(served)
        feas, km = (False, None)
        if each_once:
            parts = [route_ok(tuple(routes[v]), v, P) for v in ("V1", "V2")]
            feas = all(p[0] for p in parts)
            km = sum(p[1] for p in parts) if feas else None
        return [
            ("output file exists and is a JSON object", ok),
            ("has routes for exactly V1 and V2 and a numeric total_km", shape),
            ("every remaining order is served exactly once (cancelled orders are not)", each_once),
            ("routes respect capacity, time windows and the dispatch notes", feas),
            ("total_km matches the plan's actual distance (within 0.05 km)", km is not None and num_eq(tot, km, 0.05)),
            ("total distance is within 5% of the optimum", km is not None and km <= opt_km * 1.05 + 1e-6),
        ]

    return TaskInstance(instruction, verify, meta=dict(opt_km=opt_km, opt_routes=opt_routes, notes=notes,
                                                       served=served, P=P))
