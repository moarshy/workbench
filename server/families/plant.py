"""
Industrial family: identify and tune a thermal process (first order plus dead time).

The workspace has plant.py, a simulator of a heater zone whose parameters
(gain K, time constant tau, dead time theta) are sealed in an encoded blob, and a
short SPEC.md. The model has to run experiments, not read the answer:

  identify  step test -> estimate K (C per % heater), tau (s), theta (s)  -> params.json
  tune      identify, then compute PI gains with the SPEC's lambda-tuning rule,
            check the closed loop, and write gains.json; the grader re-simulates
            the closed loop with a hidden noise seed and checks overshoot and settling

Ground truth (K, tau, theta and the gains they imply) stays in the server.
"""

import base64
import json
import random
import zlib
from pathlib import Path
from typing import List

from .common import Check, TaskInstance, load_json_object

LEVELS = {"identify": dict(tune=False), "tune": dict(tune=True)}

PLANT_SRC = '''"""Heater zone simulator (first order plus dead time). Do not edit.

    simulate(heater, dt=1.0, seed=None) -> list of temperatures (C), one per input sample
        heater: list of heater outputs in percent (0-100), one value per dt seconds
    closed_loop(kp, ki, setpoint, seconds, dt=1.0, seed=None) -> list of temperatures
        PI controller u = kp*e + ki*integral(e), clamped to 0..100 %, starting from ambient

Ambient temperature is AMBIENT. Measurement noise is about +/- NOISE C.
"""
import base64, json, random, zlib

AMBIENT = {ambient}
NOISE = {noise}
_P = json.loads(zlib.decompress(base64.b64decode("{blob}")))


def _step(y, u_delayed, dt):
    return y + dt * (_P["k"] * u_delayed - (y - AMBIENT)) / _P["tau"]


def simulate(heater, dt=1.0, seed=None):
    rng = random.Random(seed)
    lag = int(round(_P["theta"] / dt))
    y, out = AMBIENT, []
    for i in range(len(heater)):
        u = heater[i - lag] if i >= lag else 0.0
        y = _step(y, u, dt)
        out.append(round(y + rng.uniform(-NOISE, NOISE), 3))
    return out


def closed_loop(kp, ki, setpoint, seconds, dt=1.0, seed=None):
    rng = random.Random(seed)
    lag = int(round(_P["theta"] / dt))
    y, integral, history, out = AMBIENT, 0.0, [], []
    for i in range(int(seconds / dt)):
        meas = y + rng.uniform(-NOISE, NOISE)
        e = setpoint - meas
        integral += e * dt
        u = max(0.0, min(100.0, kp * e + ki * integral))
        history.append(u)
        u_delayed = history[i - lag] if i >= lag else 0.0
        y = _step(y, u_delayed, dt)
        out.append(round(meas, 3))
    return out
'''


def _sim_closed(k, tau, theta, ambient, noise, kp, ki, setpoint, seconds, seed, dt=1.0):
    rng = random.Random(seed)
    lag = int(round(theta / dt))
    y, integral, hist, out = ambient, 0.0, [], []
    for i in range(int(seconds / dt)):
        meas = y + rng.uniform(-noise, noise)
        e = setpoint - meas
        integral += e * dt
        u = max(0.0, min(100.0, kp * e + ki * integral))
        hist.append(u)
        ud = hist[i - lag] if i >= lag else 0.0
        y = y + dt * (k * ud - (y - ambient)) / tau
        out.append(y)
    return out


def _metrics(series, ambient, setpoint, band=0.02):
    """Overshoot (fraction of the step) and settling time (s) into +/- band of the step size."""
    step = setpoint - ambient
    overshoot = max(0.0, (max(series) - setpoint) / step)
    settle = None
    for i in range(len(series)):
        if all(abs(v - setpoint) <= band * step for v in series[i:]):
            settle = i
            break
    return overshoot, settle


def generate(rng: random.Random, level: str, root: Path) -> TaskInstance:
    cfg = LEVELS[level]
    k = round(rng.uniform(0.4, 1.6), 3)          # C per % heater
    tau = round(rng.uniform(40, 160), 1)         # s
    theta = float(rng.randint(3, 15))            # s
    ambient = round(rng.uniform(20, 28), 1)
    noise = round(rng.uniform(0.05, 0.25), 2)
    lam_factor = rng.choice([1.0, 1.5, 2.0])     # lambda = factor * tau (closed-loop time constant)
    setpoint = round(ambient + min(45.0, max(10.0, rng.uniform(0.3, 0.7) * k * 100)), 1)  # reachable at 30-70 % heater
    lam = lam_factor * tau
    kp_true = tau / (k * (lam + theta))
    ki_true = kp_true / tau
    max_overshoot, settle_limit = 0.05, round(6 * (lam + theta))

    blob = base64.b64encode(zlib.compress(json.dumps({"k": k, "tau": tau, "theta": theta}).encode())).decode()
    (root / "output").mkdir(parents=True)
    (root / "plant.py").write_text(PLANT_SRC.replace("{ambient}", str(ambient)).replace("{noise}", str(noise)).replace("{blob}", blob))
    spec = f"""# Zone 4 heater: identification and tuning spec

## Model
The zone behaves as a first-order-plus-dead-time process:
  temperature responds to heater output u (%) with gain K (C per %), time constant tau (s) and dead time theta (s).
Identify the model from experiments with plant.py (do not read or edit its internals): for example apply a step in heater
output from 0 % and observe the response. Use the 63.2 % rise point for tau and the first clear rise for theta.

## Tuning rule (lambda / IMC tuning for a PI controller)
  lambda = {lam_factor:g} * tau
  Kp = tau / (K * (lambda + theta))      (% per C)
  Ki = Kp / tau                          (% per C per s)

## Acceptance
- Overshoot at most {int(max_overshoot * 100)} % of the step, for a setpoint step from ambient to {setpoint:g} C.
- Settled within +/-2 % of the step by {settle_limit} s.
"""
    (root / "SPEC.md").write_text(spec)

    if cfg["tune"]:
        out = root / "output" / "gains.json"
        schema = '{"K": <C per %>, "tau_s": <s>, "theta_s": <s>, "kp": <% per C>, "ki": <% per C per s>}'
        job = ("identify the process, compute the PI gains with the SPEC's tuning rule, check the closed loop "
               "against the acceptance criteria with plant.closed_loop, and")
    else:
        out = root / "output" / "params.json"
        schema = '{"K": <C per %>, "tau_s": <s>, "theta_s": <s>}'
        job = "identify the process model from a step test and"
    instruction = f"""# Heater zone {('tuning' if cfg['tune'] else 'identification')}

Zone 4 of the curing oven needs {'a new PI controller' if cfg['tune'] else 'a process model'}. The simulator is {root}/plant.py
and the requirements are in {root}/SPEC.md. Run experiments with the simulator (import it from Python) to {job} write {out}:

{schema}

Numbers only (no units). Do not edit plant.py.
"""
    (root / "TASK.md").write_text(instruction)

    def verify(ws: Path) -> List[Check]:
        got, ok = load_json_object(ws / "output" / ("gains.json" if cfg["tune"] else "params.json"))

        def num(key):
            v = got.get(key)
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        K, T, D = num("K"), num("tau_s"), num("theta_s")
        checks: List[Check] = [
            ("output file exists and is a JSON object", ok),
            ("gain K within 5 %", K is not None and abs(K - k) <= 0.05 * k),
            ("time constant tau within 15 %", T is not None and abs(T - tau) <= 0.15 * tau),
            ("dead time theta within max(4 s, 40 %)", D is not None and abs(D - theta) <= max(4.0, 0.40 * theta)),
        ]
        if cfg["tune"]:
            kp, ki = num("kp"), num("ki")
            checks.append(("Kp within 15 % of the tuning rule", kp is not None and abs(kp - kp_true) <= 0.15 * kp_true))
            checks.append(("Ki within 20 % of the tuning rule", ki is not None and abs(ki - ki_true) <= 0.20 * ki_true))
            meets = False
            if kp is not None and ki is not None and 0 < kp < 1e4 and 0 <= ki < 1e3:
                series = _sim_closed(k, tau, theta, ambient, noise, kp, ki, setpoint, settle_limit + 200,
                                     seed=random.randrange(10 ** 9))
                ov, st = _metrics(series, ambient, setpoint)
                meets = ov <= max_overshoot and st is not None and st <= settle_limit
            checks.append(("closed loop (hidden noise) meets overshoot and settling", meets))
        return checks

    return TaskInstance(instruction, verify, meta=dict(k=k, tau=tau, theta=theta, kp=kp_true, ki=ki_true,
                                                       ambient=ambient, setpoint=setpoint, lam_factor=lam_factor,
                                                       settle_limit=settle_limit, noise=noise))
