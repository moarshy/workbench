"""
Submit pre-registered identical replicates one per slot: for each, wait until the previous
submission's slot window ends (the arena's own slot.window_ends_at), refuse to continue if the
previous submission was rejected or its run failed, then hand over to submit_when_open.py
(which never submits twice and follows the run until it is scored).

Usage: python scripts/submit_chain.py eval-rules-desk-v1 eval-rules-desk-v1-r2 eval-rules-desk-v1-r3 ...
       (the first ID is the already-submitted run; files are submission/<id>.json)
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from huggingface_hub import get_token

B = "https://openenvarena-arena.hf.space/api/openenv"
H = {"Authorization": f"Bearer {get_token()}"}
HERE = Path(__file__).resolve().parents[1]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def get(path):
    for _ in range(30):
        try:
            r = httpx.get(f"{B}/{path}", headers=H, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log("poll error", e)
        time.sleep(120)
    return {}


ids = sys.argv[1:]
for prev, nxt in zip(ids, ids[1:]):
    while True:  # wait for the arena to report the previous submission's slot
        s = get(f"submissions/{prev}")
        ends = (s.get("slot") or {}).get("window_ends_at")
        if s.get("state") == "rejected":
            log(f"{prev} was rejected; stopping the chain before {nxt}")
            sys.exit(1)
        if ends:
            break
        time.sleep(300)
    open_at = datetime.fromtimestamp(ends, timezone.utc) + timedelta(seconds=60)
    log(f"{nxt}: waiting for the slot at {open_at.isoformat()}")
    while datetime.now(timezone.utc) < open_at - timedelta(minutes=10):
        time.sleep(600)
    run_id = (get(f"submissions/{prev}").get("run") or {}).get("run_id")
    run = get(f"runs/{run_id}") if run_id else {}
    if run.get("state") in ("failed", "cancelled"):
        log(f"{prev}'s run {run_id} ended {run.get('state')}; stopping the chain before {nxt}")
        sys.exit(1)
    env = {**os.environ, "SUB_FILE": str(HERE / "submission" / f"{nxt}.json"), "SUB_ID": nxt,
           "OPEN_AT_UTC": open_at.isoformat(), "DELAY_MIN": "0"}
    log(f"{nxt}: handing over to submit_when_open.py")
    with open(HERE / "submission" / f"{nxt}.log", "a") as f:
        code = subprocess.call([sys.executable, str(HERE / "scripts" / "submit_when_open.py")], env=env,
                               stdout=f, stderr=subprocess.STDOUT)
    if code != 0:
        log(f"{nxt}: submit_when_open exited {code}; stopping the chain")
        sys.exit(code)
log("chain finished")
