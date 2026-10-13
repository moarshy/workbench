"""Wait for the arena slot, submit once (no duplicates), then follow the run until it is scored."""
import json, os, sys, time
from datetime import datetime, timedelta, timezone
import httpx
from huggingface_hub import get_token

# SUB_FILE, SUB_ID, OPEN_AT_UTC (ISO) come from the environment; DELAY_MIN staggers a backup copy
SUB_FILE = os.environ.get("SUB_FILE", "/Users/arshath/play/rl-envs/workbench/submission/v2.json")
SUB_ID = os.environ.get("SUB_ID", "workbench-v2")
OPEN_AT = datetime.fromisoformat(os.environ.get("OPEN_AT_UTC", "2026-10-12T12:51:00+00:00")) + timedelta(minutes=int(os.environ.get("DELAY_MIN", "0")))
B = "https://openenvarena-arena.hf.space/api"
H = {"Authorization": f"Bearer {get_token()}"}
SUB = json.load(open(SUB_FILE))
assert SUB["submission_id"] == SUB_ID

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

while datetime.now(timezone.utc) < OPEN_AT:
    time.sleep(min(600, max(5, (OPEN_AT - datetime.now(timezone.utc)).total_seconds())))
def already_submitted() -> bool:
    try:
        r = httpx.get(f"{B}/openenv/submissions/{SUB_ID}", headers=H, timeout=60)
        return r.status_code == 200 and r.json().get("submission_id") == SUB_ID
    except Exception:
        return False

for attempt in range(12):  # retry only on refusal-by-timing or proxy errors; stop on any other error
    if already_submitted():
        log(f"{SUB_ID} already exists; not submitting again")
        break
    r = httpx.post(f"{B}/openenv/submissions", headers=H, json=SUB, timeout=180)
    log("submit HTTP", r.status_code, r.text[:300])
    if r.status_code == 202:
        break
    if r.status_code in (429, 502, 503, 504):
        time.sleep(120)
        continue
    sys.exit(1)
else:
    sys.exit(1)

run_id = None
while True:
    try:
        s = httpx.get(f"{B}/openenv/submissions/{SUB_ID}", headers=H, timeout=60).json()
        run_id = (s.get("run") or {}).get("run_id") or run_id
        run = httpx.get(f"{B}/openenv/runs/{run_id}", headers=H, timeout=60).json() if run_id else {}
        ev = run.get("evaluation") or {}
        log("submission:", s.get("state"), "run:", run_id, run.get("state"), "step:", run.get("policy_version"), "eval:", ev.get("state"))
        if s.get("state") == "rejected":
            log(json.dumps(s.get("report", {}).get("error_details"))[:1500]); break
        if run.get("state") in ("failed", "cancelled"):
            log(json.dumps({k: run.get(k) for k in ("error", "error_code", "last_group_error")})); break
        if ev.get("state") == "done":
            log(json.dumps({k: ev.get(k) for k in ("solved", "total", "delta", "scores")})); break
    except Exception as e:
        log("poll error", e)
    time.sleep(600)
