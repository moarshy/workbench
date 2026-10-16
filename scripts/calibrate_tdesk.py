"""
T-Desk difficulty check under the arena's budget rules (as described in its AGENTS.md):

  - system message = the action schema; first user turn = the reset observation as JSON
  - each reply must be exactly one JSON object (strict by default): an action, or {"finish": true}
  - completion budget (4,096) counts the model's tokens plus every observation after the first
  - context budget (8,192) covers the whole conversation; an observation that no longer fits ends it
  - when an episode ends without evaluate, the finish_action {"action_type": "evaluate"} grades it

Usage:
  for p in 8030 8031 8032 8033; do docker run -d -p $p:8000 ghcr.io/moarshy/workbench:tdesk-dev; done
  python scripts/calibrate_tdesk.py --episodes 8 --ports 8030 8031 8032 8033
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from workbench.server.tdesk_environment import TASKS  # noqa: E402
from workbench.tdesk_client import TDeskEnv  # noqa: E402
from workbench.tdesk_models import Tbench2Action  # noqa: E402

MODEL = "qwen/qwen3.8-27b"
COMPLETION_BUDGET, CONTEXT_BUDGET = 4096, 8192  # overridable with --completion-budget / --context-budget
SYSTEM = ("You are acting in an environment. Each of your replies must be exactly one JSON object and nothing else: "
          "either an action matching this JSON schema, or {\"finish\": true} to end the episode.\n\n"
          f"Action schema:\n{json.dumps(Tbench2Action.model_json_schema())}")


def obs_json(obs) -> str:
    return json.dumps({k: v for k, v in obs.model_dump().items() if k != "metadata"})


def parse(reply: str, lenient: bool):
    text = reply.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        if not lenient:
            return None
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    return None


def chat(client, messages, max_tokens):
    for attempt in range(4):
        try:
            for _ in range(3):  # empty provider replies are not the model's fault
                r = client.chat_completion(messages=messages, model=MODEL, max_tokens=max(16, max_tokens),
                                           temperature=1.0, extra_body={"reasoning": {"enabled": False}})
                if (r.choices[0].message.content or "").strip():
                    return r
            return r
        except Exception:
            time.sleep(2 ** attempt)
    return None


def play(client, url, task_id, seed, lenient):
    with TDeskEnv(base_url=url).sync() as env:
        r = env.reset(task_id=task_id, seed=seed)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_json(r.observation)}]
        first_prompt, outcome, actions, final = None, None, [], None
        while outcome is None:
            used = 0
            resp = chat(client, messages, COMPLETION_BUDGET)
            if resp is None:
                outcome = "api_error"
                break
            reply = resp.choices[0].message.content or ""
            p, c = resp.usage.prompt_tokens, resp.usage.completion_tokens
            first_prompt = first_prompt or p
            used = (p - first_prompt) + c          # model tokens + observations after the first
            messages.append({"role": "assistant", "content": reply})
            if used > COMPLETION_BUDGET or p + c > CONTEXT_BUDGET:
                outcome = "budget"
                break
            action = parse(reply, lenient)
            if action is None:
                outcome = "invalid_reply"
                break
            if action.get("finish") is True:
                outcome = "finished"
                break
            try:
                act = Tbench2Action(**{k: v for k, v in action.items() if k in Tbench2Action.model_fields})
            except Exception:
                outcome = "invalid_action"
                break
            actions.append(act.action_type)
            r = env.step(act)
            if r.done:
                final, outcome = r, "evaluated"
                break
            messages.append({"role": "user", "content": obs_json(r.observation)})
        if final is None:  # the arena's finish_action
            final = env.step(Tbench2Action(action_type="evaluate"))
        return dict(task=task_id, seed=seed, outcome=outcome, reward=final.reward, actions=actions,
                    budget_used=used, checks=final.observation.output.splitlines()[-1:], messages=messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--ports", nargs="+", type=int, default=[8030])
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--completion-budget", type=int, default=4096)
    ap.add_argument("--context-budget", type=int, default=8192)
    ap.add_argument("--out", default=str(ROOT / "calibration" / "tdesk.jsonl"))
    args = ap.parse_args()
    global COMPLETION_BUDGET, CONTEXT_BUDGET
    COMPLETION_BUDGET, CONTEXT_BUDGET = args.completion_budget, args.context_budget
    client = InferenceClient(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    jobs = queue.Queue()
    for t in args.tasks:
        for i in range(args.episodes):
            jobs.put((t, 7000 + i))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    results, lock = [], threading.Lock()

    def worker(port):
        while True:
            try:
                task_id, seed = jobs.get_nowait()
            except queue.Empty:
                return
            res = None
            for attempt in range(5):  # one session per server: wait for the previous one to be released
                try:
                    res = play(client, f"http://127.0.0.1:{port}", task_id, seed, args.lenient)
                    break
                except Exception as e:
                    res = dict(task=task_id, seed=seed, outcome=f"error: {str(e)[:80]}", reward=0.0, actions=[],
                               budget_used=0, checks=[], messages=[])
                    time.sleep(10 * (attempt + 1))
            with lock:
                results.append(res)
                with out.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{task_id:18s} seed={seed} {res['outcome']:14s} reward={res['reward']} actions={len(res['actions'])} "
                      f"budget={res['budget_used']} {res['checks']}", flush=True)

    threads = [threading.Thread(target=worker, args=(p,)) for p in args.ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    by = defaultdict(list)
    for r in results:
        by[r["task"]].append(r)
    print(f"\n{'task':18s} {'solved':>7s}  outcomes")
    for t in args.tasks:
        rs = by[t]
        if rs:
            oc = defaultdict(int)
            for r in rs:
                oc[r["outcome"]] += 1
            print(f"{t:18s} {sum(r['reward'] == 1.0 for r in rs):>3d}/{len(rs):<3d}  {dict(oc)}")


if __name__ == "__main__":
    main()
