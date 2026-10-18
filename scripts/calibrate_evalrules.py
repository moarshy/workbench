"""
Eval-rules desk (v9) difficulty check under the hidden test's published rules:

  - system message = the action schema; first user turn = the reset observation as JSON
  - each reply must be exactly one JSON object (strict by default): an action, or {"finish": true}
  - completion budget (8,192) counts the model's tokens plus every observation after the first
  - context budget (16,384) covers the whole conversation; an observation that no longer fits ends it
  - when an episode ends without submit, the finish_action {"type": "finalize"} grades it
  - records how each episode ended (the day-0 proxy's categories), to compare death profiles

Usage:
  for p in 8201 8202 8203 8204; do docker run -d -p $p:8000 ghcr.io/moarshy/workbench:v9-dev; done
  python scripts/calibrate_evalrules.py --episodes 8 --ports 8201 8202 8203 8204
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from workbench.evalrules_client import EvalRulesEnv  # noqa: E402
from workbench.evalrules_models import PostTrainAction  # noqa: E402
from workbench.server.evalrules_environment import TASKS  # noqa: E402

MODEL = "qwen/qwen3.8-27b"
COMPLETION_BUDGET, CONTEXT_BUDGET = 8192, 16384  # overridable with --completion-budget / --context-budget
TEMPERATURE = 1.0
SYSTEM = ("You are acting in an environment. Each of your replies must be exactly one JSON object and nothing else: "
          "either an action matching this JSON schema, or {\"finish\": true} to end the episode.\n\n"
          f"Action schema:\n{json.dumps(PostTrainAction.model_json_schema())}")


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


def classify_invalid(reply: str, finish_reason: str) -> str:
    t = reply.strip()
    if not t:
        return "empty"
    if finish_reason == "length":
        return "cut_off_by_budget"
    if "<tool_call>" in t:
        return "tool_call_tags"
    if not t.startswith("{"):
        return "prose_before_json" if "{" in t else "no_json"
    try:
        json.JSONDecoder().raw_decode(t)
        return "text_after_json"
    except Exception:
        return "broken_json_escaping"


def chat(client, messages, max_tokens):
    for attempt in range(4):
        try:
            for _ in range(3):  # empty provider replies are not the model's fault
                r = client.chat_completion(messages=messages, model=MODEL, max_tokens=max(16, max_tokens),
                                           temperature=TEMPERATURE, extra_body={"reasoning": {"enabled": False}})
                if (r.choices[0].message.content or "").strip():
                    return r
            return r
        except Exception:
            time.sleep(2 ** attempt)
    return None


def play(client, url, task_id, seed, lenient):
    with EvalRulesEnv(base_url=url).sync() as env:
        r = env.reset(task_id=task_id, seed=seed)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_json(r.observation)}]
        first_prompt, outcome, actions, final, detail, used = None, None, [], None, "", 0
        while outcome is None:
            resp = chat(client, messages, max(16, COMPLETION_BUDGET - used))
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
                outcome, detail = "invalid_reply", classify_invalid(reply, resp.choices[0].finish_reason)
                break
            if action.get("finish") is True:
                outcome = "finished"
                break
            try:
                act = PostTrainAction(**action)
            except Exception:
                outcome = "invalid_action"
                break
            actions.append(act.type)
            r = env.step(act)
            if r.done:
                final, outcome = r, ("submitted" if act.type == "submit" else "finalized")
                break
            messages.append({"role": "user", "content": obs_json(r.observation)})
        if final is None:  # the arena's finish_action
            final = env.step(PostTrainAction(type="finalize"))
        m = re.search(r"(\d+)/(\d+) checks passed", final.observation.output)
        solved = bool(m) and m.group(1) == m.group(2)
        return dict(task=task_id, seed=seed, outcome=outcome, detail=detail, reward=final.reward, solved=solved,
                    actions=actions,
                    budget_used=used, checks=final.observation.output.splitlines()[-1:], messages=messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--ports", nargs="+", type=int, default=[8030])
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--completion-budget", type=int, default=8192)
    ap.add_argument("--context-budget", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "calibration" / "v9.jsonl"))
    args = ap.parse_args()
    global COMPLETION_BUDGET, CONTEXT_BUDGET, TEMPERATURE
    COMPLETION_BUDGET, CONTEXT_BUDGET, TEMPERATURE = args.completion_budget, args.context_budget, args.temperature
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
                    res = dict(task=task_id, seed=seed, outcome=f"error: {str(e)[:80]}", detail="", reward=0.0,
                               solved=False, actions=[],
                               budget_used=0, checks=[], messages=[])
                    time.sleep(10 * (attempt + 1))
            with lock:
                results.append(res)
                with out.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"{task_id:18s} seed={seed} {res['outcome']:14s} {res['detail']:20s} reward={res['reward']} "
                      f"solved={res['solved']} actions={len(res['actions'])} "
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
            print(f"{t:18s} {sum(bool(r['solved']) for r in rs):>3d}/{len(rs):<3d}  {dict(oc)}")


if __name__ == "__main__":
    main()
