"""
Difficulty check: Qwen3.8-27B (thinking off, via OpenRouter) plays Workbench tasks
against running containers the way the arena does: system message = action
schema, observations as JSON user turns, one JSON action per reply, invalid
reply = episode ends and is graded as-is (the arena's finish fallback).

One container per worker (each server has one workspace):
  for p in 8001 8002 8003 8004; do docker run -d -p $p:8000 ghcr.io/moarshy/workbench:dev; done
  python scripts/calibrate.py --episodes 8 --ports 8001 8002 8003 8004
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

from workbench.client import WorkbenchEnv  # noqa: E402
from workbench.models import WorkbenchAction, WorkbenchObservation  # noqa: E402
from workbench.server.workbench_environment import TASKS  # noqa: E402

MODEL = "qwen/qwen3.8-27b"
SYSTEM = (
    "You are acting in an environment. Each of your replies must be exactly one JSON object and nothing else: "
    "either an action matching this JSON schema, or {\"finish\": true} to end the episode.\n\n"
    f"Action schema:\n{json.dumps(WorkbenchAction.model_json_schema())}"
)
CONTEXT_TOKENS = 16384


def parse_reply(reply: str, lenient: bool):
    """Strict: the whole reply is one JSON object. Lenient: the first JSON object anywhere in the reply."""
    text = reply.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        if not lenient:
            return None
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = text.find("{", start + 1)
    return None


def obs_json(obs) -> str:
    return json.dumps({k: v for k, v in obs.model_dump().items() if k not in ("metadata",)})


def play(client, url, task_id, seed, max_tokens, lenient=False):
    env = WorkbenchEnv(base_url=url).sync()
    with env:
        r = env.reset(task_id=task_id, seed=seed)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_json(r.observation)}]
        outcome, usage = "running", {"prompt": 0, "completion": 0, "peak_context": 0}
        final = None
        while True:
            for attempt in range(4):
                try:
                    resp = client.chat_completion(messages=messages, model=MODEL, max_tokens=2048, temperature=1.0,
                                                  extra_body={"reasoning": {"enabled": False}})
                    break
                except Exception as e:
                    if attempt == 3:
                        return dict(task=task_id, seed=seed, outcome=f"api_error: {e}", reward=0.0, usage=usage, messages=messages)
                    time.sleep(2 ** attempt)
            reply = resp.choices[0].message.content or ""
            if not reply.strip():  # empty provider responses are not the model's fault: retry once
                for _ in range(2):
                    try:
                        resp = client.chat_completion(messages=messages, model=MODEL, max_tokens=2048, temperature=1.0,
                                                      extra_body={"reasoning": {"enabled": False}})
                    except Exception:
                        continue
                    reply = resp.choices[0].message.content or ""
                    if reply.strip():
                        break
            if resp.usage:
                usage["prompt"] += resp.usage.prompt_tokens
                usage["completion"] += resp.usage.completion_tokens
                usage["peak_context"] = max(usage["peak_context"], resp.usage.prompt_tokens + resp.usage.completion_tokens)
            messages.append({"role": "assistant", "content": reply})
            action = parse_reply(reply, lenient)
            if action is None:
                outcome = "invalid_reply"
            elif lenient and reply.strip() != json.dumps(action) and not reply.strip().startswith("{"):
                usage["prose_replies"] = usage.get("prose_replies", 0) + 1
            if action is not None and action != {"finish": True}:
                try:
                    act = WorkbenchAction(**action)
                except Exception:
                    outcome, act = "invalid_action", None
                if act is not None:
                    r = env.step(act)
                    messages.append({"role": "user", "content": obs_json(r.observation)})
                    if r.done:
                        final = r
                        outcome = "submitted" if act.type == "submit" else "out_of_steps"
                    elif usage["peak_context"] > CONTEXT_TOKENS:
                        outcome = "context_full"
            elif action == {"finish": True}:
                outcome = "finished"
            if outcome != "running":
                break
        if final is None:  # arena fallback: grade whatever is on disk via submit
            final = env.step(WorkbenchAction(type="submit", answer=""))
        return dict(task=task_id, seed=seed, outcome=outcome, reward=final.reward,
                    checks=final.observation.output, steps=sum(m["role"] == "assistant" for m in messages),
                    usage=usage, messages=messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--ports", nargs="+", type=int, default=[8000])
    ap.add_argument("--out", default=str(ROOT / "calibration" / "run.jsonl"))
    ap.add_argument("--lenient", action="store_true", help="accept prose around the JSON action")
    args = ap.parse_args()

    client = InferenceClient(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    jobs = queue.Queue()
    for t in args.tasks:
        for i in range(args.episodes):
            jobs.put((t, 5000 + i))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results, lock = [], threading.Lock()

    def worker(port):
        while True:
            try:
                task_id, seed = jobs.get_nowait()
            except queue.Empty:
                return
            r = play(client, f"http://127.0.0.1:{port}", task_id, seed, 2048, args.lenient)
            with lock:
                results.append(r)
                with out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{r['task']:16s} seed={seed} {r['outcome']:14s} reward={r['reward']:.2f} steps={r.get('steps')} "
                      f"peak_ctx={r['usage']['peak_context']}  {r.get('checks', '')}", flush=True)

    out.write_text("")
    threads = [threading.Thread(target=worker, args=(p,)) for p in args.ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    by = defaultdict(list)
    for r in results:
        by[r["task"]].append(r)
    print(f"\n{'task':16s} {'mean':>5s} {'full':>5s}  outcomes")
    for t in args.tasks:
        rs = by[t]
        if not rs:
            continue
        oc = defaultdict(int)
        for r in rs:
            oc[r["outcome"]] += 1
        print(f"{t:16s} {sum(r['reward'] for r in rs) / len(rs):5.2f} {sum(r['reward'] == 1.0 for r in rs):>2d}/{len(rs):<2d}  {dict(oc)}")
    print(f"tokens: {sum(r['usage']['prompt'] for r in results):,} in, {sum(r['usage']['completion'] for r in results):,} out")


if __name__ == "__main__":
    main()
