#!/usr/bin/env python3
"""
fleet_demo.py — drive visible, honest activity across the three-B70 Ollama fleet.

Two modes:
  parallel  three prompts fired simultaneously, one per card. The whole fleet
            lights up at once in nvtop/intel_gpu_top.
  relay     an agentic plan -> build -> review chain. Stage N runs on card N and
            consumes the output of stage N-1, so compute visibly hands off
            b70-1 -> b70-2 -> b70-3.

Numbers reported are Ollama's own measured eval_count / eval_duration from the
final stream frame — not wall-clock guesses. Optional --trace writes a
timestamped NDJSON event log suitable for driving the landing-page replay panel.

Stdlib only. No dependencies.

Examples:
  ./fleet_demo.py parallel
  ./fleet_demo.py relay --loop 3 --trace r7_trace.ndjson
  ./fleet_demo.py parallel --model qwen3-coder:30b --num-predict 300
  ./fleet_demo.py parallel --hosts b70-1=http://127.0.0.1:11434,b70-2=http://127.0.0.1:11435
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- fleet config
FLEET = [
    ("b70-1", "http://127.0.0.1:11434"),
    ("b70-2", "http://127.0.0.1:11435"),
    ("b70-3", "http://127.0.0.1:11436"),
]

DEFAULT_MODEL = "qwen3-coder:30b"

PARALLEL_PROMPTS = [
    "In three short paragraphs: what makes a serving benchmark reproducible?",
    "In three short paragraphs: why does KV-cache precision matter for decode throughput?",
    "In three short paragraphs: when does tensor parallelism pay, and when doesn't it?",
]

RELAY_STAGES = [
    ("PLAN",   "You are the planning agent. Produce a numbered 4-step plan for: "
               "a health-check routine for a 3-node LLM inference fleet. Plan only, no code."),
    ("BUILD",  "You are the build agent. Implement step-by-step pseudocode for this plan. "
               "Be concrete and brief.\n\nPLAN:\n{prev}"),
    ("REVIEW", "You are the review agent. Critique this implementation in 5 bullet points: "
               "what breaks first in production?\n\nIMPLEMENTATION:\n{prev}"),
]

COLORS = ["\033[96m", "\033[92m", "\033[95m"]   # cyan / green / magenta per card
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"

trace_lock = threading.Lock()
print_lock = threading.Lock()
trace_fh = None


def trace(event: dict):
    if trace_fh is None:
        return
    event["t"] = time.time_ns()
    with trace_lock:
        trace_fh.write(json.dumps(event) + "\n")
        trace_fh.flush()


def check_endpoint(name: str, base: str) -> bool:
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=4) as r:
            r.read()
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"  {name}  {base}  UNREACHABLE ({e})")
        return False


def generate(name, base, color, model, prompt, num_predict, echo, results, idx,
             raw=False):
    """Stream one generation; collect Ollama's own final metrics."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": -1,
        "options": {"num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(base + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    text, buf, t0 = [], [], time.time()

    def flush(upto=None):
        chunk = "".join(buf)
        if upto is not None and " " in chunk[:upto]:
            cut = chunk.rfind(" ", 0, upto)
            chunk, rest = chunk[:cut], chunk[cut + 1:]
        else:
            rest = ""
        buf.clear()
        if rest:
            buf.append(rest)
        line = chunk.replace("\n", " ").strip()
        if line:
            with print_lock:
                print(f"{color}{name} │{RESET} {line}")

    trace({"ev": "start", "card": name, "model": model})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for rawline in resp:
                frame = json.loads(rawline)
                if tok := frame.get("response"):
                    text.append(tok)
                    trace({"ev": "tok", "card": name, "n": 1})
                    if echo:
                        if raw:
                            sys.stdout.write(color + tok + RESET)
                            sys.stdout.flush()
                        else:
                            buf.append(tok)
                            if "\n" in tok or sum(len(b) for b in buf) > 88:
                                flush(upto=88)
                if frame.get("done"):
                    if echo and not raw:
                        flush()
                    ec = frame.get("eval_count", 0)
                    ed = frame.get("eval_duration", 1)          # ns
                    pc = frame.get("prompt_eval_count", 0)
                    pd = frame.get("prompt_eval_duration", 1)   # ns
                    results[idx] = {
                        "card": name, "ok": True,
                        "eval_count": ec,
                        "decode_toks": ec / (ed / 1e9),
                        "prefill_toks": pc / (pd / 1e9) if pd else 0.0,
                        "wall_s": time.time() - t0,
                        "text": "".join(text),
                    }
                    trace({"ev": "done", "card": name, "eval_count": ec,
                           "decode_toks": results[idx]["decode_toks"]})
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        results[idx] = {"card": name, "ok": False, "error": str(e), "text": ""}
        trace({"ev": "error", "card": name, "error": str(e)})


def summary(results):
    print(f"\n{BOLD}{'card':<8}{'eval toks':>10}{'decode tok/s':>14}"
          f"{'prefill tok/s':>15}{'wall s':>9}{RESET}")
    for r in results:
        if not r:
            continue
        if not r["ok"]:
            print(f"{r['card']:<8}{DIM}failed: {r['error']}{RESET}")
            continue
        print(f"{r['card']:<8}{r['eval_count']:>10}{r['decode_toks']:>14.1f}"
              f"{r['prefill_toks']:>15.1f}{r['wall_s']:>9.1f}")
    print(f"{DIM}metrics are Ollama's own eval_count/eval_duration — "
          f"not wall-clock estimates{RESET}")


def run_parallel(args):
    print(f"\n{BOLD}== PARALLEL — three prompts, three cards, simultaneously =={RESET}\n")
    results = [None, None, None]
    threads = []
    for i, ((name, base), prompt) in enumerate(zip(FLEET, PARALLEL_PROMPTS)):
        print(f"{COLORS[i % len(COLORS)]}▸ {name}{RESET} {DIM}{prompt[:64]}...{RESET}")
        t = threading.Thread(target=generate, args=(
            name, base, COLORS[i % len(COLORS)], args.model, prompt,
            args.num_predict, args.echo, results, i, args.raw))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    summary(results)


def run_relay(args):
    print(f"\n{BOLD}== RELAY — agentic hand-off: plan(b70-1) → build(b70-2) → review(b70-3) =={RESET}")
    results = [None, None, None]
    prev = ""
    for i, ((name, base), (stage, template)) in enumerate(zip(FLEET, RELAY_STAGES)):
        prompt = template.format(prev=prev[:4000])
        print(f"\n{COLORS[i % len(COLORS)]}{BOLD}── {stage} on {name} ──{RESET}")
        trace({"ev": "stage", "stage": stage, "card": name})
        generate(name, base, COLORS[i % len(COLORS)], args.model, prompt,
                 args.num_predict, args.echo, results, i, args.raw)
        if not results[i] or not results[i]["ok"]:
            print(f"\n{stage} failed on {name}; aborting relay.")
            break
        prev = results[i]["text"]
        if not args.echo:
            print(f"{DIM}{prev[:180].strip()}...{RESET}")
    summary(results)


def main():
    p = argparse.ArgumentParser(description="Three-B70 fleet activity demo")
    p.add_argument("mode", choices=["parallel", "relay"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-predict", type=int, default=220,
                   help="cap tokens per generation (default 220 — snappy on camera)")
    p.add_argument("--loop", type=int, default=1, help="repeat N times (default 1)")
    p.add_argument("--raw", action="store_true",
                   help="unbuffered three-stream firehose (chaotic, but honest)")
    p.add_argument("--quiet", dest="echo", action="store_false",
                   help="suppress token streaming, show summaries only")
    p.add_argument("--trace", metavar="FILE",
                   help="write timestamped NDJSON events (replay-panel source)")
    p.add_argument("--hosts", metavar="NAME=URL[,NAME=URL...]",
                   help="override the fleet, e.g. b70-1=http://127.0.0.1:11434,b70-2=http://127.0.0.1:11435 "
                        "(default: the three instances in FLEET)")
    args = p.parse_args()

    global FLEET
    if args.hosts:
        FLEET = [tuple(h.split("=", 1)) for h in args.hosts.split(",") if "=" in h]
        if not FLEET:
            sys.exit("--hosts: expected NAME=URL pairs separated by commas")

    global trace_fh
    if args.trace:
        trace_fh = open(args.trace, "a")

    print(f"{BOLD}fleet check{RESET}")
    ok = all(check_endpoint(n, b) for n, b in FLEET)
    if not ok:
        sys.exit("fleet incomplete — fix endpoints before recording.")
    print(f"  all {len(FLEET)} endpoints up · model={args.model} · num_predict={args.num_predict}")

    for i in range(args.loop):
        if args.loop > 1:
            print(f"\n{DIM}—— pass {i + 1}/{args.loop} ——{RESET}")
        (run_parallel if args.mode == "parallel" else run_relay)(args)

    if trace_fh:
        trace_fh.close()
        print(f"\ntrace written → {args.trace}")


if __name__ == "__main__":
    main()
