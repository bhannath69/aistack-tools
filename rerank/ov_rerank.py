#!/usr/bin/env python3
"""Cross-encoder reranking across tiers (SPEC §17.4, sweep R).

Reranking sits between retrieval and generation on the user's critical path, so
its cost adds directly to the wait. Like embedding it is pure forward-pass work
with no decode phase.
"""
import argparse, json, statistics as st, sys, time


def free_mib():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) // 1024
    return None


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True)
    a.add_argument("--device", default="GPU")
    a.add_argument("--corpus", required=True)
    a.add_argument("--candidates", default="8,32,128")
    a.add_argument("--doc-tokens", type=int, default=256)
    a.add_argument("--reps", type=int, default=5)
    a.add_argument("--tag", default="")
    a.add_argument("--out", default="")
    args = a.parse_args()

    import openvino_genai as ovg
    corpus = json.load(open(args.corpus))
    docs_pool = corpus.get(str(args.doc_tokens)) or corpus["256"]
    query = "What are the memory bandwidth characteristics of integrated graphics?"

    try:
        t0 = time.time()
        pipe = ovg.TextRerankPipeline(args.model, args.device)
        load_s = round(time.time() - t0, 2)
    except Exception as e:
        rec = {"tag": args.tag, "device": args.device, "supported": False,
               "error": f"{type(e).__name__}: {e}"[:400], "free_mib": free_mib()}
        print(json.dumps(rec))
        if args.out: open(args.out, "a").write(json.dumps(rec) + "\n")
        return 0

    results = []
    for N in [int(x) for x in args.candidates.split(",")]:
        docs = [docs_pool[i % len(docs_pool)] for i in range(N)]
        try:
            pipe.rerank(query, docs)
        except Exception as e:
            print(f"  warm failed N={N}: {e}", file=sys.stderr); continue
        lat = []
        for _ in range(args.reps):
            t = time.time()
            try: pipe.rerank(query, docs)
            except Exception as e:
                print(f"  failed N={N}: {e}", file=sys.stderr); break
            lat.append((time.time() - t) * 1000)
        if not lat: continue
        med = st.median(lat)
        rec = {"tag": args.tag, "device": args.device, "supported": True,
               "candidates": N, "doc_tokens": args.doc_tokens, "reps": len(lat),
               "load_s": load_s, "latency_p50_ms": round(med, 2),
               "docs_per_s": round(N / (med / 1000), 1),
               "free_mib": free_mib()}
        results.append(rec); print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, "a") as _f: _f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())
