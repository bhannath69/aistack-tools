#!/usr/bin/env python3
"""Build a rerank/embed benchmark corpus from public-domain Project Gutenberg prose.

Produces a JSON object keyed by bucket size ("32", "256", "1024", "4096"), each
holding exactly --per-bucket document strings whose token counts fall between
0.6*L and L inclusive, measured with the tokenizer shipped with the converted
reranker so bucket sizes are exact for that model.

Usage:
    python3 make_corpus.py --tokenizer /m/bge-reranker-ov --out corpus.json
"""
import argparse, json, re, sys, urllib.request
from pathlib import Path

BOOKS = [
    (2701, "Moby-Dick"), (1342, "Pride and Prejudice"), (2600, "War and Peace"),
    (1228, "On the Origin of Species"), (98, "A Tale of Two Cities"), (4300, "Ulysses"),
]


def fetch_book(gid, title, cache):
    path = cache / f"pg{gid}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "make_corpus.py"})
        text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        path.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"warning: could not download {title} ({gid}): {e}", file=sys.stderr)
        return None


def clean(text):
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, l in enumerate(lines) if "*** START OF" in l.upper()), None)
    if start is None:
        return text
    end = next((i for i in range(start + 1, len(lines)) if "*** END OF" in lines[i].upper()), None)
    return "\n".join(lines[start + 1:end] if end is not None else lines[start + 1:])


def paragraphs_roundrobin(paragraph_lists):
    active = [i for i, p in enumerate(paragraph_lists) if p]
    while any(paragraph_lists[i] for i in active):
        for i in active:
            if paragraph_lists[i]:
                yield paragraph_lists[i].pop(0)


def build_bucket(paras, offset, L, per_bucket, ct):
    docs = []
    i, doc = offset, ""
    while len(docs) < per_bucket and i < len(paras):
        p = paras[i]
        i += 1
        if not doc:
            if ct(p) > L:
                words = p.split(" ")
                lo, hi = 1, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if ct(" ".join(words[:mid])) <= L:
                        lo = mid
                    else:
                        hi = mid - 1
                doc = " ".join(words[:lo])
                docs.append(doc)
                doc = ""
            else:
                doc = p
        elif ct(doc + " " + p) <= L:
            doc = doc + " " + p
        else:
            if ct(doc) >= 0.6 * L:
                docs.append(doc)
            doc = p if ct(p) <= L else ""
    if doc and len(docs) < per_bucket and ct(doc) >= 0.6 * L:
        docs.append(doc)
    return docs


def main():
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="corpus.json")
    ap.add_argument("--per-bucket", type=int, default=40)
    ap.add_argument("--buckets", default="32,256,1024,4096")
    ap.add_argument("--cache", default=".gutenberg")
    args = ap.parse_args()
    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    try:
        tok = Tokenizer.from_file(str(tok_path))
    except Exception as e:
        print(f"error: cannot load tokenizer from {args.tokenizer}: {e}", file=sys.stderr)
        sys.exit(1)

    def count_tokens(text):
        return len(tok.encode(text, add_special_tokens=False).ids)

    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    paras_lists = []
    for gid, title in BOOKS:
        text = fetch_book(gid, title, cache)
        if text is None:
            continue
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", clean(text))]
        paras = [p for p in paras if len(p) >= 40 and "Gutenberg" not in p]
        if paras:
            paras_lists.append(paras)
    if not paras_lists:
        print("error: no books downloaded", file=sys.stderr)
        sys.exit(1)
    paras = list(paragraphs_roundrobin(paras_lists))

    buckets = [int(b) for b in args.buckets.split(",")]
    obj = {}
    for bi, L in enumerate(buckets):
        docs = build_bucket(paras, bi * 97, L, args.per_bucket, count_tokens)
        if len(docs) != args.per_bucket or not all(0.6 * L <= count_tokens(d) <= L for d in docs):
            print(f"error: bucket {L} failed validation", file=sys.stderr)
            sys.exit(1)
        obj[str(L)] = docs

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)
    for L in buckets:
        toks = [count_tokens(d) for d in obj[str(L)]]
        print(f"{L}: {len(toks)} docs, tokens min/mean/max = {min(toks)}/{sum(toks)/len(toks):.0f}/{max(toks)}")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("pip install tokenizers")
        sys.exit(1)
