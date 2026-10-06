#!/usr/bin/env bash
# Run after `docker compose up -d` in this directory.
# Waits for the model to finish loading (first boot downloads ~2.3GB), then
# verifies the service returns REAL hybrid vectors of the right shape.
set -euo pipefail
BASE="${1:-http://localhost:8055}"
RESP="$(mktemp)"
trap 'rm -f "$RESP"' EXIT

echo "== waiting for model to load (first boot pulls ~2.3GB) =="
for i in $(seq 1 120); do          # up to ~20 min
  status="$(curl -fsS "$BASE/health" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo down)"
  if [ "$status" = "ok" ]; then echo "  model loaded."; break; fi
  printf '  [%3d] status=%s\n' "$i" "$status"
  sleep 10
done

echo; echo "== /health =="
curl -fsS "$BASE/health" | python3 -m json.tool

echo; echo "== /embed_document (real BGE-M3) =="
curl -fsS -X POST "$BASE/embed_document" \
  -H 'Content-Type: application/json' \
  -d '{"text":"# Intel Xeon 6\n\nIntel announced Xeon 6 for the data center.\n\n| Metric | Value |\n|---|---|\n| Cores | 128 |\n| TDP | 350 |","id_seed":"https://example.com/xeon6"}' \
  -o "$RESP"

python3 - "$RESP" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
assert d["n_chunks"] >= 1, "no chunks returned"
c = d["chunks"][0]
dense_dim = len(c["dense"])
nnz = len(c["sparse"]["values"])
assert dense_dim == 1024, "dense dim %d != 1024" % dense_dim
assert set(c["sparse"]) == {"indices", "values"}, "bad sparse shape: %r" % sorted(c["sparse"])
assert nnz > 0, "empty sparse -- model may not actually be running"
assert c.get("point_id"), "no deterministic point_id"
print("OK  n_chunks=%d  dense_dim=%d  sparse_nnz=%d" % (d["n_chunks"], dense_dim, nnz))
print("    point_id=%s" % c["point_id"])
print("    dense[:4]=%s" % [round(x, 4) for x in c["dense"][:4]])
PY

echo; echo "ALL GOOD -- service returns real hybrid points."
