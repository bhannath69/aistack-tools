# aistack-tools

Small, self-contained tools published from the Intel AI stack documented at
[bhannath.ai/stack](https://bhannath.ai/stack). Each directory is one tool; each is the file that
actually ran on that fleet, with host-specific wiring (docker network names, host paths) removed and
nothing else changed. The site's registry cards link here by commit, so a card's command and the
file it fetches cannot drift apart.

MIT licensed (see `LICENSE`). Model weights are not included; each tool says what it downloads.

| directory | what | needs |
|---|---|---|
| `fleet/` | `fleet_demo.py` — fire one `/api/generate` at each of N Ollama instances at the same moment and report each instance's own `eval_count` / `eval_duration` from the final stream frame; `relay` mode chains plan → build → review across instances. `--trace` writes NDJSON. | Python 3.8+, standard library only; one or more Ollama endpoints |
| `bge-embed/` | BGE-M3 hybrid (dense 1024 + sparse) embedding service: FastAPI over `FlagEmbedding`, token-aware markdown chunker, deterministic point IDs, Qdrant-shaped output. `embed_core.py` is the contract module; the service bind-mounts it rather than copying it. | Docker + compose; CPU (≈2.3 GB weight download on first boot) |
| `rerank/` | `ov_rerank.py` — cross-encoder reranking latency across candidate counts (8/32/128) on OpenVINO GenAI's `TextRerankPipeline`, any device OpenVINO can see; `corpus.json` is the document pool: 40 docs in each of four token buckets (32/256/1024/4096), public-domain Project Gutenberg prose, token-counted with the reranker's own tokenizer; `make_corpus.py` rebuilds it. | Docker; `openvino-genai` (pip); an OpenVINO IR of `BAAI/bge-reranker-base` (conversion command below) |

## fleet

```bash
curl -fsSLO https://raw.githubusercontent.com/bhannath69/aistack-tools/main/fleet/fleet_demo.py
python3 fleet_demo.py parallel --hosts b70-1=http://127.0.0.1:11434 --model qwen3.6:27b --num-predict 60
```

`--hosts` takes `NAME=URL` pairs; without it the script expects three instances on `127.0.0.1`
ports 11434–11436 (one per card, as on the fleet it came from) and exits if any is down. Numbers
are Ollama's, not wall-clock.

## bge-embed

```bash
git clone --depth 1 https://github.com/bhannath69/aistack-tools.git && cd aistack-tools/bge-embed
docker compose up -d --build        # first boot downloads BAAI/bge-m3 into the bge_m3_models volume
./smoke_test.sh                     # waits for /health, then asserts dense dim 1024 + non-empty sparse
```

Routes: `GET /health`, `POST /embed` (`{"texts":[...]}` → dense + sparse per text),
`POST /embed_document` (`{"text": markdown, "id_seed": url}` → chunks, each with dense, sparse and a
deterministic `point_id`). `BGE_DEVICE=cuda BGE_FP16=1` runs the same image on an NVIDIA host (swap
the torch wheel in the Dockerfile).

## rerank

Convert the model once (≈1.1 GB IR; this is the step the harness assumed was already done):

```bash
mkdir -p models && docker run --rm -i -v "$PWD/models:/m" python:3.12-slim bash -c '
pip install -q "optimum-intel[openvino]" transformers huggingface_hub
optimum-cli export openvino --model BAAI/bge-reranker-base --task text-classification /m/bge-reranker-ov'
```

Run the CPU arm (stock `openvino-genai`; the GPU/NPU arms on the originating fleet used an image with
a rebuilt driver layer, which CPU does not need):

```bash
mkdir -p out && docker run --rm -i -v "$PWD/models:/m:ro" -v "$PWD/out:/out" -v "$PWD/rerank:/harness:ro" python:3.12-slim bash -c '
pip install -q openvino-genai
python3 /harness/ov_rerank.py --model /m/bge-reranker-ov --device CPU --corpus /harness/corpus.json --candidates 8,32,128 --reps 5 --tag cpu --out /out/rerank.ndjson'
```

The originating fleet measured its tiers on text sampled from its own retrieval corpus; the public
corpus has the same shape (same bucket sizes, same count) but different prose, so compare numbers
across machines only when both ran the same corpus file.

One NDJSON record per candidate count: `load_s`, `latency_p50_ms` (median of `--reps`),
`docs_per_s`, `free_mib`; a device that cannot load the pipeline is recorded as `supported: false`,
not raised.
