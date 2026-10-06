#!/usr/bin/env python3
"""
bge_m3_service.py — BGE-M3 hybrid (dense+sparse) embedding HTTP service.

The n8n market-trends path cannot produce hybrid named vectors (Ollama is
dense-only; the langchain Qdrant node writes a single unnamed vector). This
service closes that gap: it returns {dense:[1024 floats], sparse:{indices,values}}
so n8n can upsert named-vector points into `market-trends` exactly like the
watchdog writes `enterprise-context`.

SINGLE SOURCE OF TRUTH: everything that must not drift between paths is imported
from embed_core (the same module ingest.py uses) — the embedder, the
sparse->Qdrant conversion, the vector names, and the token-aware chunker. This
service adds NO new embedding/chunking logic; it only wraps embed_core in HTTP
and JSON-ifies the wire shapes.

Endpoints:
  GET  /health           -> liveness + model/device/dim
  POST /embed            -> {texts:[...]}        -> batch dense+sparse (pre-chunked / queries)
  POST /embed_document   -> {text, id_seed?}     -> chunk via embed_core THEN embed each chunk

Run (CPU, single worker — encode() is not thread-safe, so we serialize):
  uvicorn bge_m3_service:app --host 0.0.0.0 --port 8055 --workers 1
"""
import os
import time
import uuid
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import embed_core as ec  # canonical contract: BGEM3Embedder, to_qdrant_sparse, chunk_document, names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("bge-m3-service")

# CPU by default; flip to "cuda" (and BGE_FP16=1) to run the SAME image on an NVIDIA host.
DEVICE = os.environ.get("BGE_DEVICE", "cpu")
USE_FP16 = os.environ.get("BGE_FP16", "0") == "1"
BATCH_SIZE = int(os.environ.get("BGE_BATCH_SIZE", "8"))

# Deterministic point IDs, matching ingest.py's scheme so re-ingest is idempotent
# across paths. Same namespace value as ingest.py's POINT_ID_NAMESPACE.
POINT_ID_NAMESPACE = uuid.UUID(os.environ.get(
    "BGE_ID_NAMESPACE", "6f9b1e2c-4a3d-5e6f-8a9b-0c1d2e3f4a5b"))

_model: Optional[ec.BGEM3Embedder] = None
_lock = threading.Lock()       # BGEM3FlagModel.encode is not thread-safe; serialize calls
_load_lock = threading.Lock()  # ensures the model loads exactly once (startup + first request)


def get_model() -> ec.BGEM3Embedder:
    global _model
    if _model is None:
        with _load_lock:                       # double-checked: only one thread loads
            if _model is None:
                log.info(f"loading {ec.MODEL_NAME} (device={DEVICE}, fp16={USE_FP16}) "
                         f"-- first boot downloads ~2.3GB ...")
                t0 = time.time()
                m = ec.BGEM3Embedder(device=DEVICE, use_fp16=USE_FP16)
                m.embed_documents(["warmup"], batch_size=1)  # pay init latency now
                _model = m
                log.info(f"model ready in {time.time() - t0:.1f}s")
    return _model


def _emb_to_wire(emb: dict) -> dict:
    """{'dense': np(1024), 'sparse': {tok:weight}} -> JSON-serializable wire form,
    reusing embed_core.to_qdrant_sparse so the sparse contract stays identical."""
    return {
        ec.DENSE_VECTOR: emb["dense"].tolist(),
        ec.SPARSE_VECTOR: ec.to_qdrant_sparse(emb["sparse"]).model_dump(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model in a BACKGROUND thread so the server starts serving /health
    # immediately (reporting "loading"), then flips to "ok" once weights are in.
    # This makes the Docker healthcheck work without a request having to trigger
    # the first (cold) load.
    threading.Thread(target=get_model, daemon=True, name="model-loader").start()
    yield


app = FastAPI(title="BGE-M3 Hybrid Embedder", version="1.0.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedDocRequest(BaseModel):
    text: str
    id_seed: Optional[str] = None  # e.g. the article URL -> deterministic chunk IDs


@app.get("/health")
def health():
    ready = _model is not None
    return {
        "status": "ok" if ready else "loading",
        "model": ec.MODEL_NAME,
        "device": DEVICE,
        "fp16": USE_FP16,
        "dense_dim": ec.DENSE_DIM,
        "vector_names": [ec.DENSE_VECTOR, ec.SPARSE_VECTOR],
        "model_loaded": ready,
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    """Batch-embed PRE-CHUNKED texts (or single queries). No chunking here."""
    texts = [t if isinstance(t, str) else "" for t in req.texts]
    try:
        with _lock:
            embs = get_model().embed_documents(texts, batch_size=BATCH_SIZE)
    except Exception as e:
        log.exception("embed failed")
        raise HTTPException(status_code=500, detail=f"embed failed: {e}")
    return {"results": [_emb_to_wire(e) for e in embs], "count": len(embs)}


@app.post("/embed_document")
def embed_document(req: EmbedDocRequest):
    """Chunk one document with embed_core.chunk_document (token-aware, table-safe)
    THEN embed each chunk. This is what the n8n market-trends flow calls so its
    chunking matches the watchdog exactly. Returns chunks IN ORDER."""
    text = (req.text or "").strip()
    if not text:
        return {"chunks": [], "n_chunks": 0}
    try:
        with _lock:
            model = get_model()
            chunks = ec.chunk_document(text, model.count_tokens)
            if not chunks:
                return {"chunks": [], "n_chunks": 0}
            embs = model.embed_documents(chunks, batch_size=BATCH_SIZE)
    except Exception as e:
        log.exception("embed_document failed")
        raise HTTPException(status_code=500, detail=f"embed_document failed: {e}")

    # Deterministic IDs need a stable seed. Prefer the caller's id_seed (the
    # article URL); fall back to the chunk text so a URL-less article still gets
    # a stable, idempotent ID instead of breaking the Qdrant upsert.
    out = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embs)):
        seed = req.id_seed if req.id_seed else chunk
        item = {"text": chunk, "chunk_index": idx, **_emb_to_wire(emb)}
        item["point_id"] = str(uuid.uuid5(POINT_ID_NAMESPACE, f"{seed}:{idx}"))
        out.append(item)
    return {"chunks": out, "n_chunks": len(out)}


if __name__ == "__main__":
    import uvicorn
    # lifespan loads the model in the background on startup (works under uvicorn too).
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8055")), workers=1)
