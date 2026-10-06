#!/usr/bin/env python3
"""
embed_core.py — shared BGE-M3 hybrid embedding contract for the market-signals stack.

Single source of truth for everything that MUST match between every reader and
writer of a hybrid collection (market-trends, enterprise-context, cio-personas):
  - the named-vector names ("dense" / "sparse") and dense dim (1024, Cosine)
  - BGE-M3 sparse-weight -> Qdrant SparseVector conversion (NO IDF modifier;
    BGE-M3 emits learned lexical weights, so plain dot-product is correct)
  - the token-aware, table-preserving markdown chunker
  - the hybrid-collection DDL (named vectors + payload indexes + alias)

Both ingest.py (enterprise-context) and embed_market_trends.py (market-trends)
import from here so the two pipelines can't silently drift. FlagEmbedding is
imported lazily inside BGEM3Embedder so this module loads fine without the model
(or the 2.3GB weights) present -- useful for DDL-only and chunker-only callers.
"""

import re
from qdrant_client import QdrantClient, models

# --- The contract (must match across all readers/writers) -------------------
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DENSE_DIM = 1024                 # BGE-M3 dense output
MODEL_NAME = "BAAI/bge-m3"

# Chunking defaults are TOKEN budgets now -- BGE-M3's ceiling is 8192, not the
# 512 that forced the old char-based band-aids. ~1024-token chunks give good
# retrieval granularity while leaving whole tables intact.
CHUNK_TARGET_TOKENS = 1024
CHUNK_OVERLAP_TOKENS = 128
CHUNK_MAX_TOKENS = 8000          # < model's 8192


# ============================================================================
# Embedder  (CPU; loaded ONCE per process, reused for every file)
# ============================================================================
class BGEM3Embedder:
    """BGE-M3 dense+sparse via FlagEmbedding. CPU by default (fp16 is a GPU win;
    on CPU it's typically slower/unsupported, so use_fp16=False).

    FlagEmbedding has churned on the device kwarg across versions (`device=` vs
    `devices=`); the try/except covers both. If it still errors on your installed
    version, that's the line to adjust.
    """

    def __init__(self, model_name=MODEL_NAME, device="cpu", use_fp16=False):
        from FlagEmbedding import BGEM3FlagModel  # lazy: heavy import + weights
        try:
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        self.tokenizer = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts, batch_size=8):
        """list[str] -> list[{'dense': np.ndarray(1024), 'sparse': {tok_id:weight}}]."""
        out = self.model.encode(
            texts, batch_size=batch_size, max_length=CHUNK_MAX_TOKENS,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        return [{"dense": d, "sparse": dict(s)}
                for d, s in zip(out["dense_vecs"], out["lexical_weights"])]

    def embed_query(self, text: str):
        """Single text. NO prefix -- BGE-M3 encodes queries like passages."""
        return self.embed_documents([text])[0]


# ============================================================================
# Sparse conversion + named-vector assembly
# ============================================================================
def to_qdrant_sparse(lexical_weights) -> models.SparseVector:
    """BGE-M3 lexical_weights {token_id -> weight} -> Qdrant SparseVector.
    Keys may be str or int; cast to int. Drop non-positive weights defensively."""
    indices, values = [], []
    for k, v in lexical_weights.items():
        w = float(v)
        if w > 0.0:
            indices.append(int(k))
            values.append(w)
    return models.SparseVector(indices=indices, values=values)


def named_vector(emb) -> dict:
    """{'dense': np, 'sparse': {...}} -> the vector= dict for a PointStruct."""
    return {
        DENSE_VECTOR: emb["dense"].tolist(),
        SPARSE_VECTOR: to_qdrant_sparse(emb["sparse"]),
    }


# ============================================================================
# Chunker — token-aware, keeps markdown tables & code blocks intact
# (the clean-.md lane uses this; the raw-doc lane uses unstructured's chunker)
# ============================================================================
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
# Sentence boundary for splitting oversized PROSE blocks (e.g. a wall-of-text
# news body that arrives as one paragraph). Rough on abbreviations, but that's
# harmless: over-splits are re-packed back up to `target` by the packer below.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _atomic_blocks(text):
    """Split into blocks that must not be broken mid-way: fenced code blocks,
    contiguous markdown tables, and paragraphs."""
    lines = text.splitlines()
    blocks, buf = [], []
    in_fence = False
    in_table = False

    def flush():
        if buf:
            blocks.append("\n".join(buf).strip("\n"))
            buf.clear()

    for line in lines:
        if line.lstrip().startswith("```"):
            if not in_fence:
                flush(); in_fence = True; buf.append(line)
            else:
                buf.append(line); in_fence = False; flush()
            continue
        if in_fence:
            buf.append(line); continue

        is_table = bool(_TABLE_LINE.match(line))
        if is_table and not in_table:
            flush(); in_table = True; buf.append(line); continue
        if in_table and not is_table:
            flush(); in_table = False
            if line.strip():
                buf.append(line)
            continue
        if line.strip() == "" and not in_table:
            flush()
        else:
            buf.append(line)
    flush()
    return [b for b in blocks if b.strip()]


def chunk_document(text, count_tokens,
                   target=CHUNK_TARGET_TOKENS,
                   overlap=CHUNK_OVERLAP_TOKENS,
                   hard_max=CHUNK_MAX_TOKENS):
    """Greedily pack atomic blocks into <=target-token chunks with a trailing-
    block token overlap. Blocks larger than hard_max are window-split as a last
    resort. Tables/code fences are never split across chunks."""
    blocks = _atomic_blocks(text)
    chunks, cur, cur_tok = [], [], 0

    def emit():
        if cur:
            chunks.append("\n\n".join(cur))

    for block in blocks:
        bt = count_tokens(block)
        if bt > hard_max:
            emit(); cur.clear(); cur_tok = 0
            chunks.extend(_window_split(block, count_tokens, hard_max))
            continue
        if bt > target and not _is_atomic(block):
            # Oversized PROSE (not a table/code block): sentence-split into
            # ~target-token chunks with overlap instead of emitting one fat
            # vector. Fold any small pending prose (e.g. a title) into the front
            # so we don't leave a thin title-only chunk -- but never fold pending
            # atomic blocks (a table), which must stay whole.
            if cur and not any(_is_atomic(b) for b in cur):
                block = "\n\n".join(cur) + "\n\n" + block
                cur.clear(); cur_tok = 0
            else:
                emit(); cur.clear(); cur_tok = 0
            chunks.extend(_chunk_prose(block, count_tokens, target, overlap))
            continue
        if cur_tok + bt > target and cur:
            emit()
            tail, tail_tok = [], 0
            for b in reversed(cur):
                t = count_tokens(b)
                if tail_tok + t > overlap:
                    break
                tail.insert(0, b); tail_tok += t
            cur[:] = tail; cur_tok = tail_tok
        cur.append(block); cur_tok += bt
    emit()
    return chunks


def _window_split(block, count_tokens, hard_max):
    """Last-resort splitter for a single oversized block; splits on lines
    (keeps table rows whole) into <= hard_max windows with small line overlap."""
    lines = block.splitlines()
    out, cur, cur_tok = [], [], 0
    for ln in lines:
        t = count_tokens(ln)
        if cur_tok + t > hard_max and cur:
            out.append("\n".join(cur))
            keep = cur[-2:] if len(cur) >= 2 else cur[:]
            cur = list(keep); cur_tok = sum(count_tokens(x) for x in keep)
        cur.append(ln); cur_tok += t
    if cur:
        out.append("\n".join(cur))
    return out


def _is_atomic(block):
    """True for blocks that must never be sentence-split: fenced code and
    markdown tables. (Prose blocks return False and may be sentence-chunked.)"""
    if block.lstrip().startswith("```"):
        return True
    return any(_TABLE_LINE.match(ln) for ln in block.splitlines())


def _word_window(text, count_tokens, target):
    """Fallback for a single 'sentence' that itself exceeds target (no/few
    punctuation): split on words into <=target-token windows."""
    out, cur, cur_tok = [], [], 0
    for w in text.split():
        t = count_tokens(w)
        if cur_tok + t > target and cur:
            out.append(" ".join(cur)); cur, cur_tok = [], 0
        cur.append(w); cur_tok += t
    if cur:
        out.append(" ".join(cur))
    return out


def _chunk_prose(block, count_tokens, target, overlap):
    """Split an oversized PROSE block into <=target-token chunks on sentence
    boundaries, carrying a trailing-sentence token overlap between chunks.
    Sentences are re-joined with single spaces (natural prose), so over-splitting
    on abbreviations doesn't change the stored text."""
    sentences = [s for s in _SENT_SPLIT.split(block.strip()) if s.strip()]
    chunks, cur, cur_tok = [], [], 0
    for sent in sentences:
        st = count_tokens(sent)
        if st > target:                      # a single monster sentence
            if cur:
                chunks.append(" ".join(cur)); cur, cur_tok = [], 0
            chunks.extend(_word_window(sent, count_tokens, target))
            continue
        if cur_tok + st > target and cur:
            chunks.append(" ".join(cur))
            tail, tail_tok = [], 0           # sentence-level overlap into next chunk
            for s in reversed(cur):
                t = count_tokens(s)
                if tail_tok + t > overlap:
                    break
                tail.insert(0, s); tail_tok += t
            cur, cur_tok = list(tail), tail_tok
        cur.append(sent); cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ============================================================================
# Hybrid-collection DDL  (named dense+sparse vectors + payload indexes + alias)
# ============================================================================
def _ensure_alias(client, physical, alias, log=print):
    existing = [a.alias_name for a in client.get_aliases().aliases]
    actions = []
    if alias in existing:
        actions.append(models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name=alias)))
    actions.append(models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=physical, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=actions)


def provision_hybrid_collection(client, physical, alias, payload_indexes,
                                recreate=False, exist_ok=True, log=print):
    """Create `physical` with named dense (1024, Cosine) + sparse vectors and the
    given payload indexes, then point `alias` at it.

      recreate=True   -> drop an existing physical collection first, then rebuild
      exist_ok=True   -> if it already exists, just ensure the alias (daemon use:
                         stand it up on first boot, no-op thereafter)
      exist_ok=False  -> if it already exists, raise (standalone DDL-script use)
    """
    exists = client.collection_exists(physical)
    if exists and recreate:
        log(f"- dropping existing '{physical}'")
        client.delete_collection(physical)
        exists = False
    if exists:
        if not exist_ok:
            raise RuntimeError(
                f"collection '{physical}' already exists; pass recreate=True to rebuild")
        log(f"- '{physical}' exists; ensuring alias '{alias}'")
        _ensure_alias(client, physical, alias, log)
        return

    log(f"- creating '{physical}'  (dense {DENSE_DIM} Cosine + sparse, no IDF)")
    client.create_collection(
        collection_name=physical,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
        },
    )
    for field, schema in payload_indexes.items():
        client.create_payload_index(collection_name=physical, field_name=field, field_schema=schema)
        log(f"    index: {field}")
    _ensure_alias(client, physical, alias, log)
    log(f"  ok: alias '{alias}' -> '{physical}'")
