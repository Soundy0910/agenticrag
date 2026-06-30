"""
backend/ingest/embed_index.py

Embeds chunks and upserts them into Pinecone.

Responsibilities:
  1. Embed each chunk's source_text using the collection's configured model.
  2. Upsert into Pinecone, namespaced by collection (multi-collection mechanism).
  3. Store full metadata on each vector for filtered retrieval, citations, and
     embedding-deprecation recovery.
  4. Create the Pinecone index if it doesn't exist yet.

Idempotency: chunk IDs are stable (same doc + same text → same ID), so
upsert is safe to run repeatedly — Pinecone overwrites the existing vector
rather than duplicating it.

Semantic chunking integration (File 5 note fulfilled here):
  chunk_document() accepts an optional embed_fn. This module creates that
  function from the collection's configured OpenAI embedding model and passes
  it to the chunker, enabling true semantic (similarity-based) parent splits
  instead of the paragraph-boundary fallback.
"""

import logging
import time
from typing import Callable

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

import backend.config as cfg
from backend.ingest.chunk import Chunk, ChunkConfig, chunk_document
from backend.ingest.parse import ParsedDocument
from backend.storage.base import DocumentMetadata

logger = logging.getLogger(__name__)

# Pinecone upsert batch size — stay well under the 4 MB request limit.
_UPSERT_BATCH = 100
# OpenAI embedding batch size — max 2048 inputs per request.
_EMBED_BATCH = 512


# ---------------------------------------------------------------------------
# OpenAI embedding client (module-level singleton)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not cfg.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        _openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Embedding function factory
# ---------------------------------------------------------------------------

def make_embed_fn(model: str) -> Callable[[list[str]], list[list[float]]]:
    """
    Return an embed_fn compatible with chunk_document()'s embed_fn parameter.

    The returned function:
      - Accepts a list of strings (sentences or chunk texts).
      - Returns a list of float vectors, one per input string.
      - Batches requests to stay within OpenAI's per-request input limit.

    Passing this to chunk_document() activates semantic parent splitting
    (cosine similarity between adjacent sentence embeddings) instead of the
    paragraph-boundary fallback.
    """
    client = _get_openai()

    def embed_fn(texts: list[str]) -> list[list[float]]:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        _MAX_TOKENS = 8191  # text-embedding-3-small hard limit

        def _safe(t: str) -> str:
            tokens = enc.encode(t)
            return enc.decode(tokens[:_MAX_TOKENS]) if len(tokens) > _MAX_TOKENS else t

        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = [_safe(t) for t in texts[i : i + _EMBED_BATCH]]
            response = client.embeddings.create(input=batch, model=model)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    return embed_fn


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """
    Embed a list of texts using the given model. Returns one vector per text.
    Used directly for query embedding in retrieval (hybrid.py).
    """
    return make_embed_fn(model)(texts)


# ---------------------------------------------------------------------------
# Pinecone index management
# ---------------------------------------------------------------------------

_pinecone_client: Pinecone | None = None


def _get_pinecone() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        if not cfg.PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )
        _pinecone_client = Pinecone(api_key=cfg.PINECONE_API_KEY)
    return _pinecone_client


def ensure_index(model: str | None = None) -> None:
    """
    Create the Pinecone index if it doesn't exist yet.

    Dimension is determined by the embedding model so vectors land in the right
    sized index. Called once at pipeline startup — subsequent calls are no-ops.
    """
    pc = _get_pinecone()
    dimension = cfg.get_embedding_dimension(model)
    existing = [idx.name for idx in pc.list_indexes()]

    if cfg.PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone index %r (dim=%d, metric=%s)",
            cfg.PINECONE_INDEX_NAME, dimension, cfg.PINECONE_METRIC,
        )
        pc.create_index(
            name=cfg.PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=cfg.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=cfg.PINECONE_CLOUD, region=cfg.PINECONE_REGION),
        )
        # Wait for the index to become ready (typically a few seconds).
        while not pc.describe_index(cfg.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
        logger.info("Index %r is ready.", cfg.PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _chunk_to_vector(chunk: Chunk, vector: list[float]) -> dict:
    """
    Build one Pinecone upsert record from a Chunk and its embedding.

    Metadata stored on every vector:
      - doc_id, chunk_id, parent_id, collection: pipeline plumbing
      - filename, file_type, source_type: provenance for UI / citations
      - access_scope: list of identifiers for permission-filtered retrieval
      - embedding_model: which model embedded this vector (deprecation tracking)
      - is_parent: lets the retriever distinguish child search vectors from
        parent context vectors in a mixed namespace
      - source_text: the chunk's verbatim text — used for:
          * Rendering citations in the UI (no second lookup needed)
          * Re-embedding if the model is deprecated (reindex.py reads this)
    """
    return {
        "id": chunk.chunk_id,
        "values": vector,
        "metadata": {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id or "",
            "collection": chunk.collection,
            "filename": chunk.filename,
            "file_type": chunk.file_type,
            "source_type": chunk.source_type,
            "access_scope": chunk.access_scope,
            "embedding_model": chunk.embedding_model,
            "is_parent": chunk.is_parent,
            "source_text": chunk.source_text,
        },
    }


def upsert_chunks(chunks: list[Chunk]) -> int:
    """
    Embed and upsert a list of chunks into Pinecone.

    All chunks must belong to the same collection (same namespace and same
    embedding model). The pipeline guarantees this by processing one document
    at a time — all its chunks share the collection from DocumentMetadata.

    Returns the number of vectors upserted.
    """
    if not chunks:
        return 0

    collection = chunks[0].collection
    model = cfg.get_embedding_model(collection)
    namespace = collection  # Pinecone namespace == collection name

    pc = _get_pinecone()
    index = pc.Index(cfg.PINECONE_INDEX_NAME)

    # Embed all chunk texts in one pass (batched internally).
    texts = [c.source_text for c in chunks]
    vectors = embed_texts(texts, model)

    # Upsert in batches to stay within Pinecone's request size limits.
    records = [_chunk_to_vector(c, v) for c, v in zip(chunks, vectors)]
    upserted = 0
    for i in range(0, len(records), _UPSERT_BATCH):
        batch = records[i : i + _UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=namespace)
        upserted += len(batch)

    return upserted


# ---------------------------------------------------------------------------
# Full pipeline entry point: document → chunks → Pinecone
# ---------------------------------------------------------------------------

def embed_and_index(
    parsed: ParsedDocument,
    metadata: DocumentMetadata,
    chunk_config: ChunkConfig | None = None,
) -> int:
    """
    Run the full ingest pipeline for one document: chunk → embed → upsert.

    This is the function pipeline.py calls per document. It wires the
    embedding function into chunk_document() so parent splits use real
    semantic similarity rather than the paragraph-boundary fallback.

    Parameters
    ----------
    parsed : ParsedDocument
        Output of parse.parse() for this document.
    metadata : DocumentMetadata
        Source document metadata — collection drives embedding model selection.
    chunk_config : ChunkConfig | None
        Override chunk sizing. None → defaults from config.DEFAULT_CHUNK_CONFIG.

    Returns
    -------
    int
        Number of vectors upserted to Pinecone (parents + children).
    """
    model = cfg.get_embedding_model(metadata.collection)
    embed_fn = make_embed_fn(model)
    config = chunk_config or cfg.DEFAULT_CHUNK_CONFIG

    chunks = chunk_document(parsed, metadata, config=config, embed_fn=embed_fn)
    if not chunks:
        logger.warning("No chunks produced for doc_id=%r", metadata.doc_id)
        return 0

    ensure_index(model)
    count = upsert_chunks(chunks)
    logger.info(
        "Upserted %d vectors for doc_id=%r into namespace=%r",
        count, metadata.doc_id, metadata.collection,
    )
    return count
