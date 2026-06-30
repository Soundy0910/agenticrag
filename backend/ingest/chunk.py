"""
backend/ingest/chunk.py

Semantic chunker with parent-document retrieval.

WHY TWO LEVELS (parent + child):
  A single chunk size is a forced tradeoff: small chunks give precise retrieval
  but lose context; large chunks give context but match noisily. Parent-document
  retrieval resolves this by keeping both:
    - Child chunks (~300 tokens) are what gets embedded and searched — small
      enough for high-precision vector matching.
    - Parent chunks (~1,750 tokens) are what gets returned to the LLM — large
      enough to contain a complete idea or section.
  Each child carries a parent_id so the retriever can fetch the parent at
  query time. The LLM never reads the tiny child; it reads the parent that
  child was extracted from.

WHY SEMANTIC SPLITTING (not fixed-size):
  Fixed character/token splits are blind to content: they cheerfully slice a
  table header from its rows, or split a sentence mid-clause. Semantic
  splitting identifies meaning-shift boundaries (via cosine similarity drops
  between adjacent sentence embeddings) and cuts there instead. This keeps
  each parent chunk thematically coherent, which improves retrieval precision.

  Cost note: semantic splitting embeds every sentence at ingest time. At demo
  scale this is negligible; for large corpora, benchmark vs. recursive
  splitting and decide if the quality gain justifies the cost.

EMBEDDING-DEPRECATION INSURANCE:
  Every Chunk retains its source_text verbatim. If the embedding model for a
  collection is deprecated, reindex.py re-embeds from source_text into a new
  namespace — no need to re-fetch from the origin storage.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingest.parse import ParsedDocument
from backend.storage.base import DocumentMetadata


# ---------------------------------------------------------------------------
# Configuration (eval-tunable, never hard-coded)
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """
    All sizing parameters in one place so RAGAS evals can tune them without
    touching implementation code.

    child_tokens : int
        Target token count for child chunks. Small = precise retrieval.
        Architecture target: 256–400 tokens. Default: 350.

    parent_tokens : int
        Target token count for parent chunks. Large = rich LLM context.
        Architecture target: 1,500–2,000 tokens. Default: 1,750.

    overlap_pct : float
        Fraction of child_tokens to overlap between adjacent children.
        Architecture target: 10–15%. Default: 0.12 (12%).
        Ensures a match near a boundary retains its lead-in context.

    semantic_threshold : float
        Cosine similarity below which adjacent sentences are considered a
        meaning-shift boundary (used only when embed_fn is provided).
        Lower = fewer cuts (larger parents); higher = more cuts (smaller parents).

    encoding_name : str
        tiktoken BPE encoding used for token counting. cl100k_base matches
        text-embedding-3-small and GPT-4 — keeps token estimates consistent
        across the pipeline.
    """
    child_tokens: int = 350
    parent_tokens: int = 1750
    overlap_pct: float = 0.12
    semantic_threshold: float = 0.5
    encoding_name: str = "cl100k_base"
    min_parent_chars: int = 80


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single chunk ready for embedding and Pinecone upsert.

    chunk_id : str
        Deterministic, unique ID. Format:
          parent → "{doc_id}__p{n}"
          child  → "{doc_id}__p{n}__c{m}"
        Deterministic means re-running chunking on unchanged text produces
        identical IDs — safe to re-ingest idempotently.

    parent_id : str | None
        For child chunks: the chunk_id of the parent this child was extracted
        from. The retriever fetches the parent by this ID to return to the LLM.
        For parent chunks: None (they ARE the parent).

    doc_id : str
        Copied from DocumentMetadata — links back to the origin document.

    source_text : str
        The chunk's own text, verbatim. Used for:
          - Rendering citations in the UI.
          - Re-embedding if the model is deprecated (embedding-deprecation insurance).

    is_parent : bool
        True for parent chunks (stored but not directly embedded for retrieval).
        False for child chunks (embedded and indexed for search).

    collection, source_type, filename, file_type, embedding_model, access_scope :
        Copied from DocumentMetadata so the Pinecone upsert has all filterable
        fields in one place without needing to re-fetch metadata.

    extra : dict
        Passthrough of DocumentMetadata.extra plus any chunk-level additions.
    """
    chunk_id: str
    parent_id: str | None
    doc_id: str
    source_text: str
    is_parent: bool
    collection: str
    source_type: str
    filename: str
    file_type: str
    embedding_model: str
    access_scope: list[str]
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str, encoding_name: str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")



def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.

    Crude but dependency-free and sufficient for the semantic similarity
    step — we only need sentence-level granularity to detect meaning shifts,
    not perfect sentence parsing.
    """
    # Preserve paragraph breaks as hard split points first.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        parts = _SENTENCE_RE.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


# ---------------------------------------------------------------------------
# Parent-level semantic splitting
# ---------------------------------------------------------------------------

def _merge_short_parents(parents: list[str], min_chars: int) -> list[str]:
    """
    Merge any parent shorter than min_chars into its successor.

    Resumes and structured docs produce standalone heading/date lines like
    "EXPERIENCE" (10 chars) or "Jan 2026 – Present" (18 chars) as their own
    parent chunks. These have near-meaningless embeddings and are never
    retrieved usefully. Merging them forward keeps the heading attached to
    the first content block it introduces.

    min_chars comes from ChunkConfig.min_parent_chars so it is tunable
    per-collection via config.py without touching this file.

    Applied after both the embedding and fallback split paths.
    """
    if not parents:
        return parents
    result: list[str] = []
    carry = ""
    for p in parents:
        combined = (carry + "\n\n" + p) if carry else p
        if len(combined) < min_chars:
            carry = combined
        else:
            result.append(combined)
            carry = ""
    if carry:
        if result:
            result[-1] = result[-1] + "\n\n" + carry
        else:
            result.append(carry)
    return result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recursive_split(text: str, max_tokens: int, encoding_name: str) -> list[str]:
    """Split text into pieces each under max_tokens using the recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # approx chars
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _semantic_parent_splits(
    text: str,
    config: ChunkConfig,
    embed_fn: Callable[[list[str]], list[list[float]]] | None,
) -> list[str]:
    """
    Split text into parent-sized segments.

    WITH embed_fn (true semantic splitting):
      1. Segment text into sentences.
      2. Embed all sentences in one batch call (cheap: short strings).
      3. Compute cosine similarity between adjacent sentence pairs.
      4. Identify valleys: points where similarity < semantic_threshold.
         These are meaning-shift boundaries.
      5. Merge sentences between boundaries into parent chunks, respecting
         the parent_tokens limit.

    WITHOUT embed_fn (paragraph-boundary fallback):
      Split on double newlines (paragraph breaks) and merge small paragraphs
      until the parent_tokens limit is reached. Less precise but zero cost
      and works correctly for documents with clear paragraph structure.

    In both cases, a parent chunk that grows beyond parent_tokens is cut even
    if no semantic boundary was found — the token limit is a hard cap.
    """
    enc_name = config.encoding_name
    max_tokens = config.parent_tokens

    if embed_fn is not None:
        return _semantic_split_with_embeddings(text, config, embed_fn)

    # Fallback: paragraph-boundary merging
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para, enc_name)
        if current_tokens + para_tokens > max_tokens and current_parts:
            parents.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0
        # If a single paragraph already exceeds the limit (e.g. a CSV markdown
        # table with \n between rows but no \n\n), split it with the recursive
        # splitter rather than emitting one giant chunk that would exceed the
        # embedding model's 8192-token input limit.
        if para_tokens > max_tokens:
            parents.extend(_recursive_split(para, max_tokens, enc_name))
            continue
        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        parents.append("\n\n".join(current_parts))

    return _merge_short_parents(parents, config.min_parent_chars)


def _semantic_split_with_embeddings(
    text: str,
    config: ChunkConfig,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[str]:
    """
    True semantic splitting via embedding similarity valleys.

    Called only when an embedding function is provided. The embed_fn receives
    a list of strings and returns a list of embedding vectors (list[list[float]]).
    Compatible with OpenAI's client.embeddings.create batch interface wrapped
    as a callable.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    vectors = embed_fn(sentences)

    # Find meaning-shift boundaries: adjacent pairs with low cosine similarity
    boundaries: set[int] = set()
    for i in range(len(vectors) - 1):
        sim = _cosine_similarity(vectors[i], vectors[i + 1])
        if sim < config.semantic_threshold:
            boundaries.add(i + 1)  # cut before sentence i+1

    # Merge sentences into parent chunks, respecting the token limit
    parents: list[str] = []
    current_sents: list[str] = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        sent_tokens = _count_tokens(sent, config.encoding_name)
        at_boundary = i in boundaries
        over_limit = current_tokens + sent_tokens > config.parent_tokens

        if (at_boundary or over_limit) and current_sents:
            parents.append(" ".join(current_sents))
            current_sents = []
            current_tokens = 0

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        parents.append(" ".join(current_sents))

    return _merge_short_parents(parents, config.min_parent_chars)


# ---------------------------------------------------------------------------
# Child splitting (sliding window with overlap)
# ---------------------------------------------------------------------------

def _child_chunks_from_parent(
    parent_text: str, config: ChunkConfig
) -> list[str]:
    """
    Split a parent chunk into overlapping child chunks using LangChain's
    RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter respects natural boundaries in priority
    order: paragraph → sentence → word → character. This means it won't
    split a sentence mid-word even when character counts don't divide evenly.

    Token → character conversion: tiktoken encodes ~4 characters per token on
    average for English prose. We multiply by 4 to get approximate char limits,
    which is accurate enough for sizing — exact token counts come from the
    embedding step, not here.
    """
    chars_per_token = 4  # rough average for English prose
    chunk_size = config.child_tokens * chars_per_token
    overlap = int(config.child_tokens * config.overlap_pct) * chars_per_token

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return splitter.split_text(parent_text)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, parent_idx: int, child_idx: int | None = None) -> str:
    """
    Deterministic chunk ID. Stable across re-ingestion of unchanged text so
    the Pinecone upsert is idempotent (same ID = overwrite, not duplicate).

    We hash the doc_id to keep IDs short and safe for Pinecone vector IDs
    (which have a 512-byte limit on the ID string).
    """
    doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
    if child_idx is None:
        return f"{doc_hash}__p{parent_idx}"
    return f"{doc_hash}__p{parent_idx}__c{child_idx}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_document(
    parsed: ParsedDocument,
    metadata: DocumentMetadata,
    config: ChunkConfig | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[Chunk]:
    """
    Chunk a ParsedDocument into parent + child Chunk objects ready for upsert.

    Returns both parents and children in a single flat list. The caller
    (ingest/pipeline.py and ingest/embed_index.py) separates them by
    checking chunk.is_parent:
      - Children (is_parent=False) are embedded and indexed in Pinecone.
      - Parents (is_parent=True) are stored (e.g. in a document store or as
        Pinecone vectors with a special namespace) so the retriever can fetch
        them by parent_id at query time.

    Parameters
    ----------
    parsed : ParsedDocument
        Output of parse.parse() — contains the full extracted text.
    metadata : DocumentMetadata
        Source document's metadata. All fields are copied into every Chunk
        so Pinecone has filterable metadata without a separate lookup.
    config : ChunkConfig | None
        Sizing parameters. None → defaults from ChunkConfig().
    embed_fn : Callable | None
        Optional embedding function for semantic parent splitting.
        Signature: embed_fn(texts: list[str]) -> list[list[float]].
        When None, paragraph-boundary splitting is used instead.
        Pass the collection's configured embedding model here once
        embed_index.py is built (File 6).

    Returns
    -------
    list[Chunk]
        Parents first, then children (in document order). Empty list if
        parsed.text is empty.
    """
    if not parsed.ok:
        return []

    cfg = config or ChunkConfig()

    parent_texts = _semantic_parent_splits(parsed.text, cfg, embed_fn)

    chunks: list[Chunk] = []

    for p_idx, parent_text in enumerate(parent_texts):
        parent_id = _make_chunk_id(metadata.doc_id, p_idx)

        parent_chunk = Chunk(
            chunk_id=parent_id,
            parent_id=None,
            doc_id=metadata.doc_id,
            source_text=parent_text,
            is_parent=True,
            collection=metadata.collection,
            source_type=metadata.source_type,
            filename=metadata.filename,
            file_type=metadata.file_type,
            embedding_model=metadata.embedding_model,
            access_scope=metadata.access_scope,
            extra={**metadata.extra, "parent_index": p_idx},
        )
        chunks.append(parent_chunk)

        child_texts = _child_chunks_from_parent(parent_text, cfg)
        for c_idx, child_text in enumerate(child_texts):
            child_chunk = Chunk(
                chunk_id=_make_chunk_id(metadata.doc_id, p_idx, c_idx),
                parent_id=parent_id,
                doc_id=metadata.doc_id,
                source_text=child_text,
                is_parent=False,
                collection=metadata.collection,
                source_type=metadata.source_type,
                filename=metadata.filename,
                file_type=metadata.file_type,
                embedding_model=metadata.embedding_model,
                access_scope=metadata.access_scope,
                extra={**metadata.extra, "parent_index": p_idx, "child_index": c_idx},
            )
            chunks.append(child_chunk)

    return chunks
