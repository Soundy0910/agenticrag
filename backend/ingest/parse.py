"""
backend/ingest/parse.py

Multi-format document parser — sits between the storage connector and the chunker.

WHY THIS LAYER EXISTS:
  The chunker (ingest/chunk.py) and embedder (ingest/embed_index.py) work on
  plain text + metadata. They should never need to know whether the source was
  a PDF, DOCX, CSV, or anything else. The parser normalises raw bytes into a
  clean, consistent ParsedDocument so every downstream stage is format-agnostic.

  Critically, the parser EXTRACTS — it does not summarise or interpret. The full
  source text is preserved verbatim. This serves two purposes:
    1. Citations: the chunk can show users exactly what the source said.
    2. Embedding-deprecation recovery: if an embedding model is retired, we can
       re-embed from the retained source text without re-fetching from the origin.

SWAPPABLE BACKEND:
  unstructured is the default — open-source, layout-aware, handles all required
  formats in one dependency. The backend is abstracted behind the ParserBackend
  Protocol so Azure Document Intelligence (or any other premium parser) can drop
  in via config without changing any caller.

  TODO: Azure Document Intelligence backend
    If RAGAS context-precision scores are weak AND the root cause is traced to
    mangled tables or mis-ordered columns in PDFs, upgrade the PDF path to
    Azure Document Intelligence (azure-ai-documentintelligence SDK).
    - Same ecosystem as Azure Blob (one cloud bill, one security boundary).
    - Activate by setting PARSER_BACKEND=azure_doc_intelligence in config/env.
    - The UnstructuredBackend and AzureDocIntelligenceBackend both implement
      ParserBackend — callers (pipeline.py) see no difference.
    - Rule: only justify the cost with RAGAS eval evidence, not upfront.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from backend.storage.base import DocumentMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """
    The normalised output of the parser layer.

    Fields
    ------
    text : str
        Full extracted text from the document. This IS the source text retained
        for embedding-deprecation recovery and citation rendering. Never
        summarised or truncated here — the chunker decides what to slice.

    doc_id : str
        Copied from DocumentMetadata.doc_id. Keeps the parsed output linked to
        its origin through the rest of the pipeline.

    structure : list[dict]
        Optional structural elements surfaced by the parser — e.g. identified
        tables, section headers, slide titles. Each entry is a small dict with
        at minimum {"type": str, "text": str}. Empty list if the backend does
        not surface structure. Lets downstream stages treat tables specially
        if they choose to.

    error : str | None
        Set to an error message if parsing failed. The pipeline should skip
        embedding this document and log the failure rather than crash.
        None means success.
    """
    text: str
    doc_id: str
    structure: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if parsing succeeded and text is non-empty."""
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------------------
# Backend protocol — the swappability contract
# ---------------------------------------------------------------------------

@runtime_checkable
class ParserBackend(Protocol):
    """
    Any object with a parse_bytes method qualifies as a ParserBackend.

    Concrete implementations:
      UnstructuredBackend  — default (open-source, handles all formats)
      AzureDocIntelligenceBackend — TODO, premium PDF/table quality

    Callers (parse() below, pipeline.py) hold a reference typed as
    ParserBackend and never import a concrete class — the backend is selected
    at startup from config.
    """
    def parse_bytes(
        self, content: bytes, file_type: str
    ) -> tuple[str, list[dict]]:
        """
        Extract text and optional structure from raw bytes.

        Parameters
        ----------
        content : bytes
            Raw file bytes as returned by DocumentSource.fetch_document().
        file_type : str
            Normalised extension without dot: 'pdf', 'docx', 'txt', etc.

        Returns
        -------
        tuple[str, list[dict]]
            (extracted_text, structure_elements)
            structure_elements may be empty if the backend doesn't surface it.
        """
        ...


# ---------------------------------------------------------------------------
# Unstructured backend (default)
# ---------------------------------------------------------------------------

class UnstructuredBackend:
    """
    Default parser backend using the open-source `unstructured` library.

    unstructured is layout-aware — it understands PDF structure, DOCX
    heading hierarchy, table cells, slide content — rather than treating
    documents as flat byte streams. The output elements carry type labels
    (Title, NarrativeText, Table, ListItem, etc.) which we expose via the
    structure field.

    Routing:
      pdf, docx, pptx, html  →  unstructured.partition (auto-routes internally)
      txt, md                →  direct UTF-8 decode (no library overhead)
      csv, xlsx              →  pandas (tabular → clean text representation)

    Azure Document Intelligence would replace the pdf path here when RAGAS
    context-precision traces weak performance to mangled tables.
    """

    def parse_bytes(
        self, content: bytes, file_type: str
    ) -> tuple[str, list[dict]]:
        if file_type in ("txt", "md"):
            return self._parse_text(content), []

        if file_type in ("csv", "xlsx"):
            return self._parse_tabular(content, file_type)

        # pdf, docx, pptx, html — all routed through unstructured
        return self._parse_unstructured(content, file_type)

    # ------------------------------------------------------------------
    # Plain text — no library needed
    # ------------------------------------------------------------------

    def _parse_text(self, content: bytes) -> str:
        """Decode with UTF-8, falling back to latin-1 for legacy docs."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")

    # ------------------------------------------------------------------
    # Tabular formats via pandas
    # ------------------------------------------------------------------

    def _parse_tabular(
        self, content: bytes, file_type: str
    ) -> tuple[str, list[dict]]:
        """
        Load CSV / XLSX into a DataFrame and produce a clean text table.

        WHY NOT RAW BYTES:
          A CSV read as raw text gives the chunker comma-separated values with
          no column context — an LLM receiving a chunk of "12345,67890,0.03"
          has no idea what those numbers mean. Converting to a markdown-style
          table or `col: value` format keeps the column headers with each row,
          making chunks self-contained and retrieval-friendly.
        """
        try:
            if file_type == "csv":
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Failed to load {file_type} as tabular data: {exc}") from exc

        # Markdown table format — column names stay with every chunk boundary
        text = df.to_markdown(index=False) if len(df) <= 500 else df.to_string(index=False)
        structure = [
            {
                "type": "Table",
                "text": f"{len(df)} rows × {len(df.columns)} columns: {list(df.columns)}",
            }
        ]
        return text, structure

    # ------------------------------------------------------------------
    # Rich formats via unstructured
    # ------------------------------------------------------------------

    def _parse_unstructured(
        self, content: bytes, file_type: str
    ) -> tuple[str, list[dict]]:
        """
        Route rich formats (PDF, DOCX, PPTX, HTML) through unstructured.

        unstructured returns a list of typed Element objects. We:
          - Concatenate their text for the main `text` field (preserving order).
          - Serialise them into the `structure` list for callers that want to
            treat headings or tables differently (e.g. parent-document chunking
            that respects section boundaries).
        """
        # Lazy import — unstructured is heavy; only pay the import cost when needed
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise ImportError(
                "unstructured is not installed. Run: pip install 'unstructured[pdf,docx,pptx]'"
            ) from exc

        # unstructured's partition() auto-routes by content_type / filename hint.
        # We pass a fake filename so it picks the right sub-partitioner.
        fake_filename = f"document.{file_type}"
        elements = partition(
            file=io.BytesIO(content),
            metadata_filename=fake_filename,
            # strategy="hi_res" would use OCR + layout model (slower, better tables).
            # "fast" is fine for text-layer PDFs; upgrade if RAGAS shows table issues.
            strategy="fast",
        )

        text = "\n\n".join(str(el) for el in elements if str(el).strip())
        structure = [
            {"type": type(el).__name__, "text": str(el)}
            for el in elements
            if str(el).strip()
        ]
        return text, structure


# ---------------------------------------------------------------------------
# Module-level default backend instance
# ---------------------------------------------------------------------------

# TODO: read PARSER_BACKEND from config.py once config.py is built (File 6).
# For now, always use UnstructuredBackend.
# When Azure Document Intelligence backend is implemented:
#   if config.PARSER_BACKEND == "azure_doc_intelligence":
#       _default_backend = AzureDocIntelligenceBackend()
#   else:
#       _default_backend = UnstructuredBackend()
_default_backend: ParserBackend = UnstructuredBackend()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse(
    content: bytes,
    metadata: DocumentMetadata,
    backend: ParserBackend | None = None,
) -> ParsedDocument:
    """
    Parse raw document bytes into a normalised ParsedDocument.

    This is the single entry point callers (pipeline.py) use. File format is
    determined entirely from metadata.file_type — callers don't pass format
    hints and don't know which backend ran.

    Parameters
    ----------
    content : bytes
        Raw file bytes from DocumentSource.fetch_document().
    metadata : DocumentMetadata
        The document's metadata record. file_type drives parser routing;
        doc_id is copied into the output.
    backend : ParserBackend | None
        Override the default backend — used in tests or when a specific
        collection is configured to use a premium parser. None → default.

    Returns
    -------
    ParsedDocument
        Always returns a ParsedDocument, even on failure (error field is set).
        Never raises — a corrupt file should not crash the ingestion pipeline.
    """
    active_backend = backend or _default_backend
    try:
        text, structure = active_backend.parse_bytes(content, metadata.file_type)
        return ParsedDocument(
            text=text,
            doc_id=metadata.doc_id,
            structure=structure,
        )
    except Exception as exc:
        logger.error(
            "Parse failed for doc_id=%r file_type=%r: %s",
            metadata.doc_id,
            metadata.file_type,
            exc,
            exc_info=True,
        )
        return ParsedDocument(
            text="",
            doc_id=metadata.doc_id,
            error=str(exc),
        )
