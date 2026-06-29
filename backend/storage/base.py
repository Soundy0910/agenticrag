"""
backend/storage/base.py

Abstract foundation for all document sources in the Agentic RAG platform.

WHY THIS ABSTRACTION EXISTS:
  The pipeline (parse → chunk → embed → index) should never know whether documents
  come from a local folder, Azure Blob, Slack, Monday.com, or a direct upload.
  By funneling every source through DocumentSource, all downstream code is
  identical regardless of backend — swapping storage is a config change, not a
  rewrite. This is the "design cloud-shaped, build locally first" principle from
  ARCHITECTURE.md made concrete.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass
class DocumentMetadata:
    """
    Canonical metadata record for every document in the system.

    Every document — regardless of where it lives (local disk, Azure Blob,
    Slack thread) — is described by this single structure before it enters the
    ingestion pipeline. Rich metadata here means: filtering, citations,
    permission-aware retrieval, and embedding-deprecation recovery all work
    without touching the pipeline code.

    Fields
    ------
    doc_id : str
        Globally unique identifier for this document. Stable across ingestion
        runs so updates and deletes can target a specific doc in the vector
        index without touching others.

    source_type : str
        Which backend produced this document. One of:
        'azure_blob' | 'slack' | 'monday' | 'upload' | 'local'
        Lets the pipeline log provenance and lets the UI show a source icon.

    collection : str
        The namespace / logical bucket this document belongs to — maps 1:1 to
        a Pinecone namespace. Scoped retrieval (legal question → legal
        namespace, never finance chunks) is enforced here at metadata level.
        Equivalent to a NotebookLM "notebook."

    filename : str
        Original filename, preserved for UI display and citation rendering.

    file_type : str
        Extension / MIME hint: 'pdf', 'docx', 'txt', 'md', 'csv', etc.
        The parser (ingest/parse.py) uses this to route to the right parser
        backend without re-detecting the type.

    last_modified : datetime
        Source-side last-modified timestamp. Used by watch_for_changes() to
        detect new or updated documents without rescanning everything.

    access_scope : list[str]
        Identifiers (user IDs, role names, org slugs) that are allowed to
        retrieve chunks from this document. Stored as Pinecone metadata so
        query-time filtering can enforce permissions WITHOUT a separate ACL
        lookup — the vector DB does the work inline.

        WHY THIS MATTERS: Pinecone account-level RBAC is not your permission
        system. Per-doc access control lives here, in your own metadata, so
        you can model "this Slack thread is only visible to team-A" without
        any additional infra.

    embedding_model : str
        The embedding model used to embed this document's collection, e.g.
        'text-embedding-3-small' or 'voyage-finance-2'.

        WHY PER-COLLECTION (not per-doc or per-query): you cannot mix
        embedding spaces in one searchable index — a query must be embedded
        with the same model as the docs it searches. Recording the model here
        means: (a) the embed step reads it rather than guessing, and (b) if
        a model is deprecated, reindex.py can re-embed from retained source
        text into a new namespace using the correct replacement model.

    extra : dict
        Source-specific fields that don't belong in the canonical schema.
        Azure Blob might store {'container': '...', 'blob_etag': '...'};
        Slack might store {'channel_id': '...', 'thread_ts': '...'}.
        Using a dict keeps the dataclass stable across connectors while still
        letting each connector attach what it needs.
    """

    doc_id: str
    source_type: str  # 'azure_blob' | 'slack' | 'monday' | 'upload' | 'local'
    collection: str
    filename: str
    file_type: str
    last_modified: datetime
    access_scope: list[str]
    embedding_model: str
    extra: dict = field(default_factory=dict)


class DocumentSource(ABC):
    """
    Abstract interface every storage backend must implement.

    WHY AN ABSTRACT CLASS (not duck-typing / protocol):
      We want Python to enforce the contract at class-definition time, not at
      call time. Any connector that forgets to implement a method raises
      TypeError on instantiation — a loud, early failure rather than a silent
      runtime surprise mid-ingestion.

    Concrete implementations live in:
      storage/local.py      — local folder (used for all dev/testing)
      storage/azure_blob.py — real Azure Blob (the production connector)
      storage/stubs.py      — Slack / Monday (interface-conformant stubs)

    The ingestion pipeline (ingest/pipeline.py) holds a reference typed as
    DocumentSource and calls these three methods. It never imports a concrete
    class — the connector is injected at startup from config.
    """

    @abstractmethod
    def list_documents(self) -> Iterator[DocumentMetadata]:
        """
        Yield a DocumentMetadata record for every document currently in this
        source.

        Called by the ingestion pipeline on startup (full scan) and
        periodically to catch documents that weren't surfaced by watch events.

        Implementation notes:
          - Local folder: os.walk / glob the target directory, stat each file
            for last_modified, yield one metadata per file.
          - Azure Blob: call BlobServiceClient.list_blobs(), map each
            BlobProperties entry to DocumentMetadata (etag → doc_id,
            content_type → file_type, etc.).
          - The method should be lazy (a generator / Iterator) so large
            collections don't need to be fully loaded into memory before
            processing begins.

        Yields
        ------
        DocumentMetadata
            One record per document. Does NOT fetch content — content is
            retrieved on demand via fetch_document().
        """
        ...

    @abstractmethod
    def fetch_document(self, doc_id: str) -> bytes:
        """
        Return the raw bytes of the document identified by doc_id.

        Called by the ingestion pipeline after list_documents() surfaces a new
        or updated document. The pipeline passes these bytes to the parser
        (ingest/parse.py), which normalises them to plain text regardless of
        file type.

        Implementation notes:
          - Local folder: open(path, 'rb').read() where path is derived from
            doc_id (e.g. the absolute file path used as the id).
          - Azure Blob: BlobClient.download_blob().readall() — returns raw
            bytes identically, so the parser doesn't need to know the source.
          - Should raise FileNotFoundError (or a subclass) if the document
            no longer exists — the pipeline can then treat it as a delete.

        Parameters
        ----------
        doc_id : str
            The same doc_id that was yielded by list_documents().

        Returns
        -------
        bytes
            Raw file content. The parser handles decoding/parsing.
        """
        ...

    @abstractmethod
    def watch_for_changes(self) -> Iterator[DocumentMetadata]:
        """
        Yield DocumentMetadata for documents that have been added or modified
        since the last scan, enabling near-real-time incremental ingestion.

        This is the event-driven half of ingestion. Rather than re-listing the
        entire source on every cycle, connectors that support change events
        (Azure Event Grid, Slack webhooks, filesystem inotify) push only the
        deltas. Connectors without native push can poll and diff against a
        stored last-seen state.

        Implementation notes:
          - Local folder: poll on an interval, compare current file mtimes
            against a cached snapshot; yield only changed entries.
          - Azure Blob: subscribe to Azure Event Grid blob-created /
            blob-modified events; each event maps to one DocumentMetadata yield.
          - Slack / Monday stubs: yield nothing for now; the real
            implementation would listen to their respective webhook streams.
          - The pipeline keeps this iterator alive in a background task;
            each yielded item triggers a single-document ingest without
            disrupting in-flight queries.

        Yields
        ------
        DocumentMetadata
            One record per new or changed document. The pipeline calls
            fetch_document() on each to get content and re-ingest.
        """
        ...
