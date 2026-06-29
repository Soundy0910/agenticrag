"""
backend/storage/local.py

Local-folder implementation of DocumentSource.

Used for all development and testing — no cloud credentials needed.
The Azure Blob connector (storage/azure_blob.py) implements the same three
methods but against BlobServiceClient; the ingestion pipeline never knows
which one it's talking to.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.storage.base import DocumentMetadata, DocumentSource
from backend.storage._utils import file_type_from_name, is_supported


class LocalFolderSource(DocumentSource):
    """
    DocumentSource backed by a local directory on disk.

    Walks `root_folder` recursively, treating every file whose extension is in
    SUPPORTED_EXTENSIONS as one document. Hidden files (names starting with '.')
    are skipped.

    doc_id strategy
    ---------------
    doc_id is the POSIX relative path from root_folder (e.g. 'reports/2024/10k.pdf').
    This is stable: the same file gets the same ID across runs, so the pipeline
    can detect unchanged files and skip re-ingestion. fetch_document() reverses
    this by joining root_folder + relative path.

    Azure difference
    ----------------
    AzureBlobSource uses the blob name as doc_id (also a relative path within
    the container), so the ID scheme is structurally identical. The main
    difference is in watch_for_changes(): Azure uses Event Grid push events
    rather than polling — no sleep loop, zero latency on new files.
    """

    def __init__(
        self,
        root_folder: str | Path,
        collection: str,
        access_scope: list[str] | None = None,
        embedding_model: str = "text-embedding-3-small",
        poll_interval_seconds: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        root_folder : str | Path
            Absolute (or relative) path to the directory to watch.
        collection : str
            Pinecone namespace / logical bucket all documents here belong to.
        access_scope : list[str] | None
            Who can retrieve these documents. Defaults to ['public'] when
            omitted — fine for local testing; tighten for multi-user setups.
        embedding_model : str
            Embedding model that will be used for this collection. Recorded in
            every DocumentMetadata so the embed step and reindex.py know which
            model to use without guessing.
        poll_interval_seconds : float
            How often watch_for_changes() rescans the folder (seconds).
            Lower = faster detection, higher = fewer stat() calls.
        """
        self.root = Path(root_folder).resolve()
        self.collection = collection
        self.access_scope = access_scope if access_scope is not None else ["public"]
        self.embedding_model = embedding_model
        self.poll_interval = poll_interval_seconds

        if not self.root.exists():
            raise FileNotFoundError(f"root_folder does not exist: {self.root}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        """Return POSIX relative path from root — used as doc_id."""
        return path.relative_to(self.root).as_posix()

    def _abs(self, doc_id: str) -> Path:
        """Reverse of _rel: reconstruct absolute path from doc_id."""
        return self.root / doc_id

    def _is_supported(self, path: Path) -> bool:
        """True if file should be ingested (right extension, not hidden)."""
        return is_supported(path.name)

    def _metadata_for(self, path: Path) -> DocumentMetadata:
        """Build a DocumentMetadata from a file's stat info."""
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        return DocumentMetadata(
            doc_id=self._rel(path),
            source_type="local",
            collection=self.collection,
            filename=path.name,
            file_type=file_type_from_name(path.name),  # 'pdf', 'docx', …
            last_modified=mtime,
            access_scope=self.access_scope,
            embedding_model=self.embedding_model,
            extra={"absolute_path": str(path)},
        )

    # ------------------------------------------------------------------
    # DocumentSource interface
    # ------------------------------------------------------------------

    def list_documents(self) -> Iterator[DocumentMetadata]:
        """
        Yield DocumentMetadata for every supported file under root_folder.

        Lazy generator — does not buffer the full listing into memory first,
        so large directories stream through the pipeline one file at a time.

        The ingestion pipeline calls this on startup (full scan) and
        periodically as a catch-all for files that weren't surfaced by
        watch_for_changes(). Azure Blob equivalent: list_blobs() iterated
        lazily via the SDK's pager.
        """
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and self._is_supported(path):
                yield self._metadata_for(path)

    def fetch_document(self, doc_id: str) -> bytes:
        """
        Return raw bytes for the document identified by doc_id.

        doc_id is the relative POSIX path produced by list_documents(); this
        method simply reconstructs the absolute path and reads it.

        Raises FileNotFoundError if the file no longer exists — the pipeline
        interprets this as a delete and can remove the doc from the index.

        Azure equivalent: BlobClient.download_blob().readall() — also returns
        raw bytes, so the parser (ingest/parse.py) is identical for both.
        """
        path = self._abs(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        return path.read_bytes()

    def watch_for_changes(self) -> Iterator[DocumentMetadata]:
        """
        Yield DocumentMetadata for files that are new or modified since the
        last scan, polling on `poll_interval_seconds`.

        Because local folders have no native push mechanism (unlike Azure Event
        Grid or Slack webhooks), this method polls: it rescans the directory
        on each interval and diffs the current mtimes against a snapshot of
        what was seen previously.

        Only files whose last_modified has advanced (or that are entirely new)
        are yielded — unchanged files are silently skipped, keeping the
        ingestion pipeline idle-efficient.

        Azure difference
        ----------------
        AzureBlobSource.watch_for_changes() subscribes to Azure Event Grid
        blob-created / blob-modified events via an async event stream. Each
        incoming event maps directly to one DocumentMetadata yield with zero
        polling latency and no wasted stat() calls. The method signature and
        yield type are identical; only the interior differs.

        Yields
        ------
        DocumentMetadata
            One record per new or changed file. The pipeline calls
            fetch_document() on each yielded item to retrieve content and
            re-ingest.
        """
        # snapshot: doc_id → last_modified timestamp (as a float for fast compare)
        seen: dict[str, float] = {}

        # Seed the snapshot from the current state so we don't re-ingest
        # everything that already existed when the watcher starts.
        for meta in self.list_documents():
            seen[meta.doc_id] = meta.last_modified.timestamp()

        while True:
            time.sleep(self.poll_interval)
            for path in sorted(self.root.rglob("*")):
                if not (path.is_file() and self._is_supported(path)):
                    continue

                doc_id = self._rel(path)
                mtime = path.stat().st_mtime

                if seen.get(doc_id) != mtime:
                    # New file (not in seen) or modified file (mtime changed).
                    seen[doc_id] = mtime
                    yield self._metadata_for(path)
