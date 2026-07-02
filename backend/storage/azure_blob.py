"""
backend/storage/azure_blob.py

Azure Blob Storage implementation of DocumentSource.

Mirrors LocalFolderSource structurally — same three methods, same doc_id
scheme (blob name = relative path within container), same bytes return type
from fetch_document(). The ingestion pipeline cannot tell the difference.

Key differences from local.py:
  - Client:         ContainerClient from azure-storage-blob SDK
  - doc_id:         blob name (set by Azure, already a relative path)
  - last_modified:  BlobProperties.last_modified (set by Azure on upload)
  - watch_for_changes(): poll-and-diff now; TODO replace with Event Grid push
"""

import os
import time
from datetime import datetime, timezone
from typing import Iterator

from azure.storage.blob import BlobProperties, ContainerClient
from dotenv import load_dotenv

from backend.storage.base import DocumentMetadata, DocumentSource
from backend.storage._utils import file_type_from_name, is_supported

load_dotenv()  # reads .env in the project root so credentials stay out of code


class AzureBlobSource(DocumentSource):
    """
    DocumentSource backed by an Azure Blob Storage container.

    Credentials are read from environment variables (never hard-coded):
      AZURE_STORAGE_CONNECTION_STRING — full connection string from the Azure portal
      AZURE_STORAGE_CONTAINER        — container name to read documents from

    doc_id strategy
    ---------------
    doc_id is the blob name exactly as Azure stores it (e.g. 'filings/2024/10k.pdf').
    This is structurally identical to LocalFolderSource's relative-path IDs, so the
    ingestion pipeline treats both connectors the same. fetch_document() passes the
    doc_id directly to BlobClient — no path reconstruction needed.

    Local difference
    ----------------
    LocalFolderSource derives doc_id by computing a relative path from root_folder.
    Here Azure assigns the name; we just use it as-is.
    """

    def __init__(
        self,
        collection: str,
        access_scope: list[str] | None = None,
        embedding_model: str = "text-embedding-3-small",
        poll_interval_seconds: float = 60.0,
        blob_prefix: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        collection : str
            Pinecone namespace / logical bucket these blobs belong to.
        access_scope : list[str] | None
            Who can retrieve documents from this container.
            Defaults to ['public'] — tighten for multi-user setups.
        embedding_model : str
            Embedding model for this collection, recorded in every
            DocumentMetadata so the embed step and reindex.py stay consistent.
        poll_interval_seconds : float
            How often watch_for_changes() rescans the container (seconds).
            Higher default than local (60 s vs 30 s) because list_blobs()
            is a network call, not a local stat().
        """
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("AZURE_STORAGE_CONTAINER")

        if not conn_str:
            raise EnvironmentError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "Add it to your .env file or environment."
            )
        if not container_name:
            raise EnvironmentError(
                "AZURE_STORAGE_CONTAINER is not set. "
                "Add it to your .env file or environment."
            )

        self.client: ContainerClient = ContainerClient.from_connection_string(
            conn_str, container_name
        )
        self.collection = collection
        self.access_scope = access_scope if access_scope is not None else ["public"]
        self.embedding_model = embedding_model
        self.poll_interval = poll_interval_seconds
        self.blob_prefix = blob_prefix  # when set, only blobs under this prefix are yielded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metadata_from_props(self, props: BlobProperties) -> DocumentMetadata:
        """
        Build a DocumentMetadata from Azure BlobProperties.

        BlobProperties is what list_blobs() yields — it contains the blob name,
        last_modified datetime, content_type, size, etc. We don't need to fetch
        content here; that happens on demand via fetch_document().

        Local equivalent: _metadata_for(path) reads Path.stat() instead.
        """
        name: str = props["name"]  # blob name is a str key in BlobProperties dict
        last_modified: datetime = props["last_modified"]

        # Azure returns last_modified as a timezone-aware datetime; normalise to UTC.
        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)

        return DocumentMetadata(
            doc_id=name,               # blob name == doc_id (stable, set by Azure)
            source_type="azure_blob",
            collection=self.collection,
            filename=name.split("/")[-1],  # final component, e.g. '10k.pdf'
            file_type=file_type_from_name(name),
            last_modified=last_modified,
            access_scope=self.access_scope,
            embedding_model=self.embedding_model,
            extra={"blob_name": name, "container": self.client.container_name},
        )

    # ------------------------------------------------------------------
    # DocumentSource interface
    # ------------------------------------------------------------------

    def list_documents(self) -> Iterator[DocumentMetadata]:
        """
        Yield DocumentMetadata for every supported blob in the container.

        list_blobs() returns an ItemPaged — a lazy pager from the Azure SDK.
        Iterating it issues requests in pages (5000 items/page by default) so
        large containers don't require fetching all metadata at once.

        Local equivalent: Path.rglob("*") iterated lazily.
        """
        for props in self.client.list_blobs(name_starts_with=self.blob_prefix):
            if is_supported(props["name"]):
                yield self._metadata_from_props(props)

    def fetch_document(self, doc_id: str) -> bytes:
        """
        Download a blob by name and return its raw bytes.

        doc_id is the blob name produced by list_documents(); Azure's SDK
        accepts it directly — no path reconstruction needed (unlike the local
        connector which joins root_folder + relative path).

        The parser (ingest/parse.py) receives the same `bytes` type regardless
        of whether it came from here or LocalFolderSource.fetch_document().

        Raises azure.core.exceptions.ResourceNotFoundError if the blob is
        gone — the pipeline can treat this as a delete and remove the doc
        from the Pinecone index.
        """
        blob_client = self.client.get_blob_client(doc_id)
        return blob_client.download_blob().readall()

    def watch_for_changes(self) -> Iterator[DocumentMetadata]:
        """
        Yield DocumentMetadata for blobs that are new or modified since the
        last scan, polling on `poll_interval_seconds`.

        # TODO: replace with Azure Event Grid push subscription
        #
        # The production implementation subscribes to BlobCreated and
        # BlobModified events via an Azure Event Grid system topic pointing at
        # this container. Each event payload contains the blob name and can be
        # used to construct a DocumentMetadata and yield it immediately —
        # sub-second latency, zero wasted list_blobs() calls, no sleep loop.
        #
        # Setup:
        #   1. Create an Event Grid system topic for the storage account.
        #   2. Add a subscription filtering for
        #      Microsoft.Storage.BlobCreated / BlobModified events.
        #   3. Point the subscription at a FastAPI webhook endpoint
        #      (api/ingest.py will expose POST /events/blob).
        #   4. Replace this polling loop with the event-driven yield inside
        #      that endpoint handler.
        #
        # For now, poll-and-diff gives correct behaviour with higher latency.

        Local equivalent: identical poll-and-diff logic using stat() mtimes.
        Azure difference: last_modified comes from BlobProperties, not stat().
        """
        # snapshot: doc_id (blob name) → last_modified as UTC timestamp float
        seen: dict[str, float] = {}

        # Seed snapshot so we don't re-ingest blobs that already existed
        # when the watcher starts.
        for meta in self.list_documents():
            seen[meta.doc_id] = meta.last_modified.timestamp()

        while True:
            time.sleep(self.poll_interval)
            for props in self.client.list_blobs(name_starts_with=self.blob_prefix):
                name: str = props["name"]
                if not is_supported(name):
                    continue

                last_modified: datetime = props["last_modified"]
                mtime = last_modified.timestamp()

                if seen.get(name) != mtime:
                    seen[name] = mtime
                    yield self._metadata_from_props(props)
