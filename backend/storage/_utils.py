"""
backend/storage/_utils.py

Shared helpers for all DocumentSource implementations.

Kept here so LocalFolderSource and AzureBlobSource (and future connectors)
use identical filtering and file_type inference — a single change here
propagates to every connector.
"""

from pathlib import PurePosixPath


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".xlsx", ".pptx"
})


def file_type_from_name(name: str) -> str:
    """
    Return the normalised file type string from a filename or blob name.

    Examples: 'report.PDF' → 'pdf', 'notes.md' → 'md', 'data.xlsx' → 'xlsx'
    Uses PurePosixPath so it works on both local filenames and Azure blob
    names (which use '/' as a path separator regardless of OS).
    """
    return PurePosixPath(name).suffix.lower().lstrip(".")


def is_supported(name: str) -> bool:
    """
    True if this filename / blob name should be ingested.

    Skips:
      - Hidden files: names (or final path components) starting with '.'
      - Unsupported extensions: anything not in SUPPORTED_EXTENSIONS
    """
    basename = PurePosixPath(name).name
    return (
        not basename.startswith(".")
        and PurePosixPath(name).suffix.lower() in SUPPORTED_EXTENSIONS
    )
