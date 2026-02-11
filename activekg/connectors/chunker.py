"""Smart text chunking with overlap for long documents."""

import re
import uuid
from typing import Any

from activekg.graph.models import Edge, Node


def _deterministic_uuid(external_id: str) -> str:
    """Generate a deterministic UUID from an external ID string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, external_id))

# Default chunking parameters
DEFAULT_MAX_CHUNK_CHARS = 8000
DEFAULT_OVERLAP_CHARS = 500


def chunk_text(
    text: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks for better retrieval.

    Args:
        text: Text to chunk
        max_chunk_chars: Maximum characters per chunk
        overlap_chars: Characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_chars:
        return [text]  # No chunking needed

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chunk_chars, len(text))

        # If not at the end, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings in last 500 chars of chunk
            chunk_end_zone = text[max(end - 500, start) : end]
            sentence_breaks = [m.end() for m in re.finditer(r"[.!?]\s+", chunk_end_zone)]

            if sentence_breaks:
                # Break at last sentence in chunk
                last_break = sentence_breaks[-1]
                end = max(end - 500, start) + last_break

        chunks.append(text[start:end].strip())

        # Move start forward, but overlap
        start = end - overlap_chars if end < len(text) else len(text)

    return chunks


def create_chunk_nodes(
    parent_node_id: str,
    parent_title: str,
    parent_classes: list[str],
    text: str,
    parent_metadata: dict[str, Any],
    repo,
    tenant_id: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Create parent + chunk nodes with DERIVED_FROM edges.

    Args:
        parent_node_id: External ID of parent document
        parent_title: Title of parent document
        parent_classes: Classes for parent (e.g., ["Document", "Job"])
        text: Full text content
        parent_metadata: Metadata for parent node (etag, modified_at, etc.)
        repo: GraphRepository instance
        tenant_id: Tenant ID
        max_chunk_chars: Max chars per chunk
        overlap_chars: Overlap between chunks

    Returns:
        List of created chunk node IDs
    """
    # Create parent node (no embedding - lightweight)
    parent_uuid = _deterministic_uuid(parent_node_id)
    repo.create_node(Node(
        id=parent_uuid,
        classes=parent_classes,
        props={
            "title": parent_title,
            "external_id": parent_node_id,
            **parent_metadata,
            "is_parent": True,
            "has_chunks": True,
        },
        embedding=None,
        tenant_id=tenant_id,
    ))

    # Chunk the text
    chunks = chunk_text(text, max_chunk_chars, overlap_chars)
    chunk_ids = []

    # Create chunk nodes
    for i, chunk_content in enumerate(chunks):
        chunk_external_id = f"{parent_node_id}#chunk{i}"
        chunk_uuid = _deterministic_uuid(chunk_external_id)

        # Inherit parent classes + add Chunk
        chunk_classes = ["Chunk"] + [c for c in parent_classes if c != "Document"]

        # Create chunk node (embedding enqueued by IngestionProcessor)
        repo.create_node(Node(
            id=chunk_uuid,
            classes=chunk_classes,
            props={
                "text": chunk_content,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parent_id": parent_node_id,
                "parent_title": parent_title,
                "external_id": chunk_external_id,
                # Inherit entity hints for typed retrieval
                "entity_type": parent_metadata.get("entity_type"),
                "role": parent_metadata.get("role"),  # For job chunks
                "skills": parent_metadata.get("skills"),  # For job/resume chunks
            },
            tenant_id=tenant_id,
        ))

        # Create DERIVED_FROM edge with chunk position metadata
        repo.create_edge(Edge(
            src=chunk_uuid,
            dst=parent_uuid,
            rel="DERIVED_FROM",
            props={
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_start": i * (max_chunk_chars - overlap_chars),  # Approximate
                "char_end": (i + 1) * max_chunk_chars,
            },
            tenant_id=tenant_id,
        ))

        chunk_ids.append(chunk_uuid)

    return chunk_ids
