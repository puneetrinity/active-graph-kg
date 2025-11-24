"""Incremental index update system for real-time data changes."""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Optional


class ChangeType(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class DocumentChange:
    """Represents a change to a document."""

    doc_id: str
    change_type: ChangeType
    document: dict[str, Any] | None = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class IncrementalIndexManager:
    """Manages incremental updates to search indexes."""

    def __init__(self, search_engine, batch_size: int = 50, max_queue_size: int = 1000):
        self.search_engine = search_engine
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # Change tracking
        self.change_queue: Deque[DocumentChange] = deque(maxlen=max_queue_size)
        self.pending_changes: dict[str, DocumentChange] = {}
        self.processing_lock = threading.RLock()

        # Background processing
        self.is_processing: bool = False
        self.processing_task: Optional[asyncio.Task[Any]] = None
        self.last_batch_process: float = time.time()

        # Statistics
        self.stats: dict[str, Any] = {
            "total_changes_processed": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "last_update_time": None,
            "queue_size": 0,
        }

        # Enable incremental updates
        self.enabled = True

    async def start_background_processing(self):
        """Start background processing of incremental updates."""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._background_processor())

    async def stop_background_processing(self):
        """Stop background processing."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

    def add_document_change(
        self, doc_id: str, change_type: ChangeType, document: dict[str, Any] | None = None
    ):
        """Add a document change to the queue."""
        if not self.enabled:
            return

        change = DocumentChange(doc_id, change_type, document)

        with self.processing_lock:
            # If there's already a pending change for this document, update it
            if doc_id in self.pending_changes:
                existing_change = self.pending_changes[doc_id]

                # Handle change consolidation logic
                if (
                    existing_change.change_type == ChangeType.ADD
                    and change_type == ChangeType.DELETE
                ):
                    # Add then delete = no operation needed
                    del self.pending_changes[doc_id]
                    return
                elif existing_change.change_type == ChangeType.DELETE:
                    # Any change after delete becomes an add
                    if change_type != ChangeType.DELETE:
                        change.change_type = ChangeType.ADD

                self.pending_changes[doc_id] = change
            else:
                self.pending_changes[doc_id] = change
                self.change_queue.append(change)

            self.stats["queue_size"] = len(self.pending_changes)

    async def force_process_pending_changes(self) -> dict[str, Any]:
        """Force processing of all pending changes immediately."""
        if not self.pending_changes:
            return {"processed": 0, "errors": 0}

        return await self._process_change_batch(list(self.pending_changes.values()))

    async def _background_processor(self):
        """Background task to process incremental updates."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Process batch if we have enough changes or enough time has passed
                should_process = (
                    len(self.pending_changes) >= self.batch_size
                    or (
                        len(self.pending_changes) > 0 and time.time() - self.last_batch_process > 30
                    )  # 30 second timeout
                )

                if should_process and not self.is_processing:
                    await self._process_pending_changes()

            except asyncio.CancelledError:
                # Process any remaining changes before shutting down
                if self.pending_changes:
                    await self._process_pending_changes()
                break
            except Exception as e:
                # Log error but continue processing
                if hasattr(self.search_engine, "logger"):
                    self.search_engine.logger.error(f"Error in background processor: {str(e)}")
                await asyncio.sleep(10)  # Wait longer after error

    async def _process_pending_changes(self):
        """Process all pending changes in batches."""
        if self.is_processing or not self.pending_changes:
            return

        self.is_processing = True
        try:
            # Get all pending changes
            with self.processing_lock:
                changes_to_process = list(self.pending_changes.values())
                self.pending_changes.clear()
                self.change_queue.clear()

            # Process in batches
            for i in range(0, len(changes_to_process), self.batch_size):
                batch = changes_to_process[i : i + self.batch_size]
                await self._process_change_batch(batch)

            self.last_batch_process = time.time()
            self.stats["queue_size"] = 0

        finally:
            self.is_processing = False

    async def _process_change_batch(self, changes: list[DocumentChange]) -> dict[str, Any]:
        """Process a batch of document changes."""
        if not changes:
            return {"processed": 0, "errors": 0}

        processed = 0
        errors = 0

        # Group changes by type for efficient processing
        adds = [c for c in changes if c.change_type == ChangeType.ADD]
        updates = [c for c in changes if c.change_type == ChangeType.UPDATE]
        deletes = [c for c in changes if c.change_type == ChangeType.DELETE]

        try:
            # Process deletions first
            for change in deletes:
                try:
                    await self._delete_document(change.doc_id)
                    processed += 1
                except Exception as e:
                    errors += 1
                    if hasattr(self.search_engine, "logger"):
                        self.search_engine.logger.error(
                            f"Failed to delete document {change.doc_id}: {str(e)}"
                        )

            # Process additions and updates together (they're handled similarly)
            add_update_docs = []
            for change in adds + updates:
                if change.document:
                    add_update_docs.append(change.document)

            if add_update_docs:
                try:
                    await self._add_update_documents(add_update_docs)
                    processed += len(add_update_docs)
                except Exception as e:
                    errors += len(add_update_docs)
                    if hasattr(self.search_engine, "logger"):
                        self.search_engine.logger.error(f"Failed to add/update documents: {str(e)}")

            # Update statistics
            self.stats["total_changes_processed"] += processed
            self.stats["successful_updates"] += processed
            self.stats["failed_updates"] += errors
            self.stats["last_update_time"] = datetime.now(timezone.utc).isoformat()

            # Save updated indexes if we processed any changes
            if processed > 0:
                self.search_engine.save_indexes()

        except Exception as e:
            errors += len(changes)
            if hasattr(self.search_engine, "logger"):
                self.search_engine.logger.error(f"Batch processing failed: {str(e)}")

        return {"processed": processed, "errors": errors}

    async def _delete_document(self, doc_id: str):
        """Delete a single document from all indexes."""
        # Remove from document storage
        if (
            hasattr(self.search_engine, "document_vectors")
            and doc_id in self.search_engine.document_vectors
        ):
            del self.search_engine.document_vectors[doc_id]

        if (
            hasattr(self.search_engine, "document_metadata")
            and doc_id in self.search_engine.document_metadata
        ):
            del self.search_engine.document_metadata[doc_id]

        if (
            hasattr(self.search_engine, "document_codes")
            and doc_id in self.search_engine.document_codes
        ):
            del self.search_engine.document_codes[doc_id]

        if (
            hasattr(self.search_engine, "document_text_features")
            and doc_id in self.search_engine.document_text_features
        ):
            del self.search_engine.document_text_features[doc_id]

        # Remove from BM25 index
        if hasattr(self.search_engine, "bm25_index") and doc_id in self.search_engine.bm25_index:
            del self.search_engine.bm25_index[doc_id]

        # Remove from LSH index (requires rebuilding signatures)
        if hasattr(self.search_engine, "lsh_index") and hasattr(
            self.search_engine.lsh_index, "signatures"
        ):
            if doc_id in self.search_engine.lsh_index.signatures:
                del self.search_engine.lsh_index.signatures[doc_id]

        # Note: HNSW index doesn't support individual deletions easily
        # We mark it for rebuild if too many deletions accumulate
        if not hasattr(self.search_engine, "_deleted_docs"):
            self.search_engine._deleted_docs = set()
        self.search_engine._deleted_docs.add(doc_id)

        # Trigger rebuild if too many documents deleted
        if len(self.search_engine._deleted_docs) > 100:  # Threshold
            await self._schedule_index_rebuild()

    async def _add_update_documents(self, documents: list[dict[str, Any]]):
        """Add or update multiple documents in the indexes."""
        if not documents:
            return

        # Generate embeddings for new/updated documents
        texts_to_embed = [self.search_engine._get_document_text(doc) for doc in documents]
        vectors = self.search_engine.embedding_model.encode(
            texts_to_embed, show_progress_bar=False, convert_to_numpy=True
        )

        for i, doc in enumerate(documents):
            doc_id = doc["id"]
            vector = vectors[i]

            # Update document storage
            self.search_engine.document_vectors[doc_id] = vector
            self.search_engine.document_metadata[doc_id] = {
                "name": doc.get("name", ""),
                "experience_years": doc.get("experience_years", 0),
                "skills": doc.get("skills", []),
                "seniority_level": doc.get("seniority_level", "unknown"),
            }

            # Update text features
            text_features = self.search_engine._extract_text_features(doc)
            self.search_engine.document_text_features[doc_id] = text_features

            # Update LSH index
            self.search_engine.lsh_index.add_document(doc_id, text_features)

            # Update PQ codes
            if (
                hasattr(self.search_engine, "pq_quantizer")
                and self.search_engine.pq_quantizer.trained
            ):
                self.search_engine.document_codes[doc_id] = self.search_engine.pq_quantizer.encode(
                    vector.reshape(1, -1)
                )[0]

            # Update BM25 index
            text = self.search_engine._get_document_text(doc)
            tokens = text.lower().split()
            tf = {token: tokens.count(token) for token in set(tokens)}

            # Update document frequencies
            if doc_id not in self.search_engine.bm25_index:  # New document
                for token in set(tokens):
                    self.search_engine.doc_frequencies[token] = (
                        self.search_engine.doc_frequencies.get(token, 0) + 1
                    )
                self.search_engine.corpus_size += 1

            self.search_engine.bm25_index[doc_id] = {"tf": tf, "length": len(tokens)}

        # Add vectors to HNSW index (this requires rebuilding for now)
        # In a production system, you'd use a more sophisticated approach
        await self._update_hnsw_index(documents, vectors)

    async def _update_hnsw_index(self, documents: list[dict[str, Any]], vectors):
        """Update HNSW index with new documents."""
        # For now, we add to the existing index
        # In production, you might use a more sophisticated incremental approach
        doc_ids = [doc["id"] for doc in documents]

        try:
            # Add normalized vectors to HNSW index
            import numpy as np

            normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            self.search_engine.hnsw_index.index.add(normalized_vectors)
            self.search_engine.hnsw_index.doc_ids.extend(doc_ids)

        except Exception as e:
            # If adding fails, schedule a rebuild
            if hasattr(self.search_engine, "logger"):
                self.search_engine.logger.warning(
                    f"HNSW incremental update failed, scheduling rebuild: {str(e)}"
                )
            await self._schedule_index_rebuild()

    async def _schedule_index_rebuild(self):
        """Schedule a full index rebuild."""
        # This would trigger a background rebuild of the indexes
        # For now, we just log the need for rebuild
        if hasattr(self.search_engine, "logger"):
            self.search_engine.logger.info(
                "Index rebuild scheduled due to incremental update limitations"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get incremental update statistics."""
        return {
            **self.stats,
            "queue_size": len(self.pending_changes),
            "is_processing": self.is_processing,
            "enabled": self.enabled,
        }

    def enable_incremental_updates(self):
        """Enable incremental updates."""
        self.enabled = True

    def disable_incremental_updates(self):
        """Disable incremental updates."""
        self.enabled = False
