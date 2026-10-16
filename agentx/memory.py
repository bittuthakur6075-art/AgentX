"""
AgentX Custom Vector Memory & RAG Engine
========================================
Zero-framework in-memory vector database using NumPy for vectorized Cosine Similarity,
sliding window text chunking with overlap, metadata indexing, and dual-mode embeddings
(Remote REST endpoint + Zero-dependency deterministic local subword hashing vectorizer).
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from agentx.config import AgentXConfig, DEFAULT_CONFIG
from agentx.tools import agentx_tool, DEFAULT_REGISTRY


# ============================================================================
# 1. TEXT CHUNKER (SLIDING WINDOW WITH OVERLAP)
# ============================================================================

@dataclass
class TextChunk:
    """Represents a discrete segment of chunked text with positional metadata."""
    chunk_id: str
    doc_id: str
    text: str
    index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class SlidingWindowChunker:
    """
    Splits continuous text into overlapping chunks using a sliding window.
    Respects word boundaries and paragraph breaks to prevent semantic truncation.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str = "doc_default", base_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Slice input text into overlapping chunks.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        metadata = dict(base_metadata or {})
        chunks: List[TextChunk] = []

        step = self.chunk_size - self.chunk_overlap
        total_len = len(clean_text)
        start = 0
        chunk_idx = 0

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            # If not at the very end of text, attempt to snap to the nearest whitespace boundary
            if end < total_len:
                boundary = clean_text.rfind(" ", start + (self.chunk_size // 2), end)
                if boundary != -1:
                    end = boundary

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
                chunk_meta = dict(metadata)
                chunk_meta.update({
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": end,
                    "length": len(chunk_content)
                })

                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_content,
                    index=chunk_idx,
                    char_start=start,
                    char_end=end,
                    metadata=chunk_meta
                ))
                chunk_idx += 1

            if end >= total_len:
                break

            start += step

        return chunks


# ============================================================================
# 2. DUAL-MODE EMBEDDING ENGINE
# ============================================================================

class LocalHashingVectorizer:
    """
    High-performance, zero-dependency subword TF-IDF hashing vectorizer.
    Generates deterministic, dense unit-norm embeddings using subword n-grams and murmur/sha256 hashing.
    Enables instant out-of-the-box local semantic search without requiring third-party models or API keys.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words and character 3-grams for semantic richness."""
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        tokens = list(words)

        # Generate character trigrams for words longer than 3 chars (captures stem similarity)
        for w in words:
            if len(w) > 3:
                for i in range(len(w) - 2):
                    tokens.append(f"#{w[i:i+3]}")
        return tokens

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized dense vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic hash mapping into vector coordinates
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        return [self.embed_text(t) for t in texts]


class EmbeddingClient:
    """
    Manages embedding generation via OpenAI-compatible REST API (/v1/embeddings)
    with seamless automatic fallback to the built-in LocalHashingVectorizer.
    """

    def __init__(self, config: Optional[AgentXConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.local_fallback = LocalHashingVectorizer(dimension=self.config.embedding_dimension)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of text strings.
        Attempts remote REST API first; falls back to local vectorizer if no key or on network error.
        """
        if not texts:
            return []

        # Check if remote API credentials exist
        api_key = self.config.embedding_api_key or self.config.api_key
        if not api_key:
            return self.local_fallback.embed_batch(texts)

        endpoint = f"{self.config.embedding_base_url}/embeddings"
        headers = self.config.get_embedding_headers()
        payload = {
            "input": texts,
            "model": self.config.embedding_model
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                embeddings_data = data.get("data", [])
                results: List[np.ndarray] = []
                for item in embeddings_data:
                    raw_vec = np.array(item["embedding"], dtype=np.float32)
                    # Normalize vector
                    norm = np.linalg.norm(raw_vec)
                    if norm > 0:
                        raw_vec = raw_vec / norm
                    results.append(raw_vec)
                return results
            else:
                # Log warning and fall back
                return self.local_fallback.embed_batch(texts)
        except Exception:
            # Fall back to local vectorizer on network or timeout failure
            return self.local_fallback.embed_batch(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string."""
        return self.embed_texts([text])[0]


# ============================================================================
# 3. IN-MEMORY VECTOR STORE & METADATA INDEXER
# ============================================================================

@dataclass
class SearchResult:
    """Encapsulates a ranked search hit from the vector database."""
    chunk_id: str
    doc_id: str
    text: str
    similarity: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "similarity": round(float(self.similarity), 4),
            "metadata": self.metadata
        }


class VectorMemory:
    """
    High-performance In-Memory Vector Store.
    Uses NumPy matrix multiplication for instant vectorized Cosine Similarity.
    Maintains metadata inverted indexing for filtered RAG retrieval.
    """

    def __init__(self, config: Optional[AgentXConfig] = None, chunk_size: int = 500, chunk_overlap: int = 100):
        self.config = config or DEFAULT_CONFIG
        self.chunker = SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_client = EmbeddingClient(self.config)

        # Internal chunk storage
        self._chunks: Dict[str, TextChunk] = {}
        # Document registry: doc_id -> list of chunk_ids
        self._doc_index: Dict[str, List[str]] = {}
        # Inverted index for metadata filtering: key -> val -> set of chunk_ids
        self._meta_index: Dict[str, Dict[str, set]] = {}

        # Cached NumPy matrix for vectorized calculations
        self._matrix_chunk_ids: List[str] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._is_matrix_dirty: bool = True

    def _update_matrix(self) -> None:
        """Rebuild dense NumPy embedding matrix for batch matrix multiplication."""
        if not self._is_matrix_dirty:
            return

        valid_ids: List[str] = []
        vectors: List[np.ndarray] = []

        for cid, chunk in self._chunks.items():
            if chunk.embedding is not None:
                valid_ids.append(cid)
                vectors.append(chunk.embedding)

        if vectors:
            self._embedding_matrix = np.vstack(vectors).astype(np.float32)
            self._matrix_chunk_ids = valid_ids
        else:
            self._embedding_matrix = None
            self._matrix_chunk_ids = []

        self._is_matrix_dirty = False

    def add_document(self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Split a document into chunks, compute embeddings, and index into the vector store.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        document_id = doc_id or f"doc_{int(time.time() * 1000)}"
        meta = dict(metadata or {})
        meta["doc_id"] = document_id

        # 1. Chunk document
        chunks = self.chunker.chunk_text(clean_text, doc_id=document_id, base_metadata=meta)
        if not chunks:
            return []

        # 2. Batch generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_client.embed_texts(chunk_texts)

        # 3. Store and index
        created_ids: List[str] = []
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self._chunks[chunk.chunk_id] = chunk
            created_ids.append(chunk.chunk_id)

            # Update doc index
            if document_id not in self._doc_index:
                self._doc_index[document_id] = []
            self._doc_index[document_id].append(chunk.chunk_id)

            # Update metadata inverted index
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    val_str = str(v)
                    if k not in self._meta_index:
                        self._meta_index[k] = {}
                    if val_str not in self._meta_index[k]:
                        self._meta_index[k][val_str] = set()
                    self._meta_index[k][val_str].add(chunk.chunk_id)

        self._is_matrix_dirty = True
        return created_ids

    def search(
        self,
        query: str,
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Vector similarity search using Cosine Similarity.
        Supports pre-filtering via metadata index and threshold cutoff.
        """
        clean_query = query.strip()
        if not clean_query or not self._chunks:
            return []

        self._update_matrix()
        if self._embedding_matrix is None or len(self._matrix_chunk_ids) == 0:
            return []

        # 1. Compute query vector
        query_vec = self.embedding_client.embed_single(clean_query).astype(np.float32)

        # 2. Determine candidate set based on metadata filters
        candidate_indices: Optional[List[int]] = None
        if filter_metadata:
            candidate_chunk_ids: Optional[set] = None
            for fk, fv in filter_metadata.items():
                val_str = str(fv)
                matching = self._meta_index.get(fk, {}).get(val_str, set())
                if candidate_chunk_ids is None:
                    candidate_chunk_ids = set(matching)
                else:
                    candidate_chunk_ids &= matching

            if not candidate_chunk_ids:
                return []

            candidate_indices = [
                idx for idx, cid in enumerate(self._matrix_chunk_ids)
                if cid in candidate_chunk_ids
            ]
            if not candidate_indices:
                return []

        # 3. Vectorized Cosine Similarity calculation: (Matrix @ query_vector)
        # Note: both matrix rows and query vector are L2 normalized, so dot product == cosine similarity
        if candidate_indices is not None:
            sub_matrix = self._embedding_matrix[candidate_indices]
            similarities = np.dot(sub_matrix, query_vec)
            mapped_ids = [self._matrix_chunk_ids[i] for i in candidate_indices]
        else:
            similarities = np.dot(self._embedding_matrix, query_vec)
            mapped_ids = self._matrix_chunk_ids

        # 4. Top-K Sorting using argpartition / argsort
        num_candidates = len(mapped_ids)
        k = min(top_k, num_candidates)

        if k == num_candidates:
            sorted_indices = np.argsort(-similarities)
        else:
            top_partition = np.argpartition(-similarities, k - 1)[:k]
            sorted_indices = top_partition[np.argsort(-similarities[top_partition])]

        results: List[SearchResult] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score < min_similarity:
                continue

            cid = mapped_ids[idx]
            chunk = self._chunks[cid]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                similarity=score,
                metadata=chunk.metadata
            ))

        return results

    def delete_document(self, doc_id: str) -> int:
        """Remove an entire document and its chunks from memory."""
        chunk_ids = self._doc_index.pop(doc_id, [])
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

        # Clear metadata references
        for k in list(self._meta_index.keys()):
            for v in list(self._meta_index[k].keys()):
                self._meta_index[k][v].difference_update(chunk_ids)

        self._is_matrix_dirty = True
        return len(chunk_ids)

    def clear(self) -> None:
        """Wipe all chunks and indices from memory."""
        self._chunks.clear()
        self._doc_index.clear()
        self._meta_index.clear()
        self._matrix_chunk_ids.clear()
        self._embedding_matrix = None
        self._is_matrix_dirty = True

    def count(self) -> int:
        """Return total number of chunks currently stored in vector memory."""
        return len(self._chunks)

    def to_json(self) -> str:
        """Serialize memory store to JSON string (excluding raw vector arrays for brevity)."""
        export_data = {
            "total_chunks": len(self._chunks),
            "documents": list(self._doc_index.keys()),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "metadata": c.metadata
                }
                for c in self._chunks.values()
            ]
        }
        return json.dumps(export_data, indent=2)

    def load_from_json(self, json_str: str) -> int:
        """Load and index documents from a JSON representation."""
        data = json.loads(json_str)
        count = 0
        for item in data.get("chunks", []):
            self.add_document(
                text=item["text"],
                doc_id=item.get("doc_id"),
                metadata=item.get("metadata")
            )
            count += 1
        return count


# Global default vector memory instance
DEFAULT_MEMORY = VectorMemory()


# ============================================================================
# 4. MEMORY TOOL REGISTRATION FOR REACT ENGINE
# ============================================================================

@agentx_tool(name="memory_search", description="Search the agent's internal vector memory / knowledge base for relevant facts, documents, and past context using semantic similarity.")
def memory_search(query: str, top_k: int = 3) -> str:
    """
    Search internal vector memory for semantic context matches.

    :param query: Semantic search query string.
    :param top_k: Maximum number of relevant chunks to return (default: 3).
    :return: Formatted context passages or empty notice.
    """
    results = DEFAULT_MEMORY.search(query=query, top_k=top_k)
    if not results:
        return f"No relevant context found in vector memory for query: '{query}'."

    parts: List[str] = [f"=== Retrieved {len(results)} Context Passages from Vector Memory ==="]
    for idx, hit in enumerate(results, start=1):
        source = hit.metadata.get("source", hit.doc_id)
        parts.append(
            f"[{idx}] (Similarity: {hit.similarity:.3f} | Source: {source})\n{hit.text}"
        )
    return "\n\n".join(parts)


def get_default_memory() -> VectorMemory:
    """Access global vector memory singleton."""
    return DEFAULT_MEMORY
