"""
agentx_memory.py - Custom Vector Memory & Semantic RAG Engine
=============================================================
Zero-framework in-memory Vector Database using NumPy for vectorized Cosine Similarity:
  similarity = (A · B) / (||A|| * ||B||)
Includes:
  1. SlidingWindowChunker: Overlapping text slicing with boundary snapping.
  2. SubwordHashingVectorizer: Zero-dependency deterministic unit-norm vectorizer.
  3. REST Embedding Client: Direct HTTP /v1/embeddings caller.
  4. VectorStore: In-memory vector database with inverted metadata filtering.
  5. Native tool integration: memory_search tool registered with agentx_tools.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from agentx_tools import agentx_tool, DEFAULT_TOOL_REGISTRY


# ============================================================================
# 1. SLIDING WINDOW TEXT CHUNKER
# ============================================================================

@dataclass
class TextChunk:
    """Discrete text chunk with positional and provenance metadata."""
    chunk_id: str
    doc_id: str
    text: str
    index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class SlidingWindowChunker:
    """Slices documents into overlapping chunks respecting whitespace and word boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str = "doc_0", base_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        clean = text.strip()
        if not clean:
            return []

        metadata = dict(base_metadata or {})
        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        total_len = len(clean)
        start = 0
        idx = 0

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            # Snap to word boundary if not at text end
            if end < total_len:
                boundary = clean.rfind(" ", start + (self.chunk_size // 2), end)
                if boundary != -1:
                    end = boundary

            chunk_str = clean[start:end].strip()
            if chunk_str:
                cid = f"{doc_id}_chunk_{idx:04d}"
                meta = dict(metadata)
                meta.update({
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "char_start": start,
                    "char_end": end,
                    "char_length": len(chunk_str)
                })

                chunks.append(TextChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    text=chunk_str,
                    index=idx,
                    char_start=start,
                    char_end=end,
                    metadata=meta
                ))
                idx += 1

            if end >= total_len:
                break
            start += step

        return chunks


# ============================================================================
# 2. DUAL-MODE EMBEDDING ENGINE
# ============================================================================

class SubwordHashingVectorizer:
    """
    Zero-dependency deterministic dense subword vectorizer using MD5 hashing trick.
    Guarantees reproducible, normalized float32 embeddings for offline semantic similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        tokens = list(words)
        for w in words:
            if len(w) > 3:
                for i in range(len(w) - 2):
                    tokens.append(f"#{w[i:i+3]}")
        return tokens

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for t in tokens:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
            vec[idx] += sign

        # Unit-norm normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class EmbeddingClient:
    """Manages embedding generation via OpenAI-compatible REST API with local fallback."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AGENTX_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.local_vectorizer = SubwordHashingVectorizer()

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        if not self.api_key:
            return self.local_vectorizer.embed_batch(texts)

        endpoint = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {"input": texts, "model": self.model}

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                results = []
                for item in data:
                    arr = np.array(item["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    results.append(arr / norm if norm > 0 else arr)
                return results
            return self.local_vectorizer.embed_batch(texts)
        except Exception:
            return self.local_vectorizer.embed_batch(texts)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


# ============================================================================
# 3. NUMPY VECTOR STORE WITH COSINE SIMILARITY MATH
# ============================================================================

@dataclass
class SearchResult:
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


class VectorStore:
    """
    In-memory vector database using NumPy for vectorized Cosine Similarity.
    Features inverted metadata indexing for filtered retrieval.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, api_key: Optional[str] = None):
        self.chunker = SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embed_client = EmbeddingClient(api_key=api_key)

        self._chunks: Dict[str, TextChunk] = {}
        self._doc_index: Dict[str, List[str]] = {}
        self._meta_index: Dict[str, Dict[str, set]] = {}

        self._matrix_ids: List[str] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._dirty: bool = True

    def _sync_matrix(self) -> None:
        """Stack vectors into a dense 2D NumPy array for batch matrix multiplication."""
        if not self._dirty:
            return

        ids, vecs = [], []
        for cid, chk in self._chunks.items():
            if chk.embedding is not None:
                ids.append(cid)
                vecs.append(chk.embedding)

        if vecs:
            self._embedding_matrix = np.vstack(vecs).astype(np.float32)
            self._matrix_ids = ids
        else:
            self._embedding_matrix = None
            self._matrix_ids = []

        self._dirty = False

    def add_document(self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text into chunks, compute embeddings, and index into memory."""
        clean = text.strip()
        if not clean:
            return []

        document_id = doc_id or f"doc_{int(time.time() * 1000)}"
        meta = dict(metadata or {})
        meta["doc_id"] = document_id

        chunks = self.chunker.chunk_text(clean, doc_id=document_id, base_metadata=meta)
        if not chunks:
            return []

        embeddings = self.embed_client.embed_texts([c.text for c in chunks])
        created_ids = []

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self._chunks[chunk.chunk_id] = chunk
            created_ids.append(chunk.chunk_id)

            if document_id not in self._doc_index:
                self._doc_index[document_id] = []
            self._doc_index[document_id].append(chunk.chunk_id)

            # Update inverted index for fast metadata lookups
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    val_s = str(v)
                    if k not in self._meta_index:
                        self._meta_index[k] = {}
                    if val_s not in self._meta_index[k]:
                        self._meta_index[k][val_s] = set()
                    self._meta_index[k][val_s].add(chunk.chunk_id)

        self._dirty = True
        return created_ids

    def search(
        self,
        query: str,
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform vectorized Cosine Similarity search:
          similarity = (A · B) / (||A|| * ||B||)
        With normalized vectors, this simplifies to the dot product: S = Matrix @ query_vector
        """
        clean_q = query.strip()
        if not clean_q or not self._chunks:
            return []

        self._sync_matrix()
        if self._embedding_matrix is None or not self._matrix_ids:
            return []

        query_vec = self.embed_client.embed_single(clean_q).astype(np.float32)

        # Apply metadata filters via inverted index
        candidate_indices: Optional[List[int]] = None
        if filter_metadata:
            candidate_ids: Optional[set] = None
            for fk, fv in filter_metadata.items():
                matched = self._meta_index.get(fk, {}).get(str(fv), set())
                candidate_ids = set(matched) if candidate_ids is None else candidate_ids & matched

            if not candidate_ids:
                return []

            candidate_indices = [i for i, cid in enumerate(self._matrix_ids) if cid in candidate_ids]
            if not candidate_indices:
                return []

        # NumPy Dot Product Cosine Similarity
        if candidate_indices is not None:
            sub_mat = self._embedding_matrix[candidate_indices]
            sims = np.dot(sub_mat, query_vec)
            target_ids = [self._matrix_ids[i] for i in candidate_indices]
        else:
            sims = np.dot(self._embedding_matrix, query_vec)
            target_ids = self._matrix_ids

        k = min(top_k, len(target_ids))
        if k == len(target_ids):
            sorted_idx = np.argsort(-sims)
        else:
            top_part = np.argpartition(-sims, k - 1)[:k]
            sorted_idx = top_part[np.argsort(-sims[top_part])]

        results: List[SearchResult] = []
        for idx in sorted_idx:
            score = float(sims[idx])
            if score < min_similarity:
                continue
            cid = target_ids[idx]
            chk = self._chunks[cid]
            results.append(SearchResult(
                chunk_id=chk.chunk_id,
                doc_id=chk.doc_id,
                text=chk.text,
                similarity=score,
                metadata=chk.metadata
            ))

        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._doc_index.clear()
        self._meta_index.clear()
        self._matrix_ids.clear()
        self._embedding_matrix = None
        self._dirty = True

    def count(self) -> int:
        return len(self._chunks)


# Global default VectorStore instance
DEFAULT_VECTOR_STORE = VectorStore()


# ============================================================================
# 4. MEMORY TOOL REGISTRATION
# ============================================================================

@agentx_tool(name="memory_search", description="Search AgentX's internal in-memory vector store for relevant documentation and past facts using semantic similarity.")
def memory_search(query: str, top_k: int = 3) -> str:
    """
    Query internal vector memory for relevant context passages.
    :param query: Semantic search query.
    :param top_k: Max chunks to return (default: 3).
    """
    hits = DEFAULT_VECTOR_STORE.search(query=query, top_k=top_k)
    if not hits:
        return f"No matching context found in vector memory for query: '{query}'."

    out = [f"=== Retrieved {len(hits)} Context Passages from Vector Memory ==="]
    for i, h in enumerate(hits, start=1):
        source = h.metadata.get("source", h.doc_id)
        out.append(f"[{i}] (Similarity: {h.similarity:.3f} | Source: {source})\n{h.text}")
    return "\n\n".join(out)
