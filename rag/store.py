"""ChromaDB vector store with pluggable local embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import requests

import config
from rag.ingest import Chunk


# ---------------------------------------------------------------- embeddings
class Embedder:
    """Turns text into vectors. Backends: 'default' (Chroma MiniLM ONNX) or 'ollama'."""

    def __init__(self, backend: str = config.EMBED_BACKEND, model: str = config.EMBED_MODEL):
        self.backend = backend
        self.model = model
        self._fn: Callable[[Sequence[str]], list] | None = None

    @property
    def name(self) -> str:
        return "minilm" if self.backend == "default" else f"ollama-{self.model}".replace(":", "-")

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self.backend == "ollama":
            r = requests.post(f"{config.OLLAMA_URL}/api/embed",
                              json={"model": self.model, "input": list(texts)}, timeout=300)
            r.raise_for_status()
            return r.json()["embeddings"]
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._fn = DefaultEmbeddingFunction()
        return [list(map(float, v)) for v in self._fn(list(texts))]


# ---------------------------------------------------------------- store
@dataclass
class Retrieved:
    chunk: Chunk
    similarity: float  # cosine similarity, higher = closer


class VectorStore:
    def __init__(self, path: Path = config.CHROMA_DIR, embedder: Embedder | None = None):
        import chromadb

        self.embedder = embedder or Embedder()
        self.client = chromadb.PersistentClient(path=str(path))
        # one collection per embedding model, so vectors never get mixed
        self.col = self.client.get_or_create_collection(
            name=f"docs_{self.embedder.name}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # -- write
    def add(self, chunks: list[Chunk], batch: int = 64) -> int:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
                embeddings=self.embedder([c.text for c in part]),
            )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})

    def reset(self) -> None:
        name, meta = self.col.name, {"hnsw:space": "cosine"}
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name=name, metadata=meta, embedding_function=None)

    # -- read
    def count(self) -> int:
        return self.col.count()

    def sources(self) -> dict[str, int]:
        got = self.col.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for m in got["metadatas"] or []:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return dict(sorted(counts.items()))

    def query(self, text: str, k: int = config.TOP_K) -> list[Retrieved]:
        if self.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder([text]),
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=cid, source=meta["source"], page=int(meta["page"]),
                          line_start=int(meta["line_start"]), line_end=int(meta["line_end"]),
                          lines=doc.split("\n"))
            out.append(Retrieved(chunk=chunk, similarity=round(1.0 - float(dist), 4)))
        return out
