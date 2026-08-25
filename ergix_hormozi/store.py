from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

Embedder = Callable[[Iterable[str]], list[list[float]]]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float


def chunk_text(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[i : i + size] for i in range(0, len(paragraph), max(1, size - overlap))]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > size:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{piece}".strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _vector_blob(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def _blob_vector(blob: Optional[bytes]) -> list[float]:
    if not blob:
        return []
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


class KnowledgeStore:
    def __init__(self, path: Path, max_vector_scan: int = 25000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_vector_scan = max_vector_scan
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              title TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              metadata_json TEXT NOT NULL DEFAULT '{}',
              indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              position INTEGER NOT NULL,
              text TEXT NOT NULL,
              embedding BLOB,
              embedding_model TEXT,
              UNIQUE(document_id, position)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='porter unicode61');
            """
        )

    def close(self) -> None:
        self.db.close()

    def stats(self) -> dict[str, int]:
        return {
            "documents": int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "chunks": int(self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        }

    def ingest(
        self,
        source: str,
        title: str,
        text: str,
        embedder: Optional[Embedder] = None,
        embedding_model: str = "",
        metadata: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
        existing = self.db.execute("SELECT content_hash FROM documents WHERE id = ?", (document_id,)).fetchone()
        model_is_current = True
        if existing and embedder:
            stale_vectors = self.db.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ? AND COALESCE(embedding_model, '') != ?",
                (document_id, embedding_model),
            ).fetchone()[0]
            model_is_current = stale_vectors == 0
        if existing and existing["content_hash"] == content_hash and model_is_current:
            return {"source": source, "status": "unchanged", "chunks": 0}

        chunks = chunk_text(text)
        embeddings: list[list[float]] = [[] for _ in chunks]
        if embedder and chunks:
            batch_size = 24
            embeddings = []
            for start in range(0, len(chunks), batch_size):
                embeddings.extend(embedder(chunks[start : start + batch_size]))
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding provider returned the wrong number of vectors")

        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            old_ids = [row[0] for row in self.db.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
            if old_ids:
                self.db.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(item,) for item in old_ids])
            self.db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.db.execute(
                """INSERT INTO documents(id, source, title, content_hash, metadata_json, indexed_at)
                   VALUES(?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, content_hash=excluded.content_hash,
                     metadata_json=excluded.metadata_json, indexed_at=excluded.indexed_at""",
                (document_id, source, title, content_hash, json.dumps(metadata or {}, sort_keys=True), now),
            )
            for position, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                chunk_id = hashlib.sha256(f"{document_id}:{position}:{chunk}".encode("utf-8")).hexdigest()
                self.db.execute(
                    "INSERT INTO chunks(id, document_id, position, text, embedding, embedding_model) VALUES(?, ?, ?, ?, ?, ?)",
                    (chunk_id, document_id, position, chunk, _vector_blob(vector) if vector else None, embedding_model or None),
                )
                self.db.execute("INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)", (chunk_id, chunk))
        return {"source": source, "status": "indexed", "chunks": len(chunks)}

    def search(self, query: str, embedder: Optional[Embedder] = None, top_k: int = 6) -> list[SearchHit]:
        tokens = re.findall(r"[\w'-]{2,}", query.lower())
        fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:16])
        lexical: dict[str, float] = {}
        if fts_query:
            rows = self.db.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fts_query, max(top_k * 8, 30)),
            ).fetchall()
            lexical = {row["chunk_id"]: 1.0 / (index + 1) for index, row in enumerate(rows)}

        semantic: dict[str, float] = {}
        if embedder:
            query_vector = embedder([query])[0]
            for row in self.db.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY rowid DESC LIMIT ?",
                (self.max_vector_scan,),
            ):
                semantic[row["id"]] = _cosine(query_vector, _blob_vector(row["embedding"]))

        candidate_ids = set(lexical)
        candidate_ids.update(key for key, _ in sorted(semantic.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 8, 30)])
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = self.db.execute(
            f"""SELECT c.id, c.text, d.source, d.title FROM chunks c
                JOIN documents d ON d.id = c.document_id WHERE c.id IN ({placeholders})""",
            tuple(candidate_ids),
        ).fetchall()
        hits = [
            SearchHit(
                chunk_id=row["id"],
                source=row["source"],
                title=row["title"],
                text=row["text"],
                score=(0.65 * max(0.0, semantic.get(row["id"], 0.0))) + (0.35 * lexical.get(row["id"], 0.0)),
            )
            for row in rows
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
