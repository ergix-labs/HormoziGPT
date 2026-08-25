from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .store import Embedder, KnowledgeStore

SUPPORTED = {".md", ".txt", ".pdf", ".csv", ".json", ".jsonl", ".html", ".htm"}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text("utf-8", errors="replace")
    if suffix == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return "\n".join(" | ".join(f"{key}: {value}" for key, value in row.items() if value) for row in csv.DictReader(handle))
    if suffix in {".json", ".jsonl"}:
        raw = path.read_text("utf-8", errors="replace")
        if suffix == ".jsonl":
            return "\n".join(json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True) for line in raw.splitlines() if line.strip())
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2, sort_keys=True)
    if suffix in {".html", ".htm"}:
        return BeautifulSoup(path.read_text("utf-8", errors="replace"), "html.parser").get_text("\n", strip=True)
    raise ValueError(f"Unsupported knowledge file: {path}")


def discover(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED)
        elif path.is_file() and path.suffix.lower() in SUPPORTED:
            files.append(path)
    return sorted(set(files))


def ingest_paths(paths: Iterable[Path], store: KnowledgeStore, embedder: Embedder, embedding_model: str = "nomic-embed-text") -> dict[str, object]:
    results = []
    for path in discover(paths):
        results.append(store.ingest(str(path), path.stem.replace("_", " "), extract_text(path), embedder, embedding_model))
    return {
        "files": len(results),
        "indexed": sum(1 for item in results if item["status"] == "indexed"),
        "unchanged": sum(1 for item in results if item["status"] == "unchanged"),
        "chunks": sum(int(item["chunks"]) for item in results),
        "results": results,
    }
