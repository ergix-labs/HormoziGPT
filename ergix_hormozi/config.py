from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    database_path: Path
    business_profile_path: Path
    motions_dir: Path
    max_vector_scan: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = Path(os.getenv("HORMOZI_DATA_DIR", repo_root / "data")).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        motions_dir = data_dir / "motions"
        motions_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            chat_model=os.getenv("HORMOZI_CHAT_MODEL", "qwen3.5:9b"),
            embedding_model=os.getenv("HORMOZI_EMBEDDING_MODEL", "nomic-embed-text"),
            database_path=Path(os.getenv("HORMOZI_DATABASE", data_dir / "knowledge.db")).expanduser().resolve(),
            business_profile_path=Path(os.getenv("HORMOZI_BUSINESS_PROFILE", data_dir / "business.md")).expanduser().resolve(),
            motions_dir=motions_dir,
            max_vector_scan=max(100, int(os.getenv("HORMOZI_MAX_VECTOR_SCAN", "25000"))),
        )
