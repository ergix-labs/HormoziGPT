from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _read_harness_secret(name: str) -> str:
    """Read an Infisical-synced secret from the local Ergix harness store."""
    configured = os.getenv("HORMOZI_HARNESS_DB", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / "Library/Application Support/ergix-dev-env/state-shared/harness.db",
        Path.home() / ".codex-dashboard/harness.db",
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT value FROM secrets WHERE name = ? LIMIT 1", (name,)
                ).fetchone()
            finally:
                connection.close()
            if row and row[0]:
                return str(row[0]).strip()
        except sqlite3.Error:
            continue
    return ""


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_provider: str
    chat_base_url: str
    chat_api_key: str
    chat_model: str
    reasoning_effort: str
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
            chat_provider=os.getenv("HORMOZI_CHAT_PROVIDER", "moonshot").strip().lower(),
            chat_base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/"),
            chat_api_key=os.getenv("MOONSHOT_API_KEY", "").strip() or _read_harness_secret("MOONSHOT_API_KEY"),
            chat_model=os.getenv("HORMOZI_CHAT_MODEL", "kimi-k3"),
            reasoning_effort=os.getenv("KIMI_REASONING_EFFORT", "high").strip().lower(),
            embedding_model=os.getenv("HORMOZI_EMBEDDING_MODEL", "nomic-embed-text"),
            database_path=Path(os.getenv("HORMOZI_DATABASE", data_dir / "knowledge.db")).expanduser().resolve(),
            business_profile_path=Path(os.getenv("HORMOZI_BUSINESS_PROFILE", data_dir / "business.md")).expanduser().resolve(),
            motions_dir=motions_dir,
            max_vector_scan=max(100, int(os.getenv("HORMOZI_MAX_VECTOR_SCAN", "25000"))),
        )
