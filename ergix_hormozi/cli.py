from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .daily import generate_daily_motions
from .ingest import ingest_paths
from .ollama import OllamaClient
from .prompts import SYSTEM_MESSAGE, grounded_user_prompt
from .store import KnowledgeStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergix-operator")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest", help="Index files or folders")
    ingest.add_argument("paths", nargs="+", type=Path)
    ask = subcommands.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question")
    daily = subcommands.add_parser("daily", help="Generate today's suggested motions")
    daily.add_argument("--count", type=int, default=3)
    subcommands.add_parser("stats", help="Show local index counts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    store = KnowledgeStore(settings.database_path, max_vector_scan=settings.max_vector_scan)
    client = OllamaClient(settings.ollama_base_url, settings.chat_model, settings.embedding_model)
    if args.command == "ingest":
        output = ingest_paths(args.paths, store, client.embed, settings.embedding_model)
    elif args.command == "ask":
        hits = store.search(args.question, client.embed, top_k=6)
        output = {"answer": client.chat([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": grounded_user_prompt(args.question, hits)}]), "sources": sorted({hit.title for hit in hits})}
    elif args.command == "daily":
        profile = settings.business_profile_path.read_text("utf-8") if settings.business_profile_path.exists() else ""
        output = generate_daily_motions(store, client, profile, settings.motions_dir, count=max(1, min(7, args.count)))
    else:
        output = store.stats()
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
