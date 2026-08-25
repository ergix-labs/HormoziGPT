from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .ollama import ModelClient
from .prompts import DAILY_MOTIONS_PROMPT, SYSTEM_MESSAGE
from .store import KnowledgeStore


def _parse_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    value = json.loads(cleaned)
    if not isinstance(value, dict) or not isinstance(value.get("motions"), list):
        raise ValueError("Model response did not contain a motions array")
    return value


def generate_daily_motions(
    store: KnowledgeStore,
    client: ModelClient,
    profile: str,
    output_dir: Path,
    count: int = 3,
    day: date | None = None,
) -> dict[str, Any]:
    run_date = day or date.today()
    query = "highest leverage action today for offers leads sales retention cash flow and delivery bottlenecks"
    hits = store.search(query, embedder=client.embed, top_k=10)
    context = "\n\n".join(f"{hit.title}: {hit.text}" for hit in hits) or "No knowledge indexed yet."
    prompt = DAILY_MOTIONS_PROMPT.format(count=count, date=run_date.isoformat(), profile=profile.strip(), context=context)
    result = _parse_json(client.chat([{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}], json_mode=True))
    motions = result["motions"][:count]
    required = {"title", "why_now", "action", "metric", "target", "timebox_minutes", "source_titles"}
    if len(motions) != count or any(not isinstance(item, dict) or not required.issubset(item) for item in motions):
        raise ValueError("Model returned incomplete daily motions")
    allowed_sources = {hit.title for hit in hits}
    normalized = []
    for motion in motions:
        target = str(motion["target"]).strip()
        if target in {"", "0", "0.0"}:
            target = "1 measured baseline"
        try:
            timebox = max(15, min(120, int(motion["timebox_minutes"])))
        except (TypeError, ValueError):
            timebox = 30
        source_titles = motion["source_titles"] if isinstance(motion["source_titles"], list) else []
        normalized.append({
            "title": str(motion["title"]).strip(),
            "why_now": str(motion["why_now"]).strip(),
            "action": str(motion["action"]).strip(),
            "metric": str(motion["metric"]).strip(),
            "target": target,
            "timebox_minutes": timebox,
            "source_titles": [str(title) for title in source_titles if str(title) in allowed_sources],
        })
    payload = {"date": run_date.isoformat(), "model": client.chat_model, "motions": normalized}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run_date.isoformat()}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return payload
