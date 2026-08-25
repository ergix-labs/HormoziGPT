# Ergix Operator (HormoziGPT fork)

Ergix Operator turns an owned business-knowledge corpus into grounded answers and daily suggested motions. It runs locally through [Ollama](https://ollama.com/), defaults to `qwen3.5:9b`, and stores indexed knowledge on the operator's machine.

This is an Ergix Labs fork of `wombyz/HormoziGPT`. See [UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md) for provenance and licensing constraints. The app does not claim to be Alex Hormozi and is not affiliated with or endorsed by Alex Hormozi or Acquisition.com.

## What changed

- OpenAI and Pinecone are no longer required.
- Chat runs against an Ollama-hosted open-weight model.
- Embeddings run locally with `nomic-embed-text`.
- Markdown, text, PDF, CSV, JSON, JSONL, and HTML can be indexed in bulk.
- Content is hashed, deduplicated, chunked, and stored in a local SQLite hybrid search index.
- A stable JSON daily-motions command is ready for the Ergix Agent Harness to consume.
- Streamlit uses native chat components rather than rendering user input as unsafe HTML.

## Local setup

Requires Python 3.9+ and Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Copy `.env.example` to `.env` if you want to override the defaults.

## Add knowledge

Only ingest material Ergix owns or has permission to use.

```bash
python -m ergix_hormozi ingest /path/to/approved/transcripts /path/to/playbooks
python -m ergix_hormozi stats
```

Unchanged files are skipped on later runs, so the same folders can be indexed repeatedly.

## Ask questions

```bash
python -m ergix_hormozi ask "What is the highest-leverage constraint in our sales motion?"
```

## Generate daily suggested motions

Fill in `data/business.md`, then run:

```bash
python -m ergix_hormozi daily --count 3
```

The command prints JSON and saves the same contract to `data/motions/YYYY-MM-DD.json`:

```json
{
  "date": "2026-08-25",
  "model": "qwen3.5:9b",
  "motions": [
    {
      "title": "Call five qualified leads",
      "why_now": "Pipeline is the current constraint.",
      "action": "Call the five warmest qualified leads before noon.",
      "metric": "completed calls",
      "target": "5",
      "timebox_minutes": 45,
      "source_titles": ["Ergix sales playbook"]
    }
  ]
}
```

## Run the interface

```bash
streamlit run app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Scaling note

The current local SQLite index is deliberately operationally simple. It combines FTS5 retrieval with local embedding reranking and is suitable for the first substantial corpus. When the approved corpus grows beyond a single-machine scan budget, keep the ingestion and JSON contracts and replace only `KnowledgeStore` with Qdrant or another open vector engine.
