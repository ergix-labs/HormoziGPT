# Ergix Operator (HormoziGPT fork)

Ergix Operator turns an owned business-knowledge corpus into grounded answers and daily suggested motions. It uses Moonshot's `kimi-k3` for reasoning, local [Ollama](https://ollama.com/) embeddings for retrieval, and stores indexed knowledge on the operator's machine.

This is an Ergix Labs fork of `wombyz/HormoziGPT`. See [UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md) for provenance and licensing constraints. The app does not claim to be Alex Hormozi and is not affiliated with or endorsed by Alex Hormozi or Acquisition.com.

## What changed

- OpenAI and Pinecone are no longer required.
- Chat runs against Moonshot-hosted Kimi K3 with configurable reasoning effort.
- Embeddings run locally with `nomic-embed-text`.
- Markdown, text, PDF, CSV, JSON, JSONL, and HTML can be indexed in bulk.
- Content is hashed, deduplicated, chunked, and stored in a local SQLite hybrid search index.
- A stable JSON daily-motions command is ready for the Ergix Agent Harness to consume.
- Streamlit uses native chat components rather than rendering user input as unsafe HTML.

## Local setup

Requires Python 3.9+, Ollama, and a [Moonshot API key](https://platform.moonshot.ai/console/api-keys).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull nomic-embed-text
cp .env.example .env
```

On an Ergix workstation, the app automatically reads `MOONSHOT_API_KEY` from the local harness store after the Infisical row `kimi` has been synced. To use a standalone key instead, add it to `.env` (which is ignored by Git):

```dotenv
MOONSHOT_API_KEY=your_key_here
```

The default reasoning level is `high`. Set `KIMI_REASONING_EFFORT=max` for the hardest runs or `low` for faster, less expensive runs.

## Data boundary

Original files, embeddings, and the full SQLite index remain local. The retrieved passages needed for each answer or daily-motion run are sent to Moonshot with the prompt. Do not ingest material you are not permitted to send to Moonshot.

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
  "model": "kimi-k3",
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
