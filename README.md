---
title: RAG Knowledge Assistant
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: apps/web/app.py
pinned: false
---

# RAG Knowledge Assistant

A production-ready RAG system featuring:
- **Grounded Answers**: Strict ciation of sources.
- **Evaluation**: Integrated offline eval harness.
- **Observability**: Langfuse tracing.
- **Deployment**: Ready for Hugging Face Spaces.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install uv
   uv sync
   ```

2. **Environment**:
   Copy `.env.example` to `.env` and set your keys.
   ```bash
   cp .env.example .env
   ```

## Usage

### Run Locally
```bash
python apps/web/app.py
```

### Ingestion
```bash
python -m services.rag.ingest --input sample_docs --out data/processed
python -m services.rag.index --processed data/processed --out data/index
```

### Evaluation
```bash
python -m eval.run_eval --data eval/sample_eval.jsonl
```

## Observability
See [observability.md](observability.md) for details on setting up Langfuse.
