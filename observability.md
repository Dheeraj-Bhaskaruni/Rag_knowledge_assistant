# Observability Setup

This project uses [Langfuse](https://langfuse.com) for tracing and observability.

## Setup

1. Create a project on Langfuse Cloud or host your own.
2. Get your API keys (Public Key, Secret Key).
3. Set them in `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Features

- **Tracing**: Follow the request through retrieval and generation.
- **Cost Tracking**: See token usage and cost per request.
- **Scores**: Log user feedback (thumbs up/down) if implemented.
