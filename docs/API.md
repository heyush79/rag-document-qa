# API Examples

Base URL:

```text
http://127.0.0.1:8000/api/v1
```

## Health

```bash
curl http://127.0.0.1:8000/api/v1/health
```

```json
{"status": "ok", "sqlite": "ok", "chroma": "ok", "embedding_backend": "sentence-transformers"}
```

`status` is `degraded` when SQLite or ChromaDB can't be reached. The response is always HTTP 200.

### Deep health check

`?deep=true` also embeds a short test string and calls `GET /models` on each LLM provider, so it confirms the embedding model loads (or the embedding API answers) and that each provider is reachable with its key. It is slower, and the first call may load the local embedding model.

```bash
curl "http://127.0.0.1:8000/api/v1/health?deep=true"
```

```json
{
  "status": "degraded",
  "sqlite": "ok",
  "chroma": "ok",
  "embedding_backend": "openai-compatible",
  "embedder": {"status": "ok"},
  "llm": [
    {"status": "error", "detail": "[Errno 61] Connection refused", "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b"},
    {"status": "ok", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"}
  ]
}
```

`llm` lists the primary provider first, then the fallbacks. It is empty with the extractive answer provider. `detail` is only present on errors.

## Authentication

With `RAG_AUTH_ENABLED=true`, every request below needs an API key:

```bash
curl http://127.0.0.1:8000/api/v1/stats -H "Authorization: Bearer rag_..."
```

`/`, `/health` and the docs need no key. A missing, unknown or revoked key gets HTTP 401 with a `WWW-Authenticate: Bearer` header and the common error body, with `"error": "unauthorized"`. The admin key (`RAG_ADMIN_API_KEY`) works on every route as well.

### Manage API keys

These routes always need the admin key, even while authentication is off. Without `RAG_ADMIN_API_KEY` configured they answer 403.

```bash
# Create: the key is returned once, then only its hash is kept.
curl -X POST http://127.0.0.1:8000/api/v1/api-keys \
  -H "Authorization: Bearer $RAG_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "reporting"}'
```

```json
{
  "id": "4f1c...",
  "name": "reporting",
  "prefix": "rag_k3pX9aQb",
  "created_at": "2026-10-27T18:50:12.345678+00:00",
  "last_used_at": null,
  "revoked_at": null,
  "key": "rag_k3pX9aQb..."
}
```

```bash
# List: prefixes and last use, never the keys.
curl http://127.0.0.1:8000/api/v1/api-keys -H "Authorization: Bearer $RAG_ADMIN_API_KEY"

# Revoke. Revoking again is a 204 too; an unknown id is a 404.
curl -i -X DELETE http://127.0.0.1:8000/api/v1/api-keys/KEY_ID \
  -H "Authorization: Bearer $RAG_ADMIN_API_KEY"
```

## Ingest raw text

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/text \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "title": "Atlas Search Notes",
  "source": "internal-wiki",
  "text": "The Atlas Search service indexes documentation every night at 02:00 UTC. Incremental updates are consumed from Kafka and become searchable within five minutes."
}
JSON
```

A new document returns HTTP `201`. Re-ingesting identical normalized text returns HTTP `200` with:

```json
{
  "duplicate": true
}
```

## Upload a file

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@sample_data/company_handbook.md" \
  -F "title=Company Handbook"
```

Supported types:

- `.txt`
- `.md`
- `.pdf`

Scanned image-only PDFs require OCR, which is deliberately not hidden behind wishful thinking in this version.

## List documents

```bash
curl "http://127.0.0.1:8000/api/v1/documents?limit=20&offset=0"
```

## Get one document

```bash
curl http://127.0.0.1:8000/api/v1/documents/DOCUMENT_ID
```

## Ask across all indexed documents

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "question": "How quickly do incremental updates become searchable?",
  "top_k": 4
}
JSON
```

`sources` lists only the chunks the answer was generated from: those scoring at least `RAG_MIN_RELEVANCE_SCORE` that fit the model's context budget, at most `top_k`. A citation `[Sn]` in the answer refers to `sources[n-1]`, the entry with `rank` n. It can be empty when nothing relevant was found.

## Ask within one document

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "question": "What is the leave policy?",
  "top_k": 3,
  "document_id": "DOCUMENT_ID"
}
JSON
```

## Stream an answer

`POST /api/v1/query/stream` takes the same body as `/query` and returns Server-Sent Events (`text/event-stream`), so a client can show the answer while it is being generated. `-N` stops curl from buffering:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leave days do employees receive?"}'
```

Each event is one `data:` line holding a JSON object, followed by a blank line:

```text
data: {"type": "source", "sources": [{"rank": 1, "chunk_id": "...", "text": "...", "score": 0.56, ...}]}

data: {"type": "answer", "answer": "Full-time employees receive 24 paid leave"}

data: {"type": "answer", "answer": " days per calendar year [S1]."}

data: {"type": "end", "query_id": "...", "latency_ms": 2310.4}
```

| Event | When | Fields |
|---|---|---|
| `source` | Always first | `sources`: the same list `/query` returns |
| `answer` | Zero or more times | `answer`: the next piece of text. Join them all to get the full answer |
| `end` | Last, on success | `query_id`, `latency_ms` |
| `error` | Last, on failure | `error`: the code a JSON error body would carry, or `internal_error`; `message` |

The offline extractive provider sends its whole answer as one `answer` event.

The query is saved, with the joined answer, just before `end`. If the model fails partway through, the stream ends with an `error` event and nothing is saved. Nothing is saved if the client disconnects early, either.

Errors found before generation starts, such as an empty question, return the usual JSON error body with its HTTP status instead of a stream.

## Query history

```bash
curl "http://127.0.0.1:8000/api/v1/queries?limit=10"
```

## Statistics

```bash
curl http://127.0.0.1:8000/api/v1/stats
```

```json
{
  "document_count": 2,
  "vector_count": 9,
  "query_count": 3,
  "llm_usage": {"calls": 4, "failed_calls": 1, "prompt_tokens": 1441, "completion_tokens": 96}
}
```

`llm_usage` totals every attempt to call an LLM provider, including failed ones and fallbacks. Token counts come from the provider's `usage` field; when a provider reports none, they are estimated at about four characters per token.

## Delete a document

```bash
curl -i -X DELETE http://127.0.0.1:8000/api/v1/documents/DOCUMENT_ID
```

The endpoint deletes both the SQLite document row and matching ChromaDB vectors.

## Common error body

```json
{
  "error": "not_found",
  "message": "Document '...' was not found",
  "request_id": "..."
}
```
