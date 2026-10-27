# API Examples

Base URL:

```text
http://127.0.0.1:8000/api/v1
```

## Health

```bash
curl http://127.0.0.1:8000/api/v1/health
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
