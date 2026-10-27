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
