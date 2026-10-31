# rag-document-qa

## What this is
- A single-process RAG service. It ingests TXT/MD/PDF/DOCX/HTML/CSV into ChromaDB and answers questions with `[S1]`-style citations. App state lives in SQLite, and a static web UI is served at `/ui/`.
- Python `>=3.11,<3.14` (CI and Dockerfile use 3.12). Stack: FastAPI, pydantic-settings, ChromaDB 1.5, httpx, pypdf, python-docx, plus optional Sentence Transformers/torch (`local` extra).
- By default, answers come from an offline extractive generator. It can also use any OpenAI-compatible `/v1/chat/completions` server, with fallback providers.

## How to run it
- Setup (README/Makefile): `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[local,dev]"`, or `make install PYTHON=python3.12`. Took ~4.5 min with pip (torch download).
- Lighter setup without torch: `pip install -e ".[dev]"` (~11s with `uv pip`). All 206 tests still pass. The app then needs `RAG_EMBEDDING_BACKEND=hashing` or `openai-compatible`.
- Config: `cp .env.example .env` (optional; every setting has a default). All vars use the `RAG_` prefix and are read from `.env` in the CWD. `app/config.py` is the source of truth.
- Run: `uvicorn app.main:create_app --factory --reload` (`make run`), default port 8000. URLs: UI `/ui/`, Swagger `/docs`, health `/api/v1/health`.
- Verified offline run, ready in ~2s: `RAG_EMBEDDING_BACKEND=hashing uvicorn app.main:create_app --factory --port 8765`. Point `RAG_SQLITE_PATH`/`RAG_CHROMA_PATH` at fresh dirs if `data/` was built with another embedder (see Gotchas).
- Tests: `python -m pytest -q` (`make test`). Result: 206 passed in ~5s.
- Single test: `python -m pytest tests/test_api.py::test_query_endpoint_returns_grounded_sources -q`. One file: `python -m pytest tests/test_guardrails.py -q`.
- Coverage: `python -m pytest --cov=app --cov-report=term-missing` (97%, ~5s).
- Lint: `ruff check app tests scripts` is clean. CI and `make lint` run `ruff check app tests`.
- Keys: none are needed for tests or the default run. `RAG_LLM_API_KEY`, `RAG_EMBEDDING_API_KEY` and `RAG_ADMIN_API_KEY` go only in `.env`. `.env.example` has commented placeholders.
- Docker (unverified, not run):
  - `docker build -t rag-document-qa .`
  - `docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-document-qa`
  - `docker compose up --build`: reads `.env`, mounts `./data`.
  - For the slim image without torch, use `--build-arg LOCAL_EMBEDDINGS=false`.
- Demo clients (not run): `python scripts/ingest_sample.py` and `python scripts/smoke_test.py`. Both need a running server and send `RAG_API_KEY` when set.

## Project layout
- `app/main.py`: the `create_app()` factory. Holds the lifespan (migrations, failing interrupted ingestions), request-id/timing middleware, `AppError` handler, routers and UI mount.
- `app/container.py`: `build_container()` DI wiring. Picks the embedder, vector store, LLM and answer generator from settings.
- `app/config.py`: `Settings` (pydantic-settings, `RAG_` prefix, cross-field validators).
- `app/api/`: `routes.py` holds every endpoint under `/api/v1` (public, API-key and admin routers). `auth.py` handles bearer keys. `ui.py` mounts the UI.
- `app/agents/`: ingestion, retrieval and answer agents, plus `orchestrator.py` (`RAGOrchestrator.run`/`stream`).
- `app/services/`: parsing, chunking, embeddings, `vector_store.py` (Chroma + in-memory), `llm.py` (`FallbackLLM`), prompting, query rewriter, guardrails, and the document/collection/conversation services.
- `app/db/`: `sqlite.py` (connection, WAL), `migrations.py` (versioned), `repositories.py` (raw `sqlite3`, frozen dataclass records).
- `app/schemas.py`: Pydantic request/response models, which are the public API contract.
- `app/static/`: the UI in plain HTML/CSS/JS with no build step. Shipped as setuptools package data.
- `tests/`: pytest. `conftest.py` holds all shared fixtures and `MockLLM`.
- `data/`: gitignored and created at runtime. Holds `rag.db` and `chroma/`.

## Architecture in short
- Ingest flow: `POST /documents/text|upload` → `FileParser` (extension, plus magic bytes for PDF/DOCX) → `IngestionAgent` → `DocumentService.register()`.
- `register()` dedups by SHA-256 per collection and creates the `processing` row.
- `index()` runs `TextChunker` → embedder → `VectorStore.upsert_document` → `ready`. On failure it deletes the vectors and marks the document `failed`.
- `?background=true` runs `index()` as a FastAPI background task and returns 202. On restart, documents still `processing` are marked `failed`.
- Query flow: `POST /query` or `/query/stream` → `RAGOrchestrator` → guardrails → `QueryRewriter` (only with history and an LLM) → `RetrievalAgent` (over-fetch, `min_relevance_score`, token budget) → `AnswerAgent`.
- `AnswerAgent` calls `ExtractiveAnswerGenerator`, or `OpenAICompatibleAnswerGenerator` via `FallbackLLM`. The query, sources, messages and `llm_calls` rows are then saved to SQLite.
- Streaming uses SSE `source`, answer-piece, `end` and `error` events. The query is saved only after the last piece.
- SQLite is canonical for lifecycle, audit and collections. Chroma holds only vectors. Collections filter retrieval by document ids loaded from SQLite.

## Conventions
- Every module starts with `from __future__ import annotations`. Code uses full type hints and `X | None`.
- Ruff config in `pyproject.toml`: line length 100, py311, rules `E,F,I,B,UP`. Only `ruff check` is enforced. The code is not `ruff format`-clean (14 files would change).
- DI uses constructor injection against Protocols (`Embedder`, `VectorStore`, `AnswerGenerator`). `create_app()` and `build_container()` accept overrides for tests.
- Errors: raise `AppError` subclasses from `app/core/exceptions.py` (404/409/415/502/401/403). They render as `{"error","message","request_id"}`. A bare `ValueError` becomes a 400.
- Logging: `logger = logging.getLogger(__name__)`, configured as INFO `basicConfig` in `app/core/logging.py`.
- New settings need `Field(description=...)`, an `.env.example` entry and a README line (rule in `docs/IMPLEMENTATION_PLAN.md`).
- Test names are plain functions like `test_<behaviour_in_words>() -> None`.
- API tests use the `client` fixture (TestClient, extractive) or `llm_client` (wired to `MockLLM` on `httpx.MockTransport`).
- Unit tests use the `container`, `vector_store` (`InMemoryVectorStore`), `embedder` (`HashingEmbedder`) and `make_hit` fixtures. The `settings` fixture writes to `tmp_path`.
- Tests are fully offline. The few Chroma tests use `pytest.importorskip("chromadb")` with `EphemeralClient`.
- Commits: the subject is `Phase N: <topic>` or a sentence-case summary, with no type prefix. The body uses `- ` bullets wrapped near 72 chars and ends with a `Co-Authored-By:` trailer.

## Do NOT
- Edit a shipped migration (v1–5) in `app/db/migrations.py`. Databases that recorded it never re-run it. Append a new `Migration`.
- Rename or remove endpoints, response fields or the `[S1]` citation format. Only add fields: `app/static/app.js` and the tests depend on them.
- Write uploaded files to disk. By design, only extracted text, chunks and vectors are kept.
- Add network access or real keys to tests. Use `MockLLM`, `HashingEmbedder` and `InMemoryVectorStore`.
- Commit `.env` or real keys. `.env.example` holds placeholders only.
- Run `ruff format` on whole files, or rely on the VS Code format-on-save in `.vscode/settings.json`. It would reformat unrelated code.
- Run `make clean` without asking. It deletes `data/rag.db` and `data/chroma`, the user's indexed documents.
- Add dependencies without asking. Keep `requirements.txt` mirrored with `pyproject.toml`.
- Add a JS framework or build step to `app/static/`.
- Change the embedding backend or model defaults without also changing `RAG_CHROMA_COLLECTION`.
- Commit phase work before the owner has reviewed it (working rule in `docs/IMPLEMENTATION_PLAN.md`).

## Gotchas
- On the dev Mac the default `python3` is 3.15, which `pyproject.toml` rejects. Use `python3.12`, e.g. `make install PYTHON=python3.12`.
- Port 8000 is used everywhere: `make run`, Dockerfile, compose, and hardcoded in both `scripts/*.py`. If 8000 is taken, pass `--port`; the scripts still only reach 8000.
- Shell `RAG_*` env vars leak into tests. The `settings` fixture only disables `.env` (`_env_file=None`). For example, `RAG_AUTH_ENABLED=true` makes 5 of 6 tests in `tests/test_api.py` fail (401s). Unset them first.
- `.env`, `data/rag.db` and `data/chroma` resolve relative to the CWD, so run from the repo root.
- The Chroma collection stores the embedder signature. If the backend or model changes but `RAG_CHROMA_COLLECTION` does not, `ChromaVectorStore.__init__` raises `ConflictError` and the app fails to start.
- The default backend is `sentence-transformers`. Without the `local` extra, `build_container` raises `RuntimeError`. The first ingestion downloads the model (~90 MB per README, unverified).
- `RAG_MIN_RELEVANCE_SCORE=0.15` is tuned for MiniLM. Other embedding models need retuning (see README).
- `TEST_RESULTS.txt` is stale (21 tests, no Chroma). Trust a fresh `pytest` run instead.
- `pip install -e` creates `rag_document_qa.egg-info/` (gitignored).
- CI on Linux installs torch from `https://download.pytorch.org/whl/cpu`. On macOS, plain pip from PyPI worked.

## Known issues
- `pip install -e .` with the default python3 3.15 fails: `ERROR: Package 'rag-document-qa' requires a different Python: 3.15.0 not in '<3.14,>=3.11'`. Under 3.12, everything above passed.

## Where things are documented
- `README.md`: features, quick start, configuration table, LLM/embedding providers, auth, guardrails, Docker.
- `docs/ARCHITECTURE.md`: Mermaid flow diagrams, design decisions, error-behaviour table, scaling path, security gaps.
- `docs/API.md`: curl request/response examples per endpoint and the common error body.
- `docs/IMPLEMENTATION_PLAN.md`: phases 0–8, working rules, backlog, decisions.
- `.env.example`: every setting, with comments.
- `.github/workflows/ci.yml`: lint and test on Python 3.12, plus a slim Docker build and health check.

## Current status / next steps
- `docs/IMPLEMENTATION_PLAN.md` marks every phase done (2026-10-28). Only `main` exists on the remote, and the code has no TODO/FIXME markers.
- Unscheduled backlog in that file: Postgres + pgvector, hybrid search, an OpenAI-compatible endpoint, per-collection retrieval settings, idempotency keys, a token-aware chunker.
