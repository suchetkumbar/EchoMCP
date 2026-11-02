# EchoMCP

EchoMCP is a Python RAG and MCP server scaffold. It currently ships a FastAPI gateway, a replaceable retrieval interface with an in-memory development adapter, a basic agent workflow, and an MCP JSON-RPC HTTP endpoint.

## Current capabilities

- `POST /mcp` supports MCP initialization, tools, resources, prompts, ping, and tool calls.
- Tools currently available: `semantic_search`, `keyword_search`, `search_database`, `list_sources`, and `health_check`.
- `sources://list`, `system_prompt`, and `safety_prompt` are exposed as MCP resources/prompts.
- The agent worker routes time-sensitive or low-confidence requests through a web-search adapter when one is injected; otherwise it returns an explicit low-confidence response.
- LLM adapters can be injected in priority order and are guarded by per-provider circuit breakers. Each real network adapter must enforce its own timeout.
- RAG ingestion now chunks documents with overlap, batches embeddings when supported, retains chunk metadata, and replaces previously indexed document chunks on re-ingest.
- An HTTP embedding adapter supports a private service exposing `POST /embed` with `{ "texts": [...] }` and `{ "embeddings": [[...]] }`; response count, dimension, numeric values, TLS URL form, and finite vectors are validated.
- `POST /query`, `GET /documents`, `GET /health`, dependency-aware `GET /health/ready`, and the legacy `/mcp/semantic_search` route remain available.
- Local defaults use an in-memory fake store. Compose runs PostgreSQL/pgvector and Redis with persistent volumes; set a conversation UUID on `/query` to use Redis-backed short-term memory.

External LLM and search providers, voice adapters, and distributed observability remain future integrations; provider injection is the seam for adding them.

## Run locally

```powershell
python -m pip install -r requirements.txt
$env:APP_ENV = "development"
python -m uvicorn api.main:app --reload
```

Local development permits MCP requests without a token. To exercise authentication, set `MCP_BEARER_TOKEN` and send `Authorization: Bearer <token>`.

## Container deployment

Set high-entropy, URL-safe `MCP_BEARER_TOKEN`, `POSTGRES_PASSWORD`, and `REDIS_PASSWORD` values in the deployment environment (or an untracked `.env` file), then run:

```powershell
docker compose -f deployment/docker-compose.yml up --build -d
```

Compose binds the API to loopback only. Postgres and Redis are not published to the host and use a private internal network plus persistent volumes. Terminate TLS at a trusted reverse proxy. Database schema is initialized idempotently on application startup; use a controlled backup and restore process for the persistent volumes. Redis conversation memory has a 30-minute idle TTL and stores at most 20 turns per conversation.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `production` requires an MCP bearer token |
| `MCP_BEARER_TOKEN` | empty | Shared token protecting MCP endpoints when set |
| `MAX_REQUEST_BYTES` | `1048576` | Request body limit (1 KiB–16 MiB) |
| `SIMILARITY_THRESHOLD` | `0.72` | Retrieval confidence configuration |
| `PROMPT_VERSION` | `v1` | Prompt asset version |
| `LLM_PROVIDER` | `fake` | Provider adapter selection (fake only is wired) |
| `EMBEDDING_PROVIDER` | `fake` | `fake` or `http` embedding adapter |
| `EMBEDDING_SERVICE_URL` | empty | Base URL for the configured HTTP embedding service |
| `EMBEDDING_API_KEY` | empty | Optional bearer credential sent to the embedding service |
| `EMBEDDING_DIMENSION` | `1024` | Expected dimension for HTTP embedding vectors |
| `CHUNK_SIZE` | `1200` | Maximum normalized characters per indexed chunk |
| `CHUNK_OVERLAP` | `160` | Character overlap between neighboring chunks |
| `VECTOR_STORE` | `memory` | `memory` or `postgres` pgvector repository |
| `DATABASE_URL` | empty | PostgreSQL connection URL; required for `VECTOR_STORE=postgres` |
| `REDIS_URL` | empty | Optional Redis URL for short-term conversation memory |

Postgres schema setup creates the `vector` extension, a document/chunk schema, an HNSW cosine index, and a full-text GIN index. The vector column dimension is pinned to the selected embedding provider on first initialization; changing embedding dimensions requires a deliberate re-embedding migration. `/health` is a liveness check; `/health/ready` checks configured storage dependencies. Supply a `conversation_id` UUID in `/query` to read and append the bounded Redis conversation window.

MCP clients should POST JSON-RPC 2.0 requests to `/mcp` and use the returned server capabilities and tool schemas. Error responses intentionally avoid exposing provider exceptions or secrets.
