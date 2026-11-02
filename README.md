# EchoMCP

EchoMCP is a Python RAG and MCP server scaffold. It currently ships a FastAPI gateway, a replaceable retrieval interface with an in-memory development adapter, a basic agent workflow, and an MCP JSON-RPC HTTP endpoint.

## Current capabilities

- `POST /mcp` supports MCP initialization, tools, resources, prompts, ping, and tool calls.
- Tools currently available: `semantic_search`, `keyword_search`, `search_database`, `list_sources`, and `health_check`.
- `sources://list`, `system_prompt`, and `safety_prompt` are exposed as MCP resources/prompts.
- The agent worker routes time-sensitive or low-confidence requests through a web-search adapter when one is injected; otherwise it returns an explicit low-confidence response.
- LLM adapters can be injected in priority order and are guarded by per-provider circuit breakers. Each real network adapter must enforce its own timeout.
- `POST /query`, `GET /documents`, `GET /health`, and the legacy `/mcp/semantic_search` route remain available.
- Local storage and providers are process-local fakes. Data is lost on restart and is not shared between replicas.

The larger target architecture, including Postgres/pgvector, Redis, authenticated user sessions, external LLM and search providers, voice adapters, and distributed observability, is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Those integrations are not represented as operational services in this scaffold yet; provider injection is the seam for adding them.

## Run locally

```powershell
python -m pip install -r requirements.txt
$env:APP_ENV = "development"
python -m uvicorn api.main:app --reload
```

Local development permits MCP requests without a token. To exercise authentication, set `MCP_BEARER_TOKEN` and send `Authorization: Bearer <token>`.

## Container deployment

Set a high-entropy `MCP_BEARER_TOKEN` in the deployment environment, then run:

```powershell
docker compose -f deployment/docker-compose.yml up --build -d
```

Compose binds the API to loopback only. Terminate TLS at a trusted reverse proxy and keep that proxy on a private network. Do not trust forwarded client-IP headers from arbitrary sources; configure Uvicorn's trusted proxy list to match the actual proxy addresses before using IP-based controls. The API uses a single-process in-memory limiter/storage model, so production deployments that need multi-user authorization, shared limits, or durable data must add a shared identity, repository, and rate-limit backend before exposing it publicly.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `production` requires an MCP bearer token |
| `MCP_BEARER_TOKEN` | empty | Shared token protecting MCP endpoints when set |
| `MAX_REQUEST_BYTES` | `1048576` | Request body limit (1 KiB–16 MiB) |
| `SIMILARITY_THRESHOLD` | `0.72` | Retrieval confidence configuration |
| `PROMPT_VERSION` | `v1` | Prompt asset version |
| `LLM_PROVIDER` | `fake` | Provider adapter selection (fake only is wired) |
| `EMBEDDING_PROVIDER` | `fake` | Embedding adapter selection (fake only is wired) |
| `VECTOR_STORE` | `memory` | Repository selection (memory only is wired) |

MCP clients should POST JSON-RPC 2.0 requests to `/mcp` and use the returned server capabilities and tool schemas. Error responses intentionally avoid exposing provider exceptions or secrets.
