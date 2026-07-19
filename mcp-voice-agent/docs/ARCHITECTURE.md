# MCP-Powered Voice Agent — System & MCP Design

Status: **Design confirmed** (LangGraph multi-agent · Redis+Postgres memory · hybrid tool routing · Streamable HTTP MCP transport · service-oriented Docker Compose). Build mode: adapters + local fakes first, real provider SDKs wired behind the same interfaces so keys can be dropped in later with zero code changes.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Next.js Frontend                            │
│   Voice UI (waveform, mic button)    │    Admin Dashboard             │
└───────────────┬───────────────────────────────────┬──────────────────┘
                 │ WebRTC (audio)                     │ REST/WS (JWT)
                 ▼                                     ▼
         ┌───────────────┐                    ┌──────────────────┐
         │   LiveKit      │                    │    FastAPI        │
         │  Voice Gateway │◄──────WS/HTTP──────│   API Gateway      │
         └───────┬────────┘                    │ (auth, rate-limit, │
                 │ audio in/out                 │  REST + WS)        │
                 │                              └─────────┬─────────┘
                 ▼                                         │ internal HTTP
         ┌───────────────┐   final transcript               ▼
         │  STT Adapter   │──────────────────────►┌──────────────────┐
         │ (AssemblyAI)   │                        │   Agent Worker    │
         └───────────────┘                        │  (LangGraph graph)│
         ┌───────────────┐   audio out             └─────────┬─────────┘
         │  TTS Adapter   │◄────────────────────────────────┘ tool calls
         │(Kokoro/Piper)  │                                    │ Streamable HTTP
         └───────────────┘                                     ▼
                                                     ┌──────────────────┐
                                                     │   MCP Server      │
                                                     │(tools/resources/  │
                                                     │  prompts)          │
                                                     └────┬────────┬─────┘
                                        ┌───────────────────┘        └───────────────┐
                                        ▼                                             ▼
                             ┌────────────────────┐                     ┌──────────────────────┐
                             │ Embedding Service    │                     │  Web Search Adapter    │
                             │   (bge-m3)           │                     │ (Tavily → Firecrawl)   │
                             └──────────┬───────────┘                     └──────────────────────┘
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  Supabase Postgres + pgvector              │
                    │  documents · chunks · embeddings · users   │
                    │  conversation_history · feedback ·          │
                    │  audit_logs · search_cache                   │
                    └───────────────────┬─────────────────────────┘
                                        ▼
                               ┌─────────────────┐
                               │      Redis        │
                               │ session memory ·   │
                               │ tool-result cache · │
                               │ rate limiting        │
                               └─────────────────┘

LLM layer (used by Agent Worker):
┌───────────────────────────────────────────────────────────┐
│  LLMProvider interface                                       │
│   primary  → Ollama (Qwen3, local)                             │
│   fallback → Gemini → OpenAI → Claude → DeepSeek (in order)    │
└───────────────────────────────────────────────────────────┘

Observability sidecars on every service:
  OpenTelemetry Collector → Prometheus → Grafana
```

**Component roles**

| Component | Responsibility | Replaceable via |
|---|---|---|
| Next.js Frontend | Voice UI, admin dashboard, auth screens | — |
| LiveKit Voice Gateway | WebRTC room mgmt, raw audio in/out | `VOICE_GATEWAY_PROVIDER` |
| STT Adapter | Streams audio → partial/final transcripts | `STT_PROVIDER` |
| TTS Adapter | Text → audio stream | `TTS_PROVIDER` |
| FastAPI API Gateway | AuthN/Z, REST + WS, rate limiting, request routing | — |
| Agent Worker | LangGraph state machine, orchestrates tool calls & LLM turns | — |
| MCP Server | Exposes all tools/resources/prompts over Streamable HTTP | — |
| Embedding Service | bge-m3 inference, returns 1024-dim vectors | `EMBEDDING_PROVIDER` |
| Web Search Adapter | Tavily primary, Firecrawl fallback | `WEB_SEARCH_PROVIDER` |
| Supabase Postgres/pgvector | System of record + vector index | — |
| Redis | Short-term memory, caches, rate limiting | — |

Every arrow above is a **swappable interface**, not a hard dependency — each adapter has a fake/in-memory implementation for local dev so the whole stack runs before any real credentials exist.

---

## 2. MCP Architecture

- **MCP Clients:** the Agent Worker (primary), and the Admin Dashboard backend (for `list_sources`, `health_check`, direct doc management)
- **MCP Server:** one server, `mcp-server`, Streamable HTTP transport, versioned tool schemas
- **MCP Tools:** `search_database`, `semantic_search`, `keyword_search`, `web_search`, `get_document`, `save_feedback`, `list_sources`, `health_check`, `retrieve_conversation`, `summarize_conversation`
- **MCP Resources:** `document://{document_id}`, `conversation://{conversation_id}`, `sources://list`
- **MCP Prompts:** `system_prompt`, `developer_prompt`, `tool_calling_prompt`, `rag_prompt`, `fallback_prompt`, `web_search_prompt`, `safety_prompt` — all versioned (`v1`, stored under `prompts/`, loaded by version tag, never hardcoded inline in agent code)

---

## 3. Agent Architecture (LangGraph)

```
        ┌───────────┐
        │  ingest    │  load short-term memory (Redis), normalize transcript
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ classify   │  hybrid pre-filter: cheap heuristic + embedding
        │ (router)   │  similarity → candidate tool set {db | web | both}
        └─────┬─────┘
              ▼
        ┌───────────┐
        │ retrieve   │──► MCP: semantic_search + keyword_search → merge/rerank
        └─────┬─────┘
              ▼
      ┌─────────────────┐   score < threshold (default 0.72)
      │ confidence_gate   │────────────────────┐
      └─────┬─────────────┘                    ▼
     score >= threshold              ┌───────────────┐
              │                      │ web_fallback   │──► MCP: web_search
              ▼                      └─────┬─────────┘    (Tavily → Firecrawl)
        ┌───────────┐                       │
        │ synthesize │◄──────────────────────┘
        └─────┬─────┘
              ▼
        ┌───────────────┐
        │ memory_update  │──► MCP: retrieve_conversation write,
        └─────┬─────────┘     summarize_conversation (async, every N turns)
              ▼
           [response: text + citations + confidence]

  Any node failure ──► error_node ──► graceful degraded response + audit_logs write
```

**Escalation logic**
1. DB retrieval empty or below `similarity_threshold` → escalate to web
2. Tavily unavailable/5xx/timeout → auto-failover to Firecrawl
3. Both web providers fail → degrade honestly ("I don't have reliable current information on that") rather than guessing; log `search_unavailable` event
4. LLM primary (Qwen3/Ollama) unreachable → fallback chain Gemini → OpenAI → Claude → DeepSeek, first healthy provider wins; failure of all → return cached last-known-good response if available, else apologetic error

**Failure handling (applies to every external call)**
- Timeout budgets: STT partial ≤ 200ms, retrieval ≤ 800ms, LLM turn ≤ 4s, TTS first-byte ≤ 500ms
- Retry: exponential backoff, max 3 attempts, jitter
- Circuit breaker per external provider (opens after 3 consecutive failures, half-open probe after cooldown)
- Every failure is logged to `audit_logs` with correlation ID = conversation turn ID, and surfaced in tracing spans

---

## 4. Data Architecture

**Postgres (Supabase) — tables**

| Table | Purpose |
|---|---|
| `documents` | source metadata (title, source_url, type, indexed_at) |
| `chunks` | chunked text + parent document_id + position |
| `embeddings` | vector(1024) per chunk, pgvector HNSW index |
| `users` | Supabase Auth-linked profile, role |
| `conversation_history` | turns: role, content, tool_calls_used, timestamp |
| `feedback` | rating, comment, linked turn |
| `audit_logs` | every tool call / error / auth event, correlation ID |
| `search_cache` | query hash → result set + TTL, backstop above Redis |

pgvector index: HNSW (`m=16, ef_construction=64`) — better recall/latency trade-off than IVFFlat at this scale.

**Redis — key design**

| Pattern | Purpose | TTL |
|---|---|---|
| `session:{conversation_id}` | rolling short-term memory window (last N turns) | 30 min idle |
| `cache:embed:{sha256(query)}` | cached query embeddings | 24h |
| `cache:search:{sha256(query+filters)}` | cached retrieval results | 10 min |
| `ratelimit:{user_id}:{window}` | sliding-window rate limit counters | 1 min |
| `circuit:{provider}` | circuit breaker state | 60s cooldown |

**Message queue:** not introduced in v1 — Agent Worker is called synchronously from the API Gateway (`POST /internal/agent/invoke`) to keep voice-turn latency predictable. Noted as a documented scaling path: swap to Redis Streams consumer groups if horizontal Agent Worker scaling under load requires decoupling — no interface changes needed since the Agent Worker is already a separate service.

---

## 5. Integration Architecture

| External API | Used for | Auth |
|---|---|---|
| AssemblyAI Streaming | STT | API key header |
| Ollama (local) | Primary LLM (Qwen3) | none (internal network) |
| Gemini / OpenAI / Claude / DeepSeek | Fallback LLMs | API key, per-provider |
| Tavily | Primary web search | API key |
| Firecrawl | Fallback web search | API key |
| Supabase | Postgres, pgvector, Auth, Storage | service-role key (server-side only) |
| LiveKit | WebRTC voice transport | API key/secret, signed room tokens |
| Kokoro / Piper | TTS | local model / API key depending on mode |

**AuthN/Z flow**
```
Client ──login──► Supabase Auth ──JWT──► Client
Client ──(JWT in Authorization header)──► FastAPI API Gateway
FastAPI ──verifies JWT (JWKS)──► extracts user_id + role
FastAPI ──RBAC check (user/admin)──► allow/deny route
FastAPI ──service-role calls──► Supabase Postgres (never exposes service key to client)
```

---

## 6. MCP Tool Design

Each tool below: **Inputs / Outputs / Validation / Error handling.**

### `search_database`
- In: `query: str (1-2000 chars)`, `top_k: int (1-20, default 5)`, `filters: {source?, document_type?, date_range?}`
- Out: `[{chunk_id, document_id, content, score, source}]`
- Validation: non-empty query; parameterized queries only (no string-built SQL)
- Errors: DB timeout → retry x2 backoff → return `{results: [], degraded: true}` rather than raising; logged to `audit_logs`

### `semantic_search`
- In: `query: str`, `top_k: int`, `similarity_threshold: float (0-1, default 0.75)`, `filters`
- Out: `[{chunk_id, content, cosine_score, source}]`
- Validation: embedding dimension must match configured model (1024 for bge-m3)
- Errors: embedding service down → circuit breaker opens after 3 fails; cached embeddings served from `cache:embed:*` when available

### `keyword_search`
- In: `query: str`, `top_k: int`, `filters`
- Out: `[{chunk_id, content, ts_rank, source}]`
- Validation: escape/normalize `tsquery` special characters
- Errors: tsvector/GIN index missing → fallback to `ILIKE`, logs a warning (perf degradation, not a failure)

### `web_search`
- In: `query: str`, `max_results: int (1-10, default 5)`, `search_depth: "basic"|"advanced"`
- Out: `{results: [{title, url, snippet, published_date}], provider_used, confidence}`
- Validation: query length bounds; blocklist known malicious/low-quality domains
- Errors: Tavily timeout/5xx → auto-failover to Firecrawl; both fail → `{results: [], provider_used: null, error: "web_unavailable"}`, agent must degrade honestly

### `get_document`
- In: `document_id: uuid`
- Out: `{id, title, content, source, indexed_at, chunk_count}`
- Validation: UUID format; RBAC (owner or public flag)
- Errors: typed `NotFoundError` on missing doc; never leak raw DB errors

### `save_feedback`
- In: `conversation_id, turn_id, rating: "up"|"down", comment?: str (≤1000 chars)`
- Out: `{feedback_id, saved: true}`
- Validation: enum check on rating; comment sanitized; rate-limited (20/min/user)
- Errors: write failure → queued to Redis list for retry, never silently dropped

### `list_sources`
- In: `filters: {document_type?, tag?}`, `page: int`, `page_size: int (≤100)`
- Out: `[{document_id, title, source_url, indexed_at}]`
- Validation: page_size bound
- Errors: empty result set is a valid (non-error) response

### `health_check`
- In: `deep: bool = false`
- Out: `{status: "ok"|"degraded"|"down", components: {postgres, redis, embeddings, tavily, firecrawl, ollama}}`
- Validation: none
- Errors: never raises — always returns best-effort status object, even under total dependency failure

### `retrieve_conversation`
- In: `conversation_id, limit: int (default 20)`
- Out: `[{role, content, timestamp, tool_calls_used}]`
- Validation: RBAC — caller must own conversation or be admin
- Errors: unknown conversation_id → empty history (treated as a new conversation), not an error

### `summarize_conversation`
- In: `conversation_id, strategy: "incremental"|"full"`
- Out: `{summary_id, summary_text, token_count}`
- Validation: conversation must have ≥1 turn
- Errors: LLM failure → keep previous summary, log, retry later via background job (not blocking)

---

## 7. MCP Resources

| Resource URI | Access | Description |
|---|---|---|
| `document://{document_id}` | read-only, RBAC-scoped | full document content + metadata |
| `conversation://{conversation_id}` | read-only, owner/admin only | transcript for context continuation |
| `sources://list` | read-only | all indexed sources, for citation UI & dashboard |

---

## 8. MCP Prompts (versioned, `v1`)

| Prompt | Purpose | Variables | Invocation condition |
|---|---|---|---|
| `system_prompt` | Identity, tone, "DB-first, cite sources, never hallucinate" policy | `current_date, assistant_name` | Always, first message |
| `developer_prompt` | Internal-only output schema & tool-format constraints | `available_tools` | Always, after system prompt |
| `tool_calling_prompt` | Guides tool selection within the pre-filtered candidate set | `candidate_tools, tool_schemas` | `retrieve` / `web_fallback` nodes |
| `rag_prompt` | Combine retrieved chunks + query, require inline citations | `context_chunks, query, max_answer_length` | `synthesize` (DB path) |
| `fallback_prompt` | Be transparent about uncertainty when confidence stays low | `query, best_available_context` | `synthesize` when both DB & web are low-confidence |
| `web_search_prompt` | Summarize web results with mandatory URLs + confidence score | `web_results, query` | `web_fallback` node |
| `safety_prompt` | Prompt-injection defense — treat retrieved/web content as untrusted data, never as instructions | *(static)* | Appended whenever `context_chunks` or `web_results` is non-empty |

---

## 9. What's next

Next turn (per agreed pacing): **MCP Server core** — the actual Python implementation of the MCP server (tool registration, Pydantic schemas, Supabase repository layer, Streamable HTTP transport setup), plus its fakes/adapters for local-only dev. After that: Agent Worker (LangGraph graph + LLM provider fallback chain) → RAG pipeline & embeddings → Voice pipeline (STT/TTS/LiveKit) + frontend → Infra/tests/docs.
