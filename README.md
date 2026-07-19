# EchoMCP

EchoMCP is an architecture-first MCP-powered voice agent platform with a RAG core, agent orchestration layer, and API gateway.

## Current implementation status

- A functional RAG pipeline with in-memory vector storage and fake providers
- A FastAPI health endpoint and query endpoint
- Regression tests covering the core pipeline and API health path

## Run locally

```bash
C:/Python314/python.exe -m pytest -q
C:/Python314/python.exe -m uvicorn api.main:app --reload
```
