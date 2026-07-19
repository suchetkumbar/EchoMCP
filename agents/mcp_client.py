from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class MCPClient:
    def __init__(self, base_url: Optional[str] = "http://localhost:8000", test_client: Optional[Any] = None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.test_client = test_client

    def semantic_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        payload = {"query": query, "top_k": top_k}
        if self.test_client is not None:
            resp = self.test_client.post("/mcp/semantic_search", json=payload)
            resp.raise_for_status()
            return resp.json()
        if not self.base_url:
            raise RuntimeError("No base_url configured for MCPClient")
        url = f"{self.base_url}/mcp/semantic_search"
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
