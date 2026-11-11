"""HTTP adapter for the local NexFolioM data API."""

from typing import Any

import requests

from core.nexfoliom_config import NEXFOLIOM_API_BASE_URL


class NexFolioMHttpDataSource:
    def __init__(self, base_url: str = NEXFOLIOM_API_BASE_URL, *, timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_schema(self) -> dict[str, Any]:
        return self._request("GET", "/api/data/schema")

    def query(self, sql: str) -> dict[str, Any]:
        if not isinstance(sql, str):
            raise ValueError("sql must be a string")
        result = self._request("POST", "/api/data/query", json={"sql": sql})
        if not isinstance(result.get("rows"), list) or any(not isinstance(row, dict) for row in result["rows"]):
            raise ValueError("NexFolioM query response rows must be a list of objects")
        if "truncated" in result and not isinstance(result["truncated"], bool):
            raise ValueError("NexFolioM query response truncated must be a boolean")
        return result

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = requests.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"NexFolioM {path} request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"NexFolioM {path} response must be an object")
        if payload.get("success") is False or "error" in payload:
            raise RuntimeError(f"NexFolioM API error: {payload.get('error', 'request failed')}")
        if path.endswith("/schema"):
            schema = payload.get("schema", payload)
            if not isinstance(schema, dict):
                raise ValueError("NexFolioM schema response must contain an object")
            return schema
        if "rows" not in payload:
            raise ValueError("NexFolioM query response must contain rows")
        return payload
