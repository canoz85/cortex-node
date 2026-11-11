"""Read-only access to normalized events from the local NexFolioM API."""

import requests
from langchain_core.tools import tool

from core.models import NexFolioMEventsRequest, ToolOutputEnvelope
from core.nexfoliom_config import NEXFOLIOM_API_BASE_URL


NEXFOLIOM_EVENTS_URL = f"{NEXFOLIOM_API_BASE_URL}/api/data/events"


def get_nexfoliom_tools() -> list:
    @tool("get_nexfoliom_events", args_schema=NexFolioMEventsRequest)
    def get_nexfoliom_events(
        accountId: int | None = None,
        from_: str | None = None,
        to: str | None = None,
    ) -> str:
        """Retrieve normalized financial event rows from the local NexFolioM API.

        Event types and values are returned as supplied by the API; this tool does
        not assign income, expense, inflow, or outflow meanings.
        """
        params = {
            key: value for key, value in {
                "accountId": accountId,
                "from": from_,
                "to": to,
            }.items() if value is not None
        }
        try:
            response = requests.get(NEXFOLIOM_EVENTS_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "rows" not in payload:
                raise ValueError("NexFolioM events response must contain a rows collection")
            rows = payload["rows"]
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("NexFolioM events response rows must be a list of event objects")
            return ToolOutputEnvelope(
                success=True,
                message=f"Retrieved {len(rows)} NexFolioM event(s).",
                data=payload,
            ).to_tool_output()
        except (requests.RequestException, ValueError) as exc:
            return ToolOutputEnvelope(
                success=False,
                message=f"NexFolioM events request failed: {exc}",
                error_code="nexfoliom_events_failed",
            ).to_tool_output()

    return [get_nexfoliom_events]
