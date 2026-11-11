"""Generic contract for read-only SQL-backed data sources."""

from typing import Any, Protocol


class SqlDataSource(Protocol):
    def get_schema(self) -> dict[str, Any]:
        """Return a structured description of the available schema."""

    def query(self, sql: str) -> dict[str, Any]:
        """Execute a read-only query and return structured rows and metadata."""

