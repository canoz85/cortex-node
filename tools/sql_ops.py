"""Brain-facing generic tools for a read-only SQL datasource."""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from core.models import ToolOutputEnvelope
from core.sql_datasource import SqlDataSource


class SqlQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sql: str = Field(min_length=1)


class SqlSchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def get_sql_tools(datasource: SqlDataSource) -> list:
    @tool("sql_get_schema", args_schema=SqlSchemaRequest)
    def sql_get_schema() -> str:
        """Return the active SQL datasource schema."""
        try:
            return _result(datasource.get_schema(), "Retrieved SQL datasource schema.")
        except Exception as exc:
            return _failure(exc)

    @tool("sql_query", args_schema=SqlQueryRequest)
    def sql_query(sql: str) -> str:
        """Run a read-only SQL query through the active datasource."""
        try:
            return _result(datasource.query(sql), "SQL query completed.")
        except Exception as exc:
            return _failure(exc)

    return [sql_get_schema, sql_query]


def _result(data: dict[str, Any], message: str) -> str:
    return ToolOutputEnvelope(success=True, message=message, data=data).to_tool_output()


def _failure(exc: Exception) -> str:
    return ToolOutputEnvelope(success=False, message=f"SQL datasource request failed: {exc}",
                              error_code="sql_datasource_failed").to_tool_output()
