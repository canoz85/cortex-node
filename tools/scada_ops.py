from langchain_core.tools import tool

from core.models import ToolResult


def get_scada_tools(_workspace_dir: str):
    @tool
    def scada_status() -> str:
        """Placeholder for future SCADA integrations via MQTT/OPC-UA."""
        return ToolResult(
            success=True,
            message="SCADA tools are not implemented yet.",
            data={"planned_modules": ["MQTT", "OPC-UA", "telemetry polling"]},
        ).to_tool_output()

    return [scada_status]
