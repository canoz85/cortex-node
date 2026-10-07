"""Explicit executable inventory. Lookup never grants execution authority."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from core.models import ReadFileRequest


@dataclass(frozen=True)
class ToolEntry:
    name: str
    tool: object


class ToolRegistry:
    def __init__(self, entries: Iterable[ToolEntry]):
        by_name = {}
        for entry in entries:
            if not isinstance(entry.name, str) or not entry.name.strip():
                raise ValueError("Tool name must be a nonempty string")
            if getattr(entry.tool, "name", None) != entry.name:
                raise ValueError(f"Tool name mismatch: {entry.name}")
            if entry.name in by_name:
                raise ValueError(f"Duplicate tool name: {entry.name}")
            by_name[entry.name] = entry.tool
        self.by_name = MappingProxyType(by_name)

    @classmethod
    def from_tools(cls, tools: Iterable[object]):
        return cls(ToolEntry(getattr(tool, "name", None), tool) for tool in tools)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.by_name)

    def resolve(self, names: Iterable[str]) -> list[object]:
        """Resolve an explicit selection in registration order; reject unknown names."""
        selected = frozenset(names)
        unknown = selected - self.names
        if unknown:
            raise ValueError(f"Unregistered tool names: {', '.join(sorted(unknown))}")
        return [tool for name, tool in self.by_name.items() if name in selected]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    family: str
    mutating: bool = False
    enabled_by_default: bool = False
    requires_knowledge: bool = False
    args_schema: type | None = None


# One production manifest, including explicit deployment defaults. A new
# definition is disabled unless its deployment default is deliberately enabled.
TOOL_DEFINITIONS = (
    ToolDefinition("list_files", "workspace", enabled_by_default=True),
    ToolDefinition("read_file", "workspace", enabled_by_default=True, args_schema=ReadFileRequest),
    ToolDefinition("write_file", "workspace", mutating=True, enabled_by_default=True),
    ToolDefinition("make_directory", "workspace", mutating=True, enabled_by_default=True),
    ToolDefinition("read_knowledge_file", "general", enabled_by_default=True, requires_knowledge=True),
    ToolDefinition("run_python", "workspace", mutating=True, enabled_by_default=True),
    ToolDefinition("install_package", "workspace", mutating=True, enabled_by_default=True),
    ToolDefinition("git_status", "workspace", enabled_by_default=True),
    ToolDefinition("git_diff", "workspace", enabled_by_default=True),
    ToolDefinition("git_log", "workspace", enabled_by_default=True),
    ToolDefinition("git_show", "workspace", enabled_by_default=True),
    ToolDefinition("agent_info", "general", enabled_by_default=True),
    ToolDefinition("token_usage", "general", enabled_by_default=True),
    ToolDefinition("current_time", "general", enabled_by_default=True),
    ToolDefinition("rag_search", "general", enabled_by_default=True),
    ToolDefinition("rag_refresh_index", "general", mutating=True, enabled_by_default=True),
    ToolDefinition("query_abap_table", "sap", enabled_by_default=True),
    ToolDefinition("execute_abap_report", "sap", mutating=True, enabled_by_default=True),
    ToolDefinition("lookup_material", "sap", enabled_by_default=True),
    ToolDefinition("get_report_data", "sap", enabled_by_default=True),
    ToolDefinition("scada_status", "general", enabled_by_default=True),
    ToolDefinition("describe_image", "general", enabled_by_default=True),
    ToolDefinition("run_comfy_workflow", "comfy", mutating=True, enabled_by_default=True),
    ToolDefinition("download_comfy_output_image", "comfy", mutating=True, enabled_by_default=True),
    ToolDefinition("get_comfy_history", "comfy", enabled_by_default=True),
    ToolDefinition("find_files", "workspace", enabled_by_default=True),
    ToolDefinition("search_text", "workspace", enabled_by_default=True),
    ToolDefinition("git_changed_files", "workspace", enabled_by_default=True),
)


def get_tool_definition(name: str) -> ToolDefinition | None:
    """Look up production metadata without granting execution authority."""
    return next((definition for definition in TOOL_DEFINITIONS if definition.name == name), None)


def get_tool_argument_schema(name: str, *, registry: ToolRegistry | None = None):
    """Read the registered executable schema, or its production declaration.

    The production declaration supplies the schema when constructing read_file
    and when validating framework-neutral Controller inputs. An explicit
    registry never falls back when an executable or its schema is missing.
    """
    if registry is None:
        definition = get_tool_definition(name)
        return definition.args_schema if definition is not None else None
    executable = registry.by_name.get(name)
    if executable is None:
        return None
    schema = getattr(executable, "args_schema", None)
    if schema is None and callable(getattr(executable, "get_input_schema", None)):
        schema = executable.get_input_schema()
    return schema


def build_tool_registry(workspace_root, knowledge_root, rag_service, model, *, resource_coordinator=None):
    # Imports are construction-time only; the inventory and generic registry do
    # not initialize tools or register anything as an import side effect.
    from tools.comfy_ops import get_comfy_tools
    from tools.discovery_ops import get_discovery_tools
    from tools.exec_ops import get_exec_tools
    from tools.file_ops import get_file_tools
    from tools.git_ops import get_git_tools
    from tools.info_ops import get_info_tools
    from tools.rag_ops import get_rag_tools
    from tools.sap_ops import get_sap_tools
    from tools.scada_ops import get_scada_tools
    from tools.vision_ops import get_vision_tools

    registry = ToolRegistry.from_tools([
        *get_file_tools(workspace_root, knowledge_dir=knowledge_root),
        *get_exec_tools(workspace_root),
        *get_git_tools(workspace_root),
        *get_info_tools(model=model, workspace_dir=workspace_root),
        *get_rag_tools(rag_service),
        *get_sap_tools(workspace_root),
        *get_scada_tools(workspace_root),
        *get_vision_tools(workspace_root),
        *get_comfy_tools(workspace_root, resource_coordinator=resource_coordinator),
        *get_discovery_tools(workspace_root),
    ])
    definitions = {definition.name for definition in TOOL_DEFINITIONS}
    if len(definitions) != len(TOOL_DEFINITIONS):
        raise ValueError("Duplicate production tool definition")
    expected = {definition.name for definition in TOOL_DEFINITIONS
                if knowledge_root or not definition.requires_knowledge}
    if registry.names != expected:
        raise ValueError(f"Production tool inventory mismatch: {sorted(registry.names ^ expected)}")
    return registry


def default_enabled_tools(registry: ToolRegistry) -> list[object]:
    """Deployment projection, independent of any Controller execution ceiling."""
    enabled = {definition.name for definition in TOOL_DEFINITIONS if definition.enabled_by_default}
    return registry.resolve(enabled & registry.names)
