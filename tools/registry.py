"""Explicit executable inventory. Lookup never grants execution authority."""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Iterable, Literal

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
class CapabilitySemantics:
    """Explicit planning facts, not an argument schema or execution authority.

    Inputs describe essential needs; executable schemas still validate arguments.
    Outputs describe evidence on success, never a guarantee that a call succeeds.
    Collection shape belongs in outputs; exact coverage remains Controller policy.
    """

    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    limits: tuple[str, ...] = ()
    pagination: str | None = None
    async_kind: Literal["submission", "poll"] | None = None


class CapabilityMetadataError(ValueError):
    """An authorized name lacks explicit planning semantics."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    family: str
    mutating: bool = False
    enabled_by_default: bool = False
    requires_knowledge: bool = False
    args_schema: type | None = None
    planning: CapabilitySemantics | None = None


# One production manifest, including explicit deployment defaults. A new
# definition is disabled unless its deployment default is deliberately enabled.
TOOL_DEFINITIONS = (
    ToolDefinition("list_files", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "List a workspace directory or identify a file", ("path: optional workspace path, defaults to .",),
        ("path, entries: immediate child names; is_file",), ("Not recursive; no file content or sizes",))),
    ToolDefinition("read_file", "workspace", enabled_by_default=True, args_schema=ReadFileRequest,
        planning=CapabilitySemantics(
            "Read workspace text content", ("path: required workspace text file", "offset/limit: optional character range"),
            ("path, content, total_chars, offset, read_chars, is_truncated",),
            ("UTF-8 text only", "total_chars counts decoded characters, not byte size"),
            pagination="offset/limit in characters; continue truncated content")),
    ToolDefinition("write_file", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Write workspace text content", ("path and content: required", "overwrite: optional, defaults to true"),
        ("path, characters_written, created/modified artifact",),
        ("Creates parent directories; replaces existing content when overwrite is true", "Write receipt does not independently verify content"))),
    ToolDefinition("make_directory", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Create a workspace directory", ("path: required workspace directory",),
        ("path, creation receipt",), ("Creates missing parents; existing directories are allowed",))),
    ToolDefinition("read_knowledge_file", "general", enabled_by_default=True, requires_knowledge=True,
        planning=CapabilitySemantics("Read a knowledge-folder text document", ("path: required knowledge-relative file",),
            ("path, content",), ("Knowledge folder only; not live workspace evidence",))),
    ToolDefinition("run_python", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Run an existing workspace Python script", ("path: required .py script", "args and timeout_seconds: optional"),
        ("exit_code, stdout, stderr",), ("May have script-defined side effects", "Process success does not prove the requested semantic outcome"))),
    ToolDefinition("install_package", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Install a package into the active Python environment", ("package_name: required pip package specifier", "timeout_seconds: optional"),
        ("package, command, bounded pip stdout on success",), ("Changes the active interpreter environment; needs package access",))),
    ToolDefinition("git_status", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect workspace repository status", (), ("exit_code, stdout/stderr: short status and branch",),
        ("Requires a Git repository; no file content",))),
    ToolDefinition("git_diff", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect unstaged repository changes", ("path: optional file restriction",),
        ("exit_code, stdout/stderr: unstaged textual diff",), ("Does not include staged or untracked file content",))),
    ToolDefinition("git_log", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect recent commit history", ("limit: optional commit count",),
        ("exit_code, stdout/stderr: commit hash, date, author, subject",), ("Commit count clamped to 1..50; no file content",))),
    ToolDefinition("git_show", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect a Git revision summary", ("revision: optional, defaults to HEAD",),
        ("exit_code, stdout/stderr: revision statistics and summary",), ("Uses --stat, not full file content or patch evidence",))),
    ToolDefinition("agent_info", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect CortexNode runtime configuration", (), ("model, workspace, context_window, max_steps, token_usage",),
        ("Configuration and last known usage, not workspace content",))),
    ToolDefinition("token_usage", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect recorded token usage", (), ("Recorded prompt/completion/total token counts",),
        ("Only recorded runtime usage; may be unavailable before any response",))),
    ToolDefinition("current_time", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Read the local system clock", ("format: optional strftime format",), ("iso, formatted, format",),
        ("Local system time; no timezone conversion",))),
    ToolDefinition("rag_search", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Search indexed knowledge", ("query: required", "top_k: optional result count"),
        ("Ranked knowledge chunks and source context",), ("Retrieved context can be stale; not live workspace discovery",))),
    ToolDefinition("rag_refresh_index", "general", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Rebuild the in-memory knowledge index", (), ("chunks_indexed",),
        ("Changes the knowledge index; does not modify source documents",))),
    ToolDefinition("query_abap_table", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Query the current SAP table stub", ("table_name: required", "fields, where_clause, max_rows: optional"),
        ("table_name, fields, row_count, data: mock record collection",),
        ("Placeholder: no live SAP connection; filters are not executed against SAP",))),
    ToolDefinition("execute_abap_report", "sap", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Invoke the current ABAP report stub", ("report_name: required", "parameters: optional mapping"),
        ("report_name, parameters_used, mock output, row_count",), ("Placeholder: no actual report execution or live SAP effects",))),
    ToolDefinition("lookup_material", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Look up material data through the current SAP stubs", ("material_id: required", "include_plant_data: optional"),
        ("material_id, material_data, optional plant_data collection",), ("Placeholder: cannot establish live material facts",))),
    ToolDefinition("get_report_data", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Obtain the current report-export stub data", ("report_name: required", "output_format and parameters: optional"),
        ("report_name, output_format, row_count, data: mock record collection",),
        ("Placeholder: no live report or exported file; csv/xlsx labels do not create files",))),
    ToolDefinition("scada_status", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Report SCADA integration availability", (), ("planned_modules",),
        ("Placeholder: no live telemetry or device status",))),
    ToolDefinition("describe_image", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Describe a workspace image with the local vision model", ("image_path: required workspace image file",),
        ("path, model-generated description",), ("Requires local llava; interpretation is not deterministic image measurement",))),
    ToolDefinition("run_comfy_workflow", "comfy", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Submit image generation using the fixed workflow", ("positive_prompt, seed, steps, cfg, width, height: required", "negative_prompt, filename_prefix, client_id, prompt_id: optional"),
        ("prompt_id, async job identity/status, submission receipt",),
        ("Fixed template and installed checkpoint required", "Submission is not completion; poll history for outputs"), async_kind="submission")),
    ToolDefinition("download_comfy_output_image", "comfy", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Download a generated image into the workspace", ("filename: required server image name", "subfolder, folder_type, save_path: optional"),
        ("message: workspace save receipt",), ("Needs an output identity discovered from generation history",))),
    ToolDefinition("get_comfy_history", "comfy", enabled_by_default=True, planning=CapabilitySemantics(
        "Observe a submitted generation and discover its outputs", ("prompt_id: required submitted job identity",),
        ("async job status/terminality, completed, filenames collection, outputs, primary_filename",),
        ("May still be queued/running; history absence is not proof of success", "Does not download outputs"), async_kind="poll")),
    ToolDefinition("find_files", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Discover workspace file paths", ("path/pattern/recursive: optional scope, basename glob and recursion", "offset/limit: optional page"),
        ("files: sorted workspace-relative path collection", "count, offset, limit, has_more, truncation"),
        ("Does not establish file content or byte size", "Case-sensitive basename glob; excludes file links and does not traverse directory links", "limit must be 1..100"),
        pagination="offset/limit over sorted paths")),
    ToolDefinition("search_text", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Discover matching workspace text lines", ("query: required literal text or regex", "path, file_pattern, regex, recursive, offset, limit: optional"),
        ("matches: path, line_number, text, text_start_column, text_truncated", "count, offset, limit, has_more, truncation"),
        ("Case-sensitive UTF-8 search; invalid bytes replaced", "Matching snippets can truncate; not full-file content", "limit must be 1..100"),
        pagination="offset/limit over matches ordered by path then line")),
    ToolDefinition("git_changed_files", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Discover changed repository paths", ("offset/limit: optional page",),
        ("changed_files: path/status collection with staged and worktree flags", "branch, upstream, count and continuation metadata"),
        ("No file content; requires a repository; ignored files omitted", "limit must be 1..100"), pagination="offset/limit over sorted changed paths")),
)


def get_tool_definition(name: str) -> ToolDefinition | None:
    """Look up production metadata without granting execution authority."""
    return next((definition for definition in TOOL_DEFINITIONS if definition.name == name), None)


def planning_capability_projection(names: Iterable[str]) -> tuple[dict, ...]:
    """Project explicit definition facts for an already-authorized selection only."""
    summaries = []
    for name in sorted(set(names)):
        definition = get_tool_definition(name)
        if definition is None or definition.planning is None:
            raise CapabilityMetadataError(f"Planning semantics unavailable for authorized capability '{name}'")
        semantics = asdict(definition.planning)
        summaries.append({"name": definition.name, "mutating": definition.mutating,
                          **{key: value for key, value in semantics.items() if value is not None}})
    return tuple(summaries)


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
