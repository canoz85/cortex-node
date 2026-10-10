"""Explicit executable inventory. Lookup never grants execution authority."""

from dataclasses import dataclass
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
    Selection guidance helps Planner choose tools; it does not grant or restrict authority.
    """

    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    limits: tuple[str, ...] = ()
    pagination: str | None = None
    async_kind: Literal["submission", "poll"] | None = None
    use_when: str | None = None
    avoid_when: str | None = None


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
    # Maximum independent synchronous native calls; never grants authorization.
    max_batch_calls: int = 1


# One production manifest, including explicit deployment defaults. A new
# definition is disabled unless its deployment default is deliberately enabled.
TOOL_DEFINITIONS = (
    ToolDefinition("list_files", "workspace", enabled_by_default=True, max_batch_calls=24, planning=CapabilitySemantics(
        "List immediate workspace children or identify a file", (),
        ("path, entries: immediate child names; is_file",), ("Not recursive; no file content or sizes",),
        use_when="Inspect the immediate children of a known directory.",
        avoid_when="Recursive discovery or content-based search is needed.")),
    ToolDefinition("read_file", "workspace", enabled_by_default=True, args_schema=ReadFileRequest, max_batch_calls=24,
        planning=CapabilitySemantics(
            "Read workspace text", ("path: required workspace text file",),
            ("content; total_chars even on partial reads",),
            ("UTF-8 only; total_chars counts decoded characters, not byte size",),
            pagination="offset/limit in characters; continue truncated content",
            use_when="Read a known workspace text file, continuing partial reads when needed.",
            avoid_when="The path still needs discovery or the target is not a text file.")),
    ToolDefinition("write_file", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Write workspace text", ("path and content: required",),
        ("path, characters_written",),
        ("Creates parents; replaces existing content by default", "Write receipt does not independently verify content"))),
    ToolDefinition("make_directory", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Create a workspace directory", ("path: required workspace directory",),
        ("path, creation receipt",), ("Creates parents; existing directories allowed",))),
    ToolDefinition("read_knowledge_file", "general", enabled_by_default=True, requires_knowledge=True,
        planning=CapabilitySemantics("Read a knowledge-folder text document", ("path: required knowledge-relative file",),
            ("content",), ("Knowledge folder only; not live workspace evidence",),
            use_when="Read a known knowledge-relative text document directly.",
            avoid_when="The target belongs to the workspace or still requires topic-based discovery.")),
    ToolDefinition("run_python", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Run an existing workspace Python script", ("path: required .py script",),
        ("exit_code, stdout, stderr",), ("Script-defined side effects; exit success does not prove task success",))),
    ToolDefinition("install_package", "workspace", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Install into the active Python environment", ("package_name: required pip specifier",),
        ("package, bounded pip stdout",), ("Changes interpreter environment; needs package access",))),
    ToolDefinition("git_status", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect repository status", (), ("short status and branch",),
        ("Requires a Git repository; no file content",),
        use_when="Obtain a concise current repository and branch overview.",
        avoid_when="Structured changed-file records or patch content are needed.")),
    ToolDefinition("git_diff", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Read unstaged textual diff", (),
        ("unstaged patch text",), ("No staged or untracked file content",),
        use_when="Inspect current unstaged textual changes, optionally for one path.",
        avoid_when="Staged changes, untracked content, or committed patch content are required.")),
    ToolDefinition("git_log", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect recent commit history", (),
        ("commit hash, date, author, subject",), ("1..50 commits; no file content",),
        use_when="Discover recent commits and their basic metadata.",
        avoid_when="Current workspace state or one revision's file statistics are required.")),
    ToolDefinition("git_show", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect revision summary (default HEAD)", (),
        ("revision statistics and summary",), ("--stat only; no patch or full file content",),
        use_when="Inspect a revision's summary and file-change statistics.",
        avoid_when="Patch text, full historical file content, or current workspace content are required.")),
    ToolDefinition("agent_info", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect runtime configuration", (), ("model, workspace, context_window, max_steps, token_usage",),
        ("Configuration/last known usage, not workspace content",))),
    ToolDefinition("token_usage", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Inspect recorded token usage", (), ("prompt/completion/total token counts",),
        ("May be unavailable before any response",))),
    ToolDefinition("current_time", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Read the local system clock", (), ("iso, formatted",),
        ("Local system time; no timezone conversion",))),
    ToolDefinition("rag_search", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Search indexed knowledge", ("query: required",),
        ("ranked chunks and sources",), ("Can be stale; not live workspace discovery",),
        use_when="Retrieve relevant knowledge chunks by topic or semantic similarity.",
        avoid_when="Current workspace facts, exhaustive matching, or a complete known document are required.")),
    ToolDefinition("rag_refresh_index", "general", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Rebuild the in-memory knowledge index", (), ("chunks_indexed",),
        ("Changes index, not source documents",),
        use_when="Refresh the knowledge index after source changes or when index freshness is explicitly required.",
        avoid_when="Ordinary retrieval is sufficient and no freshness problem is established.")),
    ToolDefinition("query_abap_table", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Query SAP table stub", ("table_name: required",),
        ("mock record collection",),
        ("Placeholder: no live SAP connection or actual filtering",))),
    ToolDefinition("execute_abap_report", "sap", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Invoke ABAP report stub", ("report_name: required",),
        ("mock report output",), ("Placeholder: no actual execution or live SAP effects",))),
    ToolDefinition("lookup_material", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Look up SAP material stub", ("material_id: required",),
        ("mock material/plant data",), ("Placeholder: cannot establish live material facts",))),
    ToolDefinition("get_report_data", "sap", enabled_by_default=True, planning=CapabilitySemantics(
        "Read report-export stub", ("report_name: required",),
        ("mock record collection",),
        ("Placeholder: no live report or exported file; csv/xlsx labels do not create files",))),
    ToolDefinition("scada_status", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Report SCADA integration availability", (), ("planned_modules",),
        ("Placeholder: no live telemetry or device status",))),
    ToolDefinition("describe_image", "general", enabled_by_default=True, planning=CapabilitySemantics(
        "Describe a workspace image", ("image_path: required workspace image file",),
        ("model-generated description",), ("Requires local llava; not deterministic measurement",),
        use_when="Visually inspect an existing workspace image.",
        avoid_when="Image generation, job status, text reading, or deterministic measurement is required.")),
    ToolDefinition("run_comfy_workflow", "comfy", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Submit image generation with fixed workflow", ("positive_prompt, seed, steps, cfg, width, height: required",),
        ("prompt_id, job status, submission receipt",),
        ("Needs fixed template/installed checkpoint", "Submission is not completion; poll history for outputs"), async_kind="submission",
        use_when="Submit a new image-generation job.",
        avoid_when="Only an existing job's status or output is needed.")),
    ToolDefinition("download_comfy_output_image", "comfy", mutating=True, enabled_by_default=True, planning=CapabilitySemantics(
        "Download generated image into workspace", ("filename: required server image name",),
        ("workspace save receipt",), ("Needs output identity from generation history",),
        use_when="Save a known Comfy output image into the workspace.",
        avoid_when="The output identity is unknown or generation/status inspection is still needed.")),
    ToolDefinition("get_comfy_history", "comfy", enabled_by_default=True, planning=CapabilitySemantics(
        "Observe submitted generation and discover outputs", ("prompt_id: required submitted job identity",),
        ("job status/completed, filenames collection, outputs",),
        ("May be queued/running; absent history is not success", "Does not download outputs"), async_kind="poll",
        use_when="Inspect a submitted Comfy job using a known prompt ID and discover its outputs.",
        avoid_when="No job identity exists or the output image itself must be downloaded.")),
    ToolDefinition("find_files", "workspace", enabled_by_default=True, max_batch_calls=24, planning=CapabilitySemantics(
        "Discover workspace paths by basename glob, optionally recursive", (),
        ("files: sorted workspace-relative path collection",),
        ("No file content or sizes", "Case-sensitive; excludes file/directory links", "limit must be 1..100"),
        pagination="offset/limit over sorted paths",
        use_when="Discover workspace files by basename glob, optionally recursively.",
        avoid_when="Selection depends on file content or Git change status.")),
    ToolDefinition("search_text", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Search workspace text and discover matching paths, optionally recursive", ("query: required literal text or regex",),
        ("matches: path, line_number, text snippet",),
        ("Case-sensitive UTF-8; invalid bytes replaced", "Snippets can truncate; not full-file content", "limit must be 1..100"),
        pagination="offset/limit over matches ordered by path then line",
        use_when="Locate implementation, symbols, literals, or regex matches inside workspace text.",
        avoid_when="The target path is already known and only complete file content is needed.")),
    ToolDefinition("git_changed_files", "workspace", enabled_by_default=True, planning=CapabilitySemantics(
        "Discover changed repository paths", (),
        ("changed_files: path/status collection with staged and worktree flags; branch/upstream",),
        ("No file content; requires repository; ignored files omitted", "limit must be 1..100"), pagination="offset/limit over sorted changed paths",
        use_when="Enumerate changed paths with staged, worktree, and untracked status.",
        avoid_when="File content, patch text, or historical changes are required.")),
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
        semantics = definition.planning
        card = {"name": definition.name, "purpose": semantics.purpose, "outputs": semantics.outputs}
        if semantics.limits:
            card["limits"] = semantics.limits
        if semantics.inputs:
            card["inputs"] = semantics.inputs
        card["mutating"] = definition.mutating
        if semantics.pagination:
            card["pagination"] = True
        if semantics.async_kind:
            card["async_kind"] = semantics.async_kind
        if semantics.use_when is not None:
            card["use_when"] = semantics.use_when
        if semantics.avoid_when is not None:
            card["avoid_when"] = semantics.avoid_when
        summaries.append(card)
    return tuple(summaries)


def get_tool_argument_schema(name: str, *, registry: ToolRegistry | None = None):
    """Read the registered executable schema, or its production declaration.

    Production declarations support construction and standalone validation.
    Composed validation uses registered executable schemas, including inferred
    schemas. An explicit registry never falls back when an executable or its
    schema is missing.
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
