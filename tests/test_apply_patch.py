"""Exact workspace edits, unchanged failure bytes, and normal authorization."""
import hashlib
import os
import stat
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from conftest import get_tool, parse_result
from core.brain import BrainService
from core.brain_batch_policy import batch_call_limit, validate_read_only_batch
from core.brain_provider import LangChainBrainProvider
from core.error_codes import (FILE_FILE_NOT_FOUND, FILE_PATCH_FAILED, FILE_PATCH_MATCH_COUNT,
                              FILE_PATCH_NOT_TEXT, FILE_PATCH_TARGET_CHANGED)
from core.graph_constants import MUTATING_TOOLS
from core.models import ApplyPatchRequest
from core.planner import PlannerService
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind, ControllerDecisionType
from core.protocol.models import PlanningCapabilities
from core.runtime.controller_transition import ControllerCoordinator
from core.runtime.execution_driver import ExecutionDriver
from core.runtime.tool_result_integration import SerializedToolRuntimePort, integrate_tool_result
from test_brain_outcomes import FakeModel
from test_execution_driver import controller_input, initial_state, FinalizerService
from test_planner_service import FakeProvider, FakeRouter
from tools.file_ops import get_file_tools
from tools.registry import (ToolRegistry, build_tool_registry, default_enabled_tools,
                            get_tool_argument_schema, get_tool_definition, planning_capability_projection)


@pytest.fixture
def patch_tool(tmp_path):
    return get_tool(get_file_tools(str(tmp_path)), "apply_patch")


def patch(patch_tool, **updates):
    return parse_result(patch_tool.invoke({"path": "source.py", "old_text": "old", "new_text": "new", **updates}))


def test_single_replacement_preserves_surroundings_and_returns_bounded_evidence(tmp_path, patch_tool):
    before = b"prefix\nold\nsuffix\n"
    target = tmp_path / "source.py"
    target.write_bytes(before)
    result = patch(patch_tool)
    after = b"prefix\nnew\nsuffix\n"
    assert target.read_bytes() == after
    assert result["success"] is True
    assert result["data"] == {
        "path": "source.py", "replacements": 1, "old_text_chars": 3, "new_text_chars": 3,
        "before_chars": len(before), "after_chars": len(after),
        "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": hashlib.sha256(after).hexdigest(),
    }
    assert result["artifacts"][0]["path"] == "source.py"
    assert result["artifacts"][0]["action"] == "modified"
    assert "content" not in result["data"]
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(("before", "old_text", "new_text", "after"), [
    (b"  old\t\r\nold\n", "  old\t\r\n", " new\t\r\n", b" new\t\r\nold\n"),
    (b"old old", "old", "", b" "),
    (b"a.b axb", "a.b", "c", b"c axb"),
    (b"old", "old", " old ", b" old "),
    (b"old", "old", "old", b"old"),
])
def test_literal_replacement_and_deletion(tmp_path, patch_tool, before, old_text, new_text, after):
    target = tmp_path / "source.py"
    target.write_bytes(before)
    expected = 2 if before == b"old old" else 1
    result = patch(patch_tool, old_text=old_text, new_text=new_text, expected_occurrences=expected)
    assert result["success"] is True
    assert target.read_bytes() == after


def test_explicit_two_occurrences(tmp_path, patch_tool):
    target = tmp_path / "source.py"
    target.write_bytes(b"old\r\nkeep\nold\r\n")
    result = patch(patch_tool, expected_occurrences=2)
    assert result["data"]["replacements"] == 2
    assert target.read_bytes() == b"new\r\nkeep\nnew\r\n"


@pytest.mark.parametrize(("before", "old_text", "expected", "actual"), [
    (b"absent\r\n", "old", 1, 0),
    (b"old old\r\n", "old", 1, 2),
    (b"old\r\n", "old", 2, 1),
    (b" old\t\r\n", "old \n", 1, 0),
    (b"ab", "abc", 1, 0),
])
def test_match_failure_leaves_exact_original_bytes(tmp_path, patch_tool, before, old_text, expected, actual):
    target = tmp_path / "source.py"
    target.write_bytes(before)
    result = patch(patch_tool, old_text=old_text, expected_occurrences=expected)
    assert result["success"] is False
    assert result["error_code"] == FILE_PATCH_MATCH_COUNT
    assert result["message"] == f"Patch match count mismatch: expected {expected}, actual {actual}."
    assert result["error_details"]["expected_occurrences"] == expected
    assert result["error_details"]["actual_occurrences"] == actual
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("invalid", [
    {"old_text": ""}, {"expected_occurrences": 0}, {"expected_occurrences": -1},
    {"expected_occurrences": True}, {"expected_occurrences": 1.0}, {"expected_occurrences": "1"},
    {"new_text": None}, {"path": ""}, {"unexpected": "ignored?"},
])
def test_invalid_schema_rejected_without_edit(tmp_path, patch_tool, invalid):
    target = tmp_path / "source.py"
    target.write_bytes(b"old\r\n")
    with pytest.raises(ValidationError):
        patch(patch_tool, **invalid)
    assert target.read_bytes() == b"old\r\n"


def test_missing_file_is_not_created(tmp_path, patch_tool):
    result = patch(patch_tool, path="missing/target.py")
    assert result["error_code"] == FILE_FILE_NOT_FOUND
    assert list(tmp_path.iterdir()) == []


def test_directory_rejected(tmp_path, patch_tool):
    directory = tmp_path / "source.py"
    directory.mkdir()
    result = patch(patch_tool)
    assert result["error_code"] == FILE_FILE_NOT_FOUND
    assert directory.is_dir() and list(directory.iterdir()) == []


@pytest.mark.parametrize("absolute", [False, True])
def test_outside_workspace_rejected_and_unchanged(tmp_path, absolute):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"old\r\n")
    tool = get_tool(get_file_tools(str(workspace)), "apply_patch")
    result = patch(tool, path=str(outside) if absolute else "../outside.py")
    assert result["success"] is False and result["error_code"] == FILE_PATCH_FAILED
    assert result["error_details"]["exception_type"] == "ValueError"
    assert outside.read_bytes() == b"old\r\n"
    assert list(workspace.iterdir()) == []


def test_outside_symlink_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"old")
    link = workspace / "source.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this system")
    result = patch(get_tool(get_file_tools(str(workspace)), "apply_patch"))
    assert result["error_code"] == FILE_PATCH_FAILED
    assert outside.read_bytes() == b"old"


@pytest.mark.parametrize("before", [b"old\xff", b"old\x00", b"old\x01", b"old\x7f"])
def test_non_text_rejected_without_edit(tmp_path, patch_tool, before):
    target = tmp_path / "source.py"
    target.write_bytes(before)
    result = patch(patch_tool)
    assert result["error_code"] == FILE_PATCH_NOT_TEXT
    assert target.read_bytes() == before


def test_binary_replacement_rejected_without_edit(tmp_path, patch_tool):
    target = tmp_path / "source.py"
    target.write_bytes(b"old")
    assert patch(patch_tool, new_text="\x00")["error_code"] == FILE_PATCH_NOT_TEXT
    assert target.read_bytes() == b"old"


def test_utf8_bom_and_mixed_newlines_preserved(tmp_path, patch_tool):
    target = tmp_path / "source.py"
    before = "\ufeffbaşlık\r\nold\n日本語 😀\r\n".encode("utf-8")
    after = "\ufeffbaşlık\r\nyeni ✅\n日本語 😀\r\n".encode("utf-8")
    target.write_bytes(before)
    result = patch(patch_tool, new_text="yeni ✅")
    assert target.read_bytes() == after
    assert result["data"]["before_chars"] == len(before.decode("utf-8"))
    assert result["data"]["after_chars"] == len(after.decode("utf-8"))


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_write_failure_preserves_original_and_cleans_temp(tmp_path, patch_tool, monkeypatch, operation):
    target = tmp_path / "source.py"
    target.write_bytes(b"old\r\n")
    def fail(*args):
        raise OSError("simulated failure")
    monkeypatch.setattr(f"tools.file_ops.os.{operation}", fail)
    result = patch(patch_tool)
    assert result["error_code"] == FILE_PATCH_FAILED
    assert target.read_bytes() == b"old\r\n"
    assert list(tmp_path.iterdir()) == [target]


def test_observed_concurrent_edit_is_not_overwritten(tmp_path, patch_tool, monkeypatch):
    target = tmp_path / "source.py"
    target.write_bytes(b"old")
    fsync = os.fsync
    def concurrent_edit(fd):
        fsync(fd)
        target.write_bytes(b"external edit")
    monkeypatch.setattr("tools.file_ops.os.fsync", concurrent_edit)
    result = patch(patch_tool)
    assert result["error_code"] == FILE_PATCH_TARGET_CHANGED
    assert target.read_bytes() == b"external edit"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_replacement_cleans_read_only_temporary_file(tmp_path, patch_tool, monkeypatch):
    target = tmp_path / "source.py"
    target.write_bytes(b"old")
    target.chmod(stat.S_IRUSR)
    def fail(*args):
        raise PermissionError("simulated replacement failure")
    monkeypatch.setattr("tools.file_ops.os.replace", fail)
    try:
        assert patch(patch_tool)["error_code"] == FILE_PATCH_FAILED
        assert target.read_bytes() == b"old"
        assert list(tmp_path.iterdir()) == [target]
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_registry_metadata_schema_and_batch_policy(tmp_path):
    registry = build_tool_registry(str(tmp_path), str(tmp_path), SimpleNamespace(), "test")
    executable = registry.by_name["apply_patch"]
    definition = get_tool_definition("apply_patch")
    assert definition.mutating and definition.enabled_by_default and definition.family == "workspace"
    assert definition.max_batch_calls == 1 and "apply_patch" in MUTATING_TOOLS
    assert executable in default_enabled_tools(registry)
    assert get_tool_argument_schema("apply_patch") is executable.args_schema is ApplyPatchRequest
    schema = executable.args_schema.model_json_schema()
    assert schema["required"] == ["path", "old_text", "new_text"]
    assert schema["properties"]["expected_occurrences"]["default"] == 1
    assert schema["properties"]["expected_occurrences"]["minimum"] == 1
    assert schema["properties"]["old_text"]["minLength"] == 1
    card, = planning_capability_projection({"apply_patch"})
    assert card["mutating"] and card["name"] == "apply_patch"
    assert card["purpose"] and card["use_when"] and card["avoid_when"]
    with pytest.raises(ValueError, match="tool_not_batch_eligible"):
        batch_call_limit("apply_patch")
    with pytest.raises(ValueError, match="tool_not_batch_eligible"):
        validate_read_only_batch(
            [("apply_patch", {"path": "source.py", "old_text": "old", "new_text": "new"}),
             ("read_file", {"path": "source.py"})], registry.names)


@pytest.mark.parametrize("route", ["info", "action"])
def test_route_authorization_native_binding_execution_and_provenance(tmp_path, route):
    target = tmp_path / "source.py"
    target.write_bytes(b"old\r\n")
    tools = get_file_tools(str(tmp_path))
    registry = ToolRegistry.from_tools(tools)
    model = FakeModel(AIMessage(content="", tool_calls=[{"name": "apply_patch", "id": "patch-call",
        "args": {"path": "source.py", "old_text": "old", "new_text": "new"}}]))
    # read_file remains the primary hint: supporting mutations use plan authority.
    proposal = {"result": "PLAN_PROPOSED", "objective": "Update source", "message": "", "steps": [
        {"step_id": "inspect", "title": "Inspect and update", "description": "Read and update the known source text",
         "primary_tool": "read_file", "dependencies": []}]}
    planner = PlannerService(provider=FakeProvider(proposal), router=FakeRouter(route), mutating_tools=MUTATING_TOOLS)
    brain = BrainService(provider=LangChainBrainProvider(brain_llm=model, executable_tools=tools), agent_system_prompt="Execute step")
    controller = CortexController(20, planning_capabilities=PlanningCapabilities(available_tools=tuple(sorted(registry.names))))
    driver = ExecutionDriver(coordinator=ControllerCoordinator(controller), planner=planner, brain=brain,
        tool_runtime=SerializedToolRuntimePort(tools), finalizer=FinalizerService())
    state = initial_state()
    planned = driver.turn(state, controller_input(state))
    accepted = driver.turn(planned.execution_state, controller_input(planned.execution_state, planned.worker_result))
    authorized = route == "action"
    assert ("apply_patch" in accepted.execution_state.protocol_visible.active_plan.available_tools) == authorized
    assert ("apply_patch" in {t.name for t in model.bound_tools if hasattr(t, "name")}) == authorized
    if authorized:
        assert accepted.worker_result.outcome == BrainOutcomeKind.TOOL_REQUESTED
        executed = driver.turn(accepted.execution_state, controller_input(accepted.execution_state, accepted.worker_result))
        assert executed.decision.decision_type == ControllerDecisionType.DISPATCH_TOOL_RUNTIME
        assert executed.worker_result.success and target.read_bytes() == b"new\r\n"
        captured = integrate_tool_result(executed.execution_state, executed.decision, executed.worker_result)
        record = captured.working.tool_execution_history[-1]
        assert record.tool_name == "apply_patch" and record.result.data["replacements"] == 1
        assert record.result.data["before_sha256"] == hashlib.sha256(b"old\r\n").hexdigest()
    else:
        assert accepted.worker_result.outcome == BrainOutcomeKind.INVALID_OUTPUT
        assert target.read_bytes() == b"old\r\n"
