from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.graph_nodes import (
    _action_required_enforcement_prompt,
    _apply_action_enforcement,
    _apply_failed_rewrite_guard,
    _apply_file_generation_fast_path,
    _apply_read_only_response_guard,
    #_apply_repeated_signature_guard,
    _apply_response_recovery,
    _apply_unchanged_write_guard,
    _apply_workspace_claim_guard,
    _build_pre_messages,
    _args_scope_repair_guidance,
    _discussion_tool_call_correction_prompt,
    _disallowed_read_only_tool_calls,
    _empty_response_fallback,
    _empty_response_retry_prompt,
    _failed_signature_advisory,
    _file_generation_enforcement_prompt,
    _file_generation_gap_rewrite_guidance,
    _file_generation_incomplete_enforcement_prompt,
    _file_generation_initial_guidance,
    _file_generation_still_incomplete_guidance,
    _has_successful_file_events,
    _makes_workspace_analysis_claim,
    _missing_required_args_guidance,
    _missing_dependency_response,
    _pseudo_tool_fallback_response,
    _pseudo_tool_retry_prompt,
    _read_only_analysis_guidance,
    _read_only_guard_correction_prompt,
    _read_only_guard_fallback_response,
    _repeated_signature_correction_prompt,
    _repeated_success_final_answer_prompt,
    _required_tool_enforcement_prompt,
    _required_first_tool_response,
    _stderr_repair_guidance,
    _successful_signature_advisory,
    _unchanged_write_retry_prompt,
    _workspace_claim_guard_response,
)
from core.models import ReadFileResult, ToolResult


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self.responses:
            raise AssertionError("SequenceLLM ran out of responses")
        return self.responses.pop(0)


def test_required_first_tool_response_for_list_files_uses_dot_path():
    response = _required_first_tool_response("list_files")

    assert response.content == "Calling list_files to answer your request."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "list_files"
    assert call["args"] == {"path": "."}
    assert call["type"] == "tool_call"
    assert call["id"].startswith("guard-required-tool-")


def test_required_first_tool_response_for_other_tools_has_empty_args():
    response = _required_first_tool_response("token_usage")

    assert response.content == "Calling token_usage to answer your request."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "token_usage"
    assert call["args"] == {}
    assert call["type"] == "tool_call"
    assert call["id"].startswith("guard-required-tool-")


def test_required_first_tool_response_for_solve_math_passes_question():
    response = _required_first_tool_response("solve_math", "1 meter is 150 cm. what is 10 meter")

    assert response.content == "Calling solve_math to answer your request."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "solve_math"
    assert call["args"] == {"question": "1 meter is 150 cm. what is 10 meter"}
    assert call["type"] == "tool_call"
    assert call["id"].startswith("guard-required-tool-")


def test_disallowed_read_only_tool_calls_filters_mutating_tools():
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "list_files", "args": {"path": "."}, "id": "ok-1", "type": "tool_call"},
            {"name": "run_python", "args": {"path": "a.py"}, "id": "bad-1", "type": "tool_call"},
        ],
    )

    disallowed = _disallowed_read_only_tool_calls(response)

    assert len(disallowed) == 1
    assert disallowed[0]["name"] == "run_python"


def test_read_only_guard_fallback_response_forces_safe_list_files_call():
    response = _read_only_guard_fallback_response()

    assert response.content == "Read-only request guard forced a safe file listing before further analysis."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "list_files"
    assert call["args"] == {"path": "."}
    assert call["type"] == "tool_call"
    assert call["id"].startswith("guard-readonly-")


def test_makes_workspace_analysis_claim_detects_review_language():
    response = AIMessage(content="I reviewed files in the workspace and analyzed the project.")

    assert _makes_workspace_analysis_claim(response) is True


def test_makes_workspace_analysis_claim_false_without_workspace_reference():
    response = AIMessage(content="I analyzed the project structure.")

    assert _makes_workspace_analysis_claim(response) is False


def test_has_successful_file_events_recognizes_read_or_list_success():
    tool_events = [
        {"name": "run_python", "success": True},
        {"name": "read_file", "success": True},
    ]

    assert _has_successful_file_events(tool_events) is True


def test_has_successful_file_events_false_when_only_failures_or_other_tools():
    tool_events = [
        {"name": "read_file", "success": False},
        {"name": "run_python", "success": True},
    ]

    assert _has_successful_file_events(tool_events) is False


def test_workspace_claim_guard_response_forces_safe_list_files_call():
    response = _workspace_claim_guard_response()

    assert response.content == "Listing workspace files first to avoid fabricated file-analysis claims."
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call["name"] == "list_files"
    assert call["args"] == {"path": "."}
    assert call["type"] == "tool_call"
    assert call["id"].startswith("guard-")


def test_missing_dependency_response_includes_install_command():
    response = _missing_dependency_response("pendulum")

    assert "missing dependency: 'pendulum'" in response.content
    assert "pip install pendulum" in response.content


def test_failed_signature_advisory_mentions_signature_and_no_repeat():
    advisory = _failed_signature_advisory('run_python:{"path":"a.py"}')

    assert "The previous tool call failed." in advisory.content
    assert 'Failed signature: run_python:{"path":"a.py"}.' in advisory.content
    assert "Do not repeat" in advisory.content


def test_successful_signature_advisory_mentions_signature_and_next_step():
    advisory = _successful_signature_advisory('list_files:{"path":"."}')

    assert "The previous tool call already succeeded." in advisory.content
    assert 'Successful signature: list_files:{"path":"."}.' in advisory.content
    assert "Choose the next distinct step" in advisory.content


def test_file_generation_initial_guidance_requires_tool_first_execution():
    guidance = _file_generation_initial_guidance()

    assert "concrete file-generation task inside the sandbox workspace" in guidance.content
    assert "start with executable tool calls only" in guidance.content
    assert "main() function" in guidance.content


def test_file_generation_gap_rewrite_guidance_mentions_detected_gap():
    guidance = _file_generation_gap_rewrite_guidance("missing input validation")

    assert "The generated file is not complete for this request yet." in guidance.content
    assert "Detected gap: missing input validation" in guidance.content
    assert "must call write_file now" in guidance.content


def test_file_generation_still_incomplete_guidance_mentions_detected_gap():
    guidance = _file_generation_still_incomplete_guidance("no CLI arg parsing")

    assert "A tool succeeded, but the implementation is still incomplete." in guidance.content
    assert "Detected gap: no CLI arg parsing" in guidance.content
    assert "Do NOT call run_python again before rewriting the file." in guidance.content


def test_file_generation_enforcement_prompt_requires_tool_calls():
    prompt = _file_generation_enforcement_prompt()

    assert "must continue with executable tool calls now" in prompt
    assert "Return tool calls only" in prompt


def test_file_generation_incomplete_enforcement_prompt_mentions_gap():
    prompt = _file_generation_incomplete_enforcement_prompt("missing validation")

    assert "file-generation request is still incomplete" in prompt
    assert "Detected gap: missing validation." in prompt
    assert "Call write_file now" in prompt


def test_required_tool_enforcement_prompt_mentions_kind_and_tool_name():
    prompt = _required_tool_enforcement_prompt("token_usage", "info")

    assert "You ignored the required info tool." in prompt
    assert "Call token_usage now." in prompt


def test_action_required_enforcement_prompt_requires_executable_action():
    prompt = _action_required_enforcement_prompt()

    assert "The user requested concrete actions." in prompt
    assert "Return at least one executable tool call now." in prompt


def test_pseudo_tool_retry_prompt_blocks_pseudo_tool_text():
    prompt = _pseudo_tool_retry_prompt()

    assert "pseudo tool invocation text" in prompt.content
    assert "emit real tool calls only" in prompt.content


def test_pseudo_tool_fallback_response_explains_no_action_taken():
    response = _pseudo_tool_fallback_response()

    assert "no action was taken" in response.content
    assert "task phrased as file changes" in response.content


def test_empty_response_retry_prompt_requests_tool_or_answer():
    prompt = _empty_response_retry_prompt()

    assert "previous response was empty" in prompt.content
    assert "concrete tool calls" in prompt.content
    assert "concise final answer" in prompt.content


def test_empty_response_fallback_mentions_model_availability():
    response = _empty_response_fallback()

    assert "could not produce a valid action or answer" in response.content
    assert "model availability in Ollama" in response.content


def test_discussion_tool_call_correction_prompt_blocks_tools():
    prompt = _discussion_tool_call_correction_prompt()

    assert "discussion-only request" in prompt.content
    assert "Do not call tools" in prompt.content


def test_repeated_signature_correction_prompt_mentions_reason_and_signature():
    prompt = _repeated_signature_correction_prompt('run_python:{"path":"a.py"}', "already failed")

    assert 'Repeated signature: run_python:{"path":"a.py"}.' in prompt.content
    assert "That signature already failed." in prompt.content
    assert "Do not emit that same tool call again." in prompt.content


def test_repeated_success_final_answer_prompt_blocks_tool_calls():
    prompt = _repeated_success_final_answer_prompt('read_file:{"path":"workspace/a.py"}')

    assert 'Repeated signature: read_file:{"path":"workspace/a.py"}.' in prompt.content
    assert "Do not call any tools now." in prompt.content


def test_read_only_analysis_guidance_blocks_mutating_tools():
    prompt = _read_only_analysis_guidance()

    assert "read-only file analysis request" in prompt.content
    assert "Never call write_file, make_directory, or run_python" in prompt.content


def test_missing_required_args_guidance_mentions_v_args():
    prompt = _missing_required_args_guidance()

    assert "required command-line arguments were missing" in prompt.content
    assert "using v__args" in prompt.content


def test_stderr_repair_guidance_truncates_and_labels_error():
    prompt = _stderr_repair_guidance("x" * 1500)

    assert prompt.content.startswith("Latest Python/tool error to fix before re-verification:\n")
    assert len(prompt.content) < 1300


def test_args_scope_repair_guidance_mentions_parse_args_and_main():
    prompt = _args_scope_repair_guidance()

    assert "args scope is broken" in prompt.content
    assert "parse_args() result is defined and used inside main()" in prompt.content


def test_read_only_guard_correction_prompt_blocks_modifications():
    prompt = _read_only_guard_correction_prompt()

    assert "Read-only request guard" in prompt.content
    assert "Do not modify files" in prompt.content


def test_unchanged_write_retry_prompt_blocks_noop_rewrites():
    prompt = _unchanged_write_retry_prompt()

    assert "rewrites the file with identical content" in prompt.content
    assert "Do not emit an unchanged write_file call" in prompt.content


def test_apply_read_only_response_guard_returns_original_response_when_not_read_only():
    response = AIMessage(content="ok")
    llm = DummyLLM(response)

    guarded = _apply_read_only_response_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        read_only_file_request=False,
        tool_name_set={"list_files"},
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_read_only_response_guard_keeps_allowed_tool_calls():
    response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "ok-1", "type": "tool_call"}],
    )
    llm = DummyLLM(response)

    guarded = _apply_read_only_response_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        read_only_file_request=True,
        tool_name_set={"list_files", "read_file"},
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_read_only_response_guard_falls_back_to_list_files_after_repeat_violation():
    initial_response = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "a.py"}, "id": "bad-1", "type": "tool_call"}],
    )
    violating_retry_response = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "a.py", "content": "x"}, "id": "bad-2", "type": "tool_call"}],
    )
    llm = DummyLLM(violating_retry_response)

    guarded = _apply_read_only_response_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        read_only_file_request=True,
        tool_name_set={"list_files", "read_file"},
    )

    assert len(llm.invocations) == 1
    assert guarded.content == "Read-only request guard forced a safe file listing before further analysis."
    assert guarded.tool_calls[0]["name"] == "list_files"


def test_apply_read_only_response_guard_returns_repaired_response_when_safe():
    initial_response = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "a.py"}, "id": "bad-1", "type": "tool_call"}],
    )
    repaired_response = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "ok-2", "type": "tool_call"}],
    )
    llm = DummyLLM(repaired_response)

    guarded = _apply_read_only_response_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        read_only_file_request=True,
        tool_name_set={"list_files", "read_file"},
    )

    assert len(llm.invocations) == 1
    assert guarded is repaired_response


def test_apply_unchanged_write_guard_returns_original_response_when_not_file_generation():
    response = AIMessage(content="ok")
    llm = DummyLLM(response)

    guarded, skip_enforcement = _apply_unchanged_write_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        state={"last_tool_success": True, "last_tool_output": {"path": "workspace/a.py", "content": "same"}},
        file_generation_requested=False,
        tool_name_set={"write_file"},
    )

    assert guarded is response
    assert skip_enforcement is False
    assert llm.invocations == []


def test_apply_unchanged_write_guard_returns_original_response_when_write_changes_content():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "new"},
                "id": "write-1",
                "type": "tool_call",
            }
        ],
    )
    llm = DummyLLM(response)

    guarded, skip_enforcement = _apply_unchanged_write_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        state={"last_tool_success": True, "last_tool_output": {"path": "workspace/a.py", "content": "old"}},
        file_generation_requested=True,
        tool_name_set={"write_file"},
    )

    assert guarded is response
    assert skip_enforcement is False
    assert llm.invocations == []


def test_apply_unchanged_write_guard_retries_until_response_changes():
    initial_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "same"},
                "id": "write-1",
                "type": "tool_call",
            }
        ],
    )
    repaired_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "fixed"},
                "id": "write-2",
                "type": "tool_call",
            }
        ],
    )
    llm = DummyLLM(repaired_response)

    guarded, skip_enforcement = _apply_unchanged_write_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        state={"last_tool_success": True, "last_tool_output": {"path": "workspace/a.py", "content": "same"}},
        file_generation_requested=True,
        tool_name_set={"write_file"},
    )

    assert len(llm.invocations) == 1
    assert guarded is repaired_response
    assert skip_enforcement is False


def test_apply_unchanged_write_guard_stops_after_repeated_identical_writes():
    initial_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "same"},
                "id": "write-1",
                "type": "tool_call",
            }
        ],
    )
    repeated_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "same"},
                "id": "write-2",
                "type": "tool_call",
            }
        ],
    )
    llm = DummyLLM(repeated_response)

    guarded, skip_enforcement = _apply_unchanged_write_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        state={"last_tool_success": True, "last_tool_output": {"path": "workspace/a.py", "content": "same"}},
        file_generation_requested=True,
        tool_name_set={"write_file"},
    )

    assert len(llm.invocations) == 2
    assert "repair attempts kept rewriting identical file content" in guarded.content
    assert skip_enforcement is True


def test_apply_failed_rewrite_guard_returns_original_response_without_matching_failed_context():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "new"},
                "id": "write-1",
                "type": "tool_call",
            }
        ],
    )
    llm = DummyLLM(response)

    guarded = _apply_failed_rewrite_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        history=[],
        response=response,
        state={"last_tool_success": False, "last_tool_signature": "run_python:{\"path\":\"workspace/a.py\"}"},
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_failed_rewrite_guard_requests_corrected_write_when_content_repeats_failed_version():
    failed_write = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "bad"},
                "id": "write-1",
                "type": "tool_call",
            }
        ],
    )
    failed_write_result = ToolMessage(
        content=ToolResult(success=True, message="Wrote file", data={"path": "workspace/a.py"}).to_tool_output(),
        tool_call_id="write-1",
    )
    failed_run = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_python",
                "args": {"path": "workspace/a.py"},
                "id": "run-1",
                "type": "tool_call",
            }
        ],
    )
    failed_run_result = ToolMessage(
        content=ToolResult(
            success=False,
            message="Execution failed",
            data={"stderr": "Traceback: boom"},
        ).to_tool_output(),
        tool_call_id="run-1",
    )
    repeated_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "bad"},
                "id": "write-2",
                "type": "tool_call",
            }
        ],
    )
    repaired_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "workspace/a.py", "content": "fixed"},
                "id": "write-3",
                "type": "tool_call",
            }
        ],
    )
    llm = DummyLLM(repaired_response)

    guarded = _apply_failed_rewrite_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        history=[failed_write, failed_write_result, failed_run, failed_run_result],
        response=repeated_response,
        state={
            "last_tool_success": False,
            "last_tool_signature": 'run_python:{"path":"workspace/a.py"}',
            "last_tool_output": {"data": {"stderr": "Traceback: boom"}},
        },
    )

    assert guarded is repaired_response
    assert len(llm.invocations) == 1
    prompt = llm.invocations[0][-1]
    assert "same file content that already failed verification" in prompt.content
    assert "File: workspace/a.py." in prompt.content
    assert "Failure details: Traceback: boom" in prompt.content


def test_apply_action_enforcement_returns_original_response_when_tool_calls_already_present():
    response = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "ok-1", "type": "tool_call"}],
    )
    llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_action_enforcement(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        tool_name_set={"list_files"},
        file_generation_requested=False,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        preferred_required_tool_name=None,
        preferred_required_tool_kind=None,
        preferred_required_tool_pending=False,
        action_required=True,
        skip_action_enforcement=False,
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_action_enforcement_uses_required_info_tool_prompt_and_finalizes_plain_text():
    initial_response = AIMessage(content="I can answer from memory")
    llm_response = AIMessage(content="Still plain text")
    llm = DummyLLM(llm_response)

    guarded = _apply_action_enforcement(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        tool_name_set={"token_usage"},
        file_generation_requested=False,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        preferred_required_tool_name="token_usage",
        preferred_required_tool_kind="info",
        preferred_required_tool_pending=True,
        action_required=False,
        skip_action_enforcement=False,
    )

    assert len(llm.invocations) == 1
    prompt = llm.invocations[0][-1]
    assert "You ignored the required info tool." in prompt.content
    assert "Call token_usage now." in prompt.content
    assert "returned plain text instead of an executable tool call" in guarded.content


def test_apply_action_enforcement_prefers_incomplete_file_generation_prompt_over_generic_action_prompt():
    initial_response = AIMessage(content="Need more work")
    llm_response = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "workspace/a.py", "content": "fixed"}, "id": "w-1", "type": "tool_call"}],
    )
    llm = DummyLLM(llm_response)

    guarded = _apply_action_enforcement(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        tool_name_set={"write_file"},
        file_generation_requested=True,
        successful_tool_result_in_turn=True,
        current_filegen_issue="missing validation",
        preferred_required_tool_name=None,
        preferred_required_tool_kind=None,
        preferred_required_tool_pending=False,
        action_required=True,
        skip_action_enforcement=False,
    )

    assert guarded is llm_response
    assert len(llm.invocations) == 1
    prompt = llm.invocations[0][-1]
    assert "file-generation request is still incomplete" in prompt.content
    assert "Detected gap: missing validation." in prompt.content


def test_apply_action_enforcement_skips_generic_action_prompt_when_skip_flag_set():
    response = AIMessage(content="plain text")
    llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_action_enforcement(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        tool_name_set={"list_files"},
        file_generation_requested=False,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        preferred_required_tool_name=None,
        preferred_required_tool_kind=None,
        preferred_required_tool_pending=False,
        action_required=True,
        skip_action_enforcement=True,
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_action_enforcement_does_not_reenforce_file_tool_after_success():
    response = AIMessage(content="final listing answer")
    llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_action_enforcement(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        tool_name_set={"list_files"},
        file_generation_requested=False,
        successful_tool_result_in_turn=True,
        current_filegen_issue=None,
        preferred_required_tool_name="list_files",
        preferred_required_tool_kind="file",
        preferred_required_tool_pending=False,
        action_required=True,
        skip_action_enforcement=False,
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_workspace_claim_guard_returns_original_response_when_no_claim():
    response = AIMessage(content="I analyzed the project structure.")

    guarded = _apply_workspace_claim_guard(
        history=[],
        response=response,
        tool_name_set={"list_files"},
    )

    assert guarded is response


def test_apply_workspace_claim_guard_forces_list_files_when_claim_has_no_file_events():
    response = AIMessage(content="I reviewed the files in the workspace and analyzed the project.")

    guarded = _apply_workspace_claim_guard(
        history=[],
        response=response,
        tool_name_set={"list_files"},
    )

    assert guarded.content == "Listing workspace files first to avoid fabricated file-analysis claims."
    assert guarded.tool_calls[0]["name"] == "list_files"


def test_apply_workspace_claim_guard_keeps_response_when_file_events_exist():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "call-1", "type": "tool_call"}],
    )
    tool_message = ToolMessage(
        content=ToolResult(
            success=True,
            message="Listed files",
            data={"entries": ["workspace/a.py"]},
        ).to_tool_output(),
        tool_call_id="call-1",
    )
    response = AIMessage(content="I reviewed the files in the workspace and analyzed the project.")

    guarded = _apply_workspace_claim_guard(
        history=[ai, tool_message],
        response=response,
        tool_name_set={"list_files"},
    )

    assert guarded is response



"""
def test_apply_repeated_signature_guard_returns_original_response_when_signature_not_repeated():
    response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_repeated_signature_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        action_required=True,
        state={"last_tool_signature": 'run_python:{"path": "a.py"}', "last_tool_success": False},
    )

    assert guarded is response
    assert llm.invocations == []


def test_apply_repeated_signature_guard_requests_different_step_for_repeated_signature():
    repeated_response = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    repaired_response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call-2", "type": "tool_call"}],
    )
    llm = DummyLLM(repaired_response)

    guarded = _apply_repeated_signature_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=repeated_response,
        action_required=True,
        state={"last_tool_signature": 'run_python:{"path": "a.py"}', "last_tool_success": False},
    )

    assert guarded is repaired_response
    assert len(llm.invocations) == 1
    prompt = llm.invocations[0][-1]
    assert 'Repeated signature: run_python:{"path": "a.py"}.' in prompt.content
    assert "That signature already failed." in prompt.content


def test_apply_repeated_signature_guard_forces_final_answer_when_success_signature_repeats_again():
    repeated_response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    still_repeated_response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "call-2", "type": "tool_call"}],
    )
    final_answer = AIMessage(content="device_id is DEV_001")
    llm = SequenceLLM([still_repeated_response, final_answer])

    guarded = _apply_repeated_signature_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=repeated_response,
        action_required=True,
        state={"last_tool_signature": 'read_file:{"path": "workspace/a.py"}', "last_tool_success": True},
    )

    assert guarded is final_answer
    assert len(llm.invocations) == 2
    assert "already succeeded" in llm.invocations[0][-1].content
    assert "Do not call any tools now." in llm.invocations[1][-1].content


def test_apply_repeated_signature_guard_returns_fallback_when_success_repeat_persists_after_finalization():
    repeated_response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "call-1", "type": "tool_call"}],
    )
    still_repeated_response = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "call-2", "type": "tool_call"}],
    )
    still_tool_call_after_finalization = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "call-3", "type": "tool_call"}],
    )
    llm = SequenceLLM([still_repeated_response, still_tool_call_after_finalization])

    guarded = _apply_repeated_signature_guard(
        llm=llm,
        pre_messages=[],
        recent_history=[],
        response=repeated_response,
        action_required=True,
        state={"last_tool_signature": 'read_file:{"path": "workspace/a.py"}', "last_tool_success": True},
    )

    assert guarded.tool_calls == []
    assert "will not repeat" in guarded.content
    assert len(llm.invocations) == 2
"""

def test_apply_response_recovery_returns_pseudo_tool_fallback_when_unrecoverable():
    response = AIMessage(content="run_python(path=lambda: 0)")
    response_llm = SequenceLLM([
        AIMessage(content="run_python(path=lambda: 1)"),
        AIMessage(content="run_python(path=lambda: 2)"),
    ])
    planner_llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_response_recovery(
        response_llm=response_llm,
        planner_llm=planner_llm,
        pre_messages=[],
        recent_history=[],
        response=response,
        allow_tool_recovery=True,
        route="action",
        tool_name_set={"run_python"},
    )

    assert "pseudo tool-call text instead of executable tool calls" in guarded.content
    assert len(response_llm.invocations) == 2
    assert planner_llm.invocations == []


def test_apply_response_recovery_returns_empty_response_fallback_after_retry():
    response_llm = SequenceLLM([AIMessage(content="   ")])
    planner_llm = DummyLLM(AIMessage(content="unused"))

    guarded = _apply_response_recovery(
        response_llm=response_llm,
        planner_llm=planner_llm,
        pre_messages=[],
        recent_history=[],
        response=AIMessage(content=""),
        allow_tool_recovery=True,
        route="action",
        tool_name_set={"run_python"},
    )

    assert "could not produce a valid action or answer" in guarded.content
    assert len(response_llm.invocations) == 1
    assert "previous response was empty" in response_llm.invocations[0][-1].content


def test_apply_response_recovery_corrects_tool_calls_for_discussion_routes():
    initial_response = AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "call-1", "type": "tool_call"}],
    )
    response_llm = DummyLLM(AIMessage(content="unused"))
    corrected_response = AIMessage(content="Direct answer only")
    planner_llm = DummyLLM(corrected_response)

    guarded = _apply_response_recovery(
        response_llm=response_llm,
        planner_llm=planner_llm,
        pre_messages=[],
        recent_history=[],
        response=initial_response,
        allow_tool_recovery=False,
        route="coding_discussion",
        tool_name_set={"list_files"},
    )

    assert guarded is corrected_response
    assert response_llm.invocations == []
    assert len(planner_llm.invocations) == 1
    assert "discussion-only request" in planner_llm.invocations[0][-1].content


def test_build_pre_messages_returns_missing_dependency_response_when_detected():
    pre_messages, early_response = _build_pre_messages(
        active_system_prompt="system",
        retrieval_messages=[],
        rolling_summary="",
        route="action",
        preferred_required_tool_name=None,
        preferred_required_tool_pending=False,
        planner_plan_source="planner",
        planner_plan="inspect and fix",
        planner_domain="general",
        planner_domain_enforced=False,
        planner_confidence=0.0,
        file_generation_requested=False,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        read_only_file_request=False,
        action_required=True,
        state={"last_tool_output": {"data": {"stderr": "ModuleNotFoundError: No module named 'pendulum'"}}},
    )

    assert len(pre_messages) >= 1
    assert early_response is not None
    assert "missing dependency: 'pendulum'" in early_response.content


def test_build_pre_messages_adds_action_brief_and_read_only_guidance():
    retrieval_messages = [SystemMessage(content="retrieved context")]

    pre_messages, early_response = _build_pre_messages(
        active_system_prompt="system",
        retrieval_messages=retrieval_messages,
        rolling_summary="summary text",
        route="action",
        preferred_required_tool_name="list_files",
        preferred_required_tool_pending=True,
        planner_plan_source="planner",
        planner_plan="1. inspect files\n2. summarize",
        planner_domain="general",
        planner_domain_enforced=False,
        planner_confidence=0.0,
        file_generation_requested=False,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        read_only_file_request=True,
        action_required=True,
        state={},
    )

    assert early_response is None
    contents = [message.content for message in pre_messages]
    assert contents[0] == "system"
    assert "retrieved context" in contents
    assert any("Planner execution brief below" in content for content in contents)
    assert any("This request must be answered by calling the list_files tool first." in content for content in contents)
    assert any("read-only file analysis request" in content for content in contents)


def test_build_pre_messages_skips_file_tool_first_guidance_when_already_satisfied():
    pre_messages, early_response = _build_pre_messages(
        active_system_prompt="system",
        retrieval_messages=[],
        rolling_summary="",
        route="action",
        preferred_required_tool_name="list_files",
        preferred_required_tool_pending=False,
        planner_plan_source="planner",
        planner_plan="1. list files\n2. answer",
        planner_domain="general",
        planner_domain_enforced=False,
        planner_confidence=0.0,
        file_generation_requested=False,
        successful_tool_result_in_turn=True,
        current_filegen_issue=None,
        read_only_file_request=True,
        action_required=True,
        state={},
    )

    assert early_response is None
    contents = [message.content for message in pre_messages]
    assert not any("must be answered by calling the list_files tool first" in content for content in contents)


def test_build_pre_messages_adds_failure_guidance_for_stderr_and_args_scope():
    pre_messages, early_response = _build_pre_messages(
        active_system_prompt="system",
        retrieval_messages=[],
        rolling_summary="",
        route="action:file_generation",
        preferred_required_tool_name=None,
        preferred_required_tool_pending=False,
        planner_plan_source="planner",
        planner_plan="repair file",
        planner_domain="general",
        planner_domain_enforced=False,
        planner_confidence=0.0,
        file_generation_requested=True,
        successful_tool_result_in_turn=False,
        current_filegen_issue=None,
        read_only_file_request=False,
        action_required=True,
        state={
            "last_tool_success": False,
            "last_tool_signature": 'run_python:{"path": "workspace/a.py"}',
            "last_tool_output": {"data": {"stderr": "NameError: name 'args' is not defined"}},
        },
    )

    assert early_response is None
    contents = [message.content for message in pre_messages]
    assert any("concrete file-generation task inside the sandbox workspace" in content for content in contents)
    assert any("The previous tool call failed." in content for content in contents)
    assert any("Latest Python/tool error to fix before re-verification" in content for content in contents)
    assert any("args scope is broken" in content for content in contents)


def test_apply_file_generation_fast_path_returns_none_when_not_requested():
    response = _apply_file_generation_fast_path(
        history=[],
        state={},
        route="action",
        file_generation_requested=False,
    )

    assert response is None


def test_apply_file_generation_fast_path_stops_after_repeated_verification_failures():
    run_1 = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "workspace/a.py"}, "id": "run-1", "type": "tool_call"}],
    )
    result_1 = ToolMessage(
        content=ToolResult(success=False, message="Execution failed", data={"stderr": "first error"}).to_tool_output(),
        tool_call_id="run-1",
    )
    run_2 = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "workspace/a.py"}, "id": "run-2", "type": "tool_call"}],
    )
    result_2 = ToolMessage(
        content=ToolResult(success=False, message="Execution failed", data={"stderr": "latest error"}).to_tool_output(),
        tool_call_id="run-2",
    )

    response = _apply_file_generation_fast_path(
        history=[run_1, result_1, run_2, result_2],
        state={},
        route="action:file_generation",
        file_generation_requested=True,
    )

    assert response is not None
    assert "repeated verification failures while generating the file" in response.content
    assert "Latest verification error: latest error" in response.content


def test_apply_file_generation_fast_path_returns_args_scope_autofix_call_when_available():
    failed_run = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"path": "workspace/a.py"}, "id": "run-1", "type": "tool_call"}],
    )
    failed_result = ToolMessage(
        content=ToolResult(
            success=False,
            message="Execution failed",
            data={"stderr": "NameError: name 'args' is not defined"},
        ).to_tool_output(),
        tool_call_id="run-1",
    )
    read_call = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "workspace/a.py"}, "id": "read-1", "type": "tool_call"}],
    )
    read_result = ToolMessage(
        content=ReadFileResult(
            success=True,
            message="Read file",
            path="workspace/a.py",
            content="def main():\n    args = parser.parse_args()\ndata = read_and_validate_json(args.file_path)\nif data is not None:\nprint(data)\n",
        ).to_tool_output(),
        tool_call_id="read-1",
    )

    response = _apply_file_generation_fast_path(
        history=[failed_run, failed_result, read_call, read_result],
        state={},
        route="action:file_generation",
        file_generation_requested=True,
    )

    assert response is not None
    assert response.content == "Applying deterministic args-scope repair before re-verification."
    assert response.tool_calls[0]["name"] == "write_file"
    assert response.tool_calls[0]["args"]["path"] == "workspace/a.py"


def test_apply_file_generation_fast_path_returns_repair_read_when_last_run_failed():
    response = _apply_file_generation_fast_path(
        history=[],
        state={
            "last_tool_success": False,
            "last_tool_signature": 'run_python:{"path":"workspace/a.py"}',
            "last_tool_output": {"data": {"stderr": "Traceback: boom"}},
        },
        route="action:file_generation",
        file_generation_requested=True,
    )

    assert response is not None
    assert response.content == "Inspecting the generated Python file before another verification attempt."
    assert response.tool_calls[0]["name"] == "read_file"
    assert response.tool_calls[0]["args"] == {"path": "workspace/a.py"}


def test_apply_file_generation_fast_path_returns_verification_call_after_successful_write():
    write_call = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": "workspace/a.py", "content": "print('ok')"}, "id": "write-1", "type": "tool_call"}],
    )
    write_result = ToolMessage(
        content=ToolResult(success=True, message="Wrote file", data={"path": "workspace/a.py"}).to_tool_output(),
        tool_call_id="write-1",
    )

    response = _apply_file_generation_fast_path(
        history=[write_call, write_result],
        state={},
        route="action:file_generation",
        file_generation_requested=True,
    )

    assert response is not None
    assert response.content == "Proceeding to verify the generated Python file."
    assert response.tool_calls[0]["name"] == "run_python"
    assert response.tool_calls[0]["args"] == {"path": "workspace/a.py"}
    

