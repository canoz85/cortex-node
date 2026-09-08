from langchain_core.messages import AIMessage

from core.brain_normalization import normalize_brain_output
from core.protocol.enums import BrainOutcomeKind
from core.protocol.models import BrainInput, ExecutionIdentity, ExecutionCursor, ExecutionContext, ExecutionPlan, ExecutionStep


def recover_pseudo_tool_response(message, tools):
    step = ExecutionStep(step_id="s1", title="Write the file")
    return normalize_brain_output(message, BrainInput(
        identity=ExecutionIdentity(execution_id="test", protocol_version="1.0"),
        cursor=ExecutionCursor(step_id="s1"),
        context=ExecutionContext(user_request="write file"),
        active_plan=ExecutionPlan(plan_id="p1", steps=(step,)), active_step=step,
    ), tools, allow_text_tool_calls=True)


def test_recover_pseudo_tool_response_recovers_fenced_json_with_nested_arguments_and_multiline_content():
    message = AIMessage(
        content=(
            "```json\n"
            "{\n"
            "  \"name\": \"write_file\",\n"
            "  \"arguments\": {\n"
            "    \"path\": \"workspace/advanced_sensor.py\",\n"
            "    \"content\": \"#!/usr/bin/env python\\nimport argparse\\n\\ndef validate_data(data):\\n    if data.get('meta', {}).get('ok') is not True:\\n        return False\\n    return True\\n\"\n"
            "  }\n"
            "}\n"
            "```"
        )
    )

    recovered = recover_pseudo_tool_response(message, {"write_file", "read_file", "list_files"})

    assert recovered.kind == BrainOutcomeKind.TOOL_REQUESTED
    call = {"name": recovered.tool_request.tool_name, "args": recovered.tool_request.arguments}
    assert call["name"] == "write_file"
    assert call["args"]["path"] == "workspace/advanced_sensor.py"
    assert "validate_data" in call["args"]["content"]


def test_recover_pseudo_tool_response_recovers_tool_calls_wrapper_shape():
    message = AIMessage(
        content=(
            "```json\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\n"
            "      \"name\": \"read_file\",\n"
            "      \"arguments\": {\n"
            "        \"path\": \"workspace/dummy.json\"\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )
    )

    recovered = recover_pseudo_tool_response(message, {"write_file", "read_file", "list_files"})

    assert recovered.kind == BrainOutcomeKind.TOOL_REQUESTED
    call = {"name": recovered.tool_request.tool_name, "args": recovered.tool_request.arguments}
    assert call["name"] == "read_file"
    assert call["args"] == {"path": "workspace/dummy.json"}


def test_normalizer_rejects_unescaped_quotes_without_repairing_file_content():
    message = AIMessage(
        content=(
            "```json\n"
            "{\n"
            "  \"name\": \"write_file\",\n"
            "  \"arguments\": {\n"
            "    \"path\": \"workspace/advanced_sensor.py\",\n"
            "    \"overwrite\": true,\n"
            "    \"content\": \"print(\"hello\")\\nprint('ok')\"\n"
            "  }\n"
            "}\n"
            "```"
        )
    )

    recovered = recover_pseudo_tool_response(message, {"write_file", "read_file", "list_files"})

    assert recovered.kind == BrainOutcomeKind.INVALID_OUTPUT
    assert recovered.tool_request is None


def test_normalizer_rejects_unclosed_fence_without_salvaging_a_partial_output():
    message = AIMessage(
        content=(
            "```json\n"
            "{\n"
            "  \"name\": \"write_file\",\n"
            "  \"arguments\": {\n"
            "    \"path\": \"workspace/advanced_sensor.py\",\n"
            "    \"content\": \"#!/usr/bin/env python\\ndef main():\\n    print('partial but usable')\"\n"
            "  }\n"
            "}\n"
        )
    )

    recovered = recover_pseudo_tool_response(message, {"write_file", "read_file", "list_files"})

    assert recovered.kind == BrainOutcomeKind.INVALID_OUTPUT
    assert recovered.tool_request is None
