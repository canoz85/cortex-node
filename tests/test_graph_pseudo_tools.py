from langchain_core.messages import AIMessage

from core.graph_pseudo_tools import recover_pseudo_tool_response


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

    assert len(recovered.tool_calls) == 1
    call = recovered.tool_calls[0]
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

    assert len(recovered.tool_calls) == 1
    call = recovered.tool_calls[0]
    assert call["name"] == "read_file"
    assert call["args"] == {"path": "workspace/dummy.json"}


def test_recover_pseudo_tool_response_relaxed_write_file_recovery_with_unescaped_quotes():
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

    assert len(recovered.tool_calls) == 1
    call = recovered.tool_calls[0]
    assert call["name"] == "write_file"
    assert call["args"]["path"] == "workspace/advanced_sensor.py"
    assert call["args"]["overwrite"] is True
    assert 'print("hello")' in call["args"]["content"]


def test_recover_pseudo_tool_response_relaxed_write_file_recovery_without_closing_fence():
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

    assert len(recovered.tool_calls) == 1
    call = recovered.tool_calls[0]
    assert call["name"] == "write_file"
    assert call["args"]["path"] == "workspace/advanced_sensor.py"
    assert "partial but usable" in call["args"]["content"]
