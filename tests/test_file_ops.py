from pathlib import Path

from tools.file_ops import get_file_tools

from conftest import get_tool, parse_result


def test_write_and_read_file_round_trip(tmp_path: Path):
    tools = get_file_tools(str(tmp_path))
    write_file = get_tool(tools, "write_file")
    read_file = get_tool(tools, "read_file")

    write_result = parse_result(write_file.invoke({"path": "hello.txt", "content": "hello"}))
    assert write_result["success"] is True

    read_result = parse_result(read_file.invoke({"path": "hello.txt"}))
    assert read_result["success"] is True
    assert read_result["content"] == "hello"


def test_list_files_includes_directories_and_files(tmp_path: Path):
    tools = get_file_tools(str(tmp_path))
    make_directory = get_tool(tools, "make_directory")
    write_file = get_tool(tools, "write_file")
    list_files = get_tool(tools, "list_files")

    parse_result(make_directory.invoke({"path": "docs"}))
    parse_result(write_file.invoke({"path": "readme.md", "content": "x"}))

    listed = parse_result(list_files.invoke({"path": "."}))
    assert listed["success"] is True
    entries = listed["entries"]
    assert "docs/" in entries
    assert "readme.md" in entries


def test_read_file_rejects_path_outside_workspace(tmp_path: Path):
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    tools = get_file_tools(str(tmp_path))
    read_file = get_tool(tools, "read_file")

    result = parse_result(read_file.invoke({"path": "../outside.txt"}))
    assert result["success"] is False
    assert "outside sandbox workspace" in result["message"]


def test_read_knowledge_file_is_available_when_knowledge_dir_is_set(tmp_path: Path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir(parents=True, exist_ok=True)
    (knowledge / "rules.md").write_text("rule text", encoding="utf-8")

    tools = get_file_tools(str(tmp_path), knowledge_dir=str(knowledge))
    read_knowledge_file = get_tool(tools, "read_knowledge_file")

    result = parse_result(read_knowledge_file.invoke({"path": "rules.md"}))
    assert result["success"] is True
    assert result["content"] == "rule text"
