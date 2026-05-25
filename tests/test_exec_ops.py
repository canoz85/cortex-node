import subprocess
from pathlib import Path

from tools.exec_ops import get_exec_tools

from conftest import get_tool, parse_result


def test_run_python_executes_script_and_returns_stdout(tmp_path: Path):
    script = tmp_path / "echo_args.py"
    script.write_text(
        "import sys\n"
        "print('args:' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    tools = get_exec_tools(str(tmp_path))
    run_python = get_tool(tools, "run_python")
    result = parse_result(run_python.invoke({"path": "echo_args.py"}))

    assert result["success"] is True
    assert result["data"]["exit_code"] == 0
    assert "args:" in result["data"]["stdout"]


def test_run_python_accepts_nested_extra_args(tmp_path: Path):
    script = tmp_path / "echo_args.py"
    script.write_text(
        "import sys\n"
        "print('args:' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    tools = get_exec_tools(str(tmp_path))
    run_python = get_tool(tools, "run_python")
    result = parse_result(
        run_python.invoke(
            {
                "path": "echo_args.py",
                "extra_kwargs": {"args": ["alpha", "beta"]},
            }
        )
    )

    assert result["success"] is True
    assert "args:alpha|beta" in result["data"]["stdout"]


def test_run_python_reports_timeout(monkeypatch, tmp_path: Path):
    script = tmp_path / "slow.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    tools = get_exec_tools(str(tmp_path))
    run_python = get_tool(tools, "run_python")
    result = parse_result(run_python.invoke({"path": "slow.py", "timeout_seconds": 1}))

    assert result["success"] is False
    assert "timed out" in result["message"]


def test_run_python_rejects_path_outside_workspace(tmp_path: Path):
    outside_script = tmp_path.parent / "outside_exec_test.py"
    outside_script.write_text("print('secret')\n", encoding="utf-8")

    tools = get_exec_tools(str(tmp_path))
    run_python = get_tool(tools, "run_python")
    result = parse_result(run_python.invoke({"path": "../outside_exec_test.py"}))

    assert result["success"] is False
    assert "outside sandbox workspace" in result["message"]
