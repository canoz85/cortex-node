# CortexNode

Local-first AI software agent built with LangGraph + Ollama.

CortexNode runs a bounded reasoning loop that can call sandboxed tools for file operations, Python execution, git inspection, runtime info, and future SCADA integration.

## Features

- Local LLM orchestration with `ChatOllama`
- Tool-calling workflow with `LangGraph` (`brain -> tools -> capture_tool_output -> brain`)
- Sandboxed file tools: `list_files`, `read_file`, `write_file`, `make_directory`
- Sandboxed Python execution tool: `run_python`
- Git read tools: `git_status`, `git_diff`, `git_log`, `git_show`
- Runtime info tools: `agent_info`, `token_usage`, `current_time`
- SCADA stub tool for planned MQTT/OPC-UA integrations
- Interactive CLI mode and one-shot prompt mode

## Requirements

- Python 3.10+
- Ollama installed and running locally
- An available Ollama model (default: `qwen2.5:7b`)

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Ensure Ollama model is available

```bash
ollama pull qwen2.5:7b
```

## Run

### Interactive mode

```bash
python main.py
```

### Single prompt mode

```bash
python main.py --prompt "Create hello.py in the workspace and run it"
```

### Custom model and workspace

```bash
python main.py --model qwen2.5:7b --workspace workspace
```

## CLI Arguments

- `--workspace`: Sandbox directory used by tools (default: `workspace`)
- `--model`: Ollama model name (default: `qwen2.5:7b`)
- `--prompt`: Run one prompt and exit (if omitted, interactive mode starts)

## How It Works

1. `main.py` parses CLI args and builds the LangGraph app.
2. `core/graph.py` assembles tool sets and binds them to `ChatOllama`.
3. The `brain` node generates the next assistant step.
4. If tool calls are requested, execution routes through `ToolNode`.
5. Tool outputs are normalized and fed back into state.
6. Loop exits when no tool call remains or step limit is reached (max 12 steps per prompt).

## Tool Output Format

Tools return a human-readable summary plus structured JSON payload marker:

```text
<summary line>
<tool_result_json>
{ ...json payload... }
```

This format allows both readable terminal output and reliable parsing in the agent loop.

## Project Structure

```text
cortex-node/
|-- .gitignore
|-- README.md
|-- main.py
|-- requirements.txt
|-- core/
|   |-- __init__.py
|   |-- graph.py
|   |-- models.py
|   |-- state.py
|   `-- tool_output.py
|-- scripts/
|   `-- hello.py
|-- tools/
|   |-- __init__.py
|   |-- exec_ops.py
|   |-- file_ops.py
|   |-- git_ops.py
|   |-- info_ops.py
|   `-- scada_ops.py
`-- workspace/
    `-- version2.md
```

## Notes

- File and execution tools enforce sandbox boundaries relative to the selected workspace.
- Git tools execute in the selected workspace directory and return stdout/stderr/exit code.
- `current_time` is the preferred path for time/date questions to avoid guessed values.
- SCADA integrations are currently placeholders and not yet connected to real PLC/telemetry endpoints.
