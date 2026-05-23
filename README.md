# CortexNode

Local-first AI software agent built with LangGraph + Ollama.

CortexNode runs a bounded reasoning loop that can call sandboxed tools for file operations, Python execution, git inspection, runtime info, and future SCADA integration.

## Features

- Local LLM orchestration with `ChatOllama`
- Tool-calling workflow with `LangGraph` (`planner -> brain -> tools -> capture_tool_output -> brain`)
- Sandboxed file tools: `list_files`, `read_file`, `write_file`, `make_directory`
- Sandboxed Python execution tool: `run_python`
- Git read tools: `git_status`, `git_diff`, `git_log`, `git_show`
- Runtime info tools: `agent_info`, `token_usage`, `current_time`
- Simple local RAG over `.md` and `.json` files in `knowledge/`
- Retrieval tools: `rag_search`, `rag_refresh_index`
- SCADA stub tool for planned MQTT/OPC-UA integrations
- Interactive CLI mode and one-shot prompt mode

## Requirements

- Python 3.10+
- Ollama installed and running locally
- An available Ollama model (default: `qwen2.5-coder:14b`)

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
ollama pull qwen2.5-coder:14b
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
python main.py --model qwen2.5-coder:14b --workspace workspace
```

### Custom knowledge folder

```bash
python main.py --knowledge-dir knowledge
```

## CLI Arguments

- `--workspace`: Sandbox directory used by tools (default: `workspace`)
- `--knowledge-dir`: Folder used as the RAG knowledge base (default: `knowledge`)
- `--model`: Ollama model name (default: `qwen2.5-coder:14b`)
- `--embedding-model`: Ollama embedding model for retrieval (default: `nomic-embed-text`)
- `--raw-llm` / `--no-raw-llm`: Enable or disable raw LLM debug output (default: disabled)
- `--prompt`: Run one prompt and exit (if omitted, interactive mode starts)

## How It Works

1. `main.py` parses CLI args and builds the LangGraph app.
2. `core/graph.py` builds a small RAG index from `knowledge/` and injects retrieved context into the planner and brain prompts.
3. The `planner` node analyzes the prompt and creates a step-by-step plan **without** taking actions.
4. The `brain` node executes the plan by generating tool calls.
5. If tool calls are requested, execution routes through `ToolNode`.
6. Tool outputs are normalized and fed back into state.
7. Loop exits when no tool call remains or step limit is reached (max 24 steps per prompt, after planning).
8. File-generation turns can run a verification pass and automatically repair obvious CLI argument issues before finalizing.

### Planning Phase

The planner node runs first and creates a clear execution strategy. This improves:
- **Multi-step task completion:** Complex prompts are broken into logical steps upfront
- **Step efficiency:** Brain execution is more direct, wasting fewer steps on trial-and-error
- **Multilingual support:** Planning clarifies non-English prompts before reasoning begins
- **Reliability:** Reduces repetition errors and ensures all tasks are considered

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
- Generated files and verification artifacts are created inside the sandboxed `workspace/` directory.
- SCADA integrations are currently placeholders and not yet connected to real PLC/telemetry endpoints.

## Best Practices for Prompts

- **Use English:** Simpler prompts in English work best; complex non-ASCII text may confuse tokenization.
- **One task per prompt:** Bundle logically related steps, but avoid 5+ independent operations.
- **Be explicit:** State expected output format and verification steps clearly.
- **Break into steps:** If your prompt requires multiple independent scripts/files, consider running them separately.
- **Example good prompt:** `"Create sensor.py that reads temperature and saves to temp.json. Run it and show me the output."`
