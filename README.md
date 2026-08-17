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

### Optional config file (JSON)

```bash
python main.py --config config.json
```

Config merge order is: `defaults -> environment -> config file -> CLI`.

Example `config.json`:

```json
{
    "workspace": "workspace",
    "knowledge_dir": "knowledge",
    "model": "qwen2.5-coder:14b",
    "embedding_model": "nomic-embed-text",
    "rag_top_k": 4,
    "raw_llm": false,
    "show_summary": false,
    "log_level": "INFO",
    "json_logs": false
}
```

## CLI Arguments

- `--workspace`: Sandbox directory used by tools (default: `workspace`)
- `--knowledge-dir`: Folder used as the RAG knowledge base (default: `knowledge`)
- `--model`: Ollama model name (default: `qwen2.5-coder:14b`)
- `--embedding-model`: Ollama embedding model for retrieval (default: `nomic-embed-text`)
- `--rag-top-k`: Number of retrieved knowledge chunks per query (default: `4`)
- `--raw-llm` / `--no-raw-llm`: Enable or disable raw LLM debug output (default: disabled)
- `--show-summary` / `--no-show-summary`: Enable or disable rolling summary output
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--json-logs` / `--no-json-logs`: Enable or disable JSON log output
- `--config`: Optional JSON config file
- `--prompt`: Run one prompt and exit (if omitted, interactive mode starts)

### Environment variables

- `CORTEX_WORKSPACE`
- `CORTEX_KNOWLEDGE_DIR`
- `CORTEX_MODEL`
- `CORTEX_EMBEDDING_MODEL`
- `CORTEX_RAG_TOP_K`
- `CORTEX_RAW_LLM`
- `CORTEX_SHOW_SUMMARY`
- `CORTEX_LOG_LEVEL`
- `CORTEX_JSON_LOGS`

## Tests

Run unit tests:

```bash
python -m pytest
```

Coverage is enforced via `pytest.ini` with:

- `--cov=core --cov=tools`
- `--cov-report=term-missing`
- `--cov-fail-under=71` (baseline gate, intended to be raised over time)

Ratcheting policy is configured in `.github/coverage-policy.json`.
Run policy check manually:

```bash
python scripts/coverage_ratchet.py --coverage-json coverage.json --policy .github/coverage-policy.json
```

Run the graph-focused regression suite used during orchestration refactors:

```bash
python -m pytest tests/test_graph_nodes.py tests/test_graph_runner.py tests/test_graph_capture.py tests/test_graph_messages.py tests/test_graph_planner.py tests/test_graph_routing.py tests/test_graph_intents.py
```

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

CI runs:

- dependency installation
- full pytest run with coverage gate
- benchmark skeleton smoke run (dry mode)
- evaluation dataset validation with route/global policy checks (`.github/evaluation-policy.json`)

## PR/Release Checklist

Before merging a change, confirm:

- Tests pass locally and in CI.
- Coverage gate passes and ratchet status is reviewed (`benchmarks/results/coverage-ratchet.md`).
- Benchmark smoke run passes and trend delta is checked (`benchmarks/results/trend.md`).
- If ratchet status is ready-to-ratchet, raise `--cov-fail-under` in `pytest.ini` and update `.github/coverage-policy.json`.

## Benchmark Skeleton

Benchmark scenarios live in `benchmarks/scenarios.json` and are executed by `scripts/benchmark.py`.

Dry-run validation (CI-safe):

```bash
python scripts/benchmark.py --cases benchmarks/scenarios.json --output benchmarks/results/local-skeleton.json
```

Live benchmark execution:

```bash
python scripts/benchmark.py --live --cases benchmarks/scenarios.json --output benchmarks/results/local-live.json
```

Generate trend markdown from benchmark result JSON files:

```bash
python scripts/benchmark_trend.py --results-dir benchmarks/results --output benchmarks/results/trend.md
```

## Evaluation Dataset And Scoring Dashboard

Evaluation dataset file:

- `benchmarks/evaluation_dataset.json`

Run validation-only evaluation (dataset/schema check):

```bash
python scripts/run_evaluation.py --dataset benchmarks/evaluation_dataset.json
```

Run live evaluation and generate dashboard outputs:

```bash
python scripts/run_evaluation.py --live --semantic-scoring --semantic-model nomic-embed-text --dataset benchmarks/evaluation_dataset.json --policy .github/evaluation-policy.json --enforce-policy --output benchmarks/results/evaluation-latest.json --dashboard-md benchmarks/results/evaluation-dashboard.md --dashboard-json benchmarks/results/evaluation-dashboard.json
```

Use one-command local checks including evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File run-checks.ps1 -RunEvaluation -EvaluationMaxCases 3
```

## How It Works

1. `main.py` parses CLI args and builds the LangGraph app.
2. `core/graph.py` builds a small RAG index from `knowledge/` and injects retrieved context into the planner and brain prompts.
3. The `planner` node analyzes the prompt and creates a step-by-step plan **without** taking actions.
4. The `brain` node in `core/graph_brain.py` executes the plan by generating tool calls and delegates branching decisions to `core/graph_state_machine.py` for execution policy, action recovery, and repeated-signature enforcement.
5. If tool calls are requested, execution routes through `ToolNode`.
6. Tool outputs are normalized and fed back into state.
7. Loop exits when no tool call remains or step limit is reached (max 24 steps per prompt, after planning).
8. File-generation turns can run a verification pass and automatically repair obvious CLI argument issues before finalizing.

### Brain Node Flow

- Early-return fast path handles domain clarification, read-audit answers, required-first-tool enforcement, direct discussion turns, file-generation deterministic next steps, and action completion summaries.
- Pre-message assembly builds route-aware guidance for SAP, read-only analysis, file generation, preferred tool usage, and failure recovery.
- Response recovery normalizes pseudo-tool text, retries empty outputs, and suppresses accidental tool calls on discussion-only turns.
- Post-response guards prevent repeated tool signatures, unsafe read-only mutations, unchanged rewrites after failed verification, fabricated workspace-analysis claims, and plain-text action loops.

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
|-- README.md
|-- main.py
|-- pytest.ini
|-- requirements.txt
|-- core/
|   |-- __init__.py
|   |-- error_codes.py
|   |-- graph_capture.py
|   |-- graph_constants.py
|   |-- graph_context.py
|   |-- graph_filegen_policy.py
|   |-- graph_intents.py
|   |-- graph_messages.py
|   |-- graph_node_helpers.py
|   |-- graph_nodes.py
|   |-- graph_planner.py
|   |-- graph_pseudo_tools.py
|   |-- graph_response_formatters.py
|   |-- graph_routing.py
|   |-- graph_runner.py
|   |-- graph_state_machine.py
|   |-- graph_tool_events.py
|   |-- graph.py
|   |-- graph_brain.py
|   |-- logging_utils.py
|   |-- models.py
|   |-- rag.py
|   |-- state.py
|   `-- tool_output.py
|-- knowledge/
|   |-- example_rules.md
|   |-- examples.json
|   |-- sap_examples.json
|   `-- sap_rules.md
|-- prompts/
|   `-- systemprompts_sap.md
|-- scripts/
|   `-- hello.py
|-- tests/
|   |-- conftest.py
|   |-- test_exec_ops.py
|   |-- test_file_ops.py
|   |-- test_git_ops.py
|   |-- test_graph_capture.py
|   |-- test_graph_intents.py
|   |-- test_graph_messages.py
|   |-- test_graph_node_helpers.py
|   |-- test_graph_nodes.py
|   |-- test_graph_planner.py
|   |-- test_graph_routing.py
|   |-- test_graph_runner.py
|   |-- test_graph_tool_events.py
|   `-- test_info_ops.py
|-- tools/
|   |-- __init__.py
|   |-- exec_ops.py
|   |-- file_ops.py
|   |-- git_ops.py
|   |-- info_ops.py
|   |-- rag_ops.py
|   |-- sap_ops.py
|   `-- scada_ops.py
|-- workspace/
|   |-- sandbox files created or modified by the agent
|   |-- is_prime.py
|   |-- lcm_calculator.py
|   `-- lcm_calculator_cli.py
```

### Core Module Guide

- `core/graph.py`: graph wiring, model setup, and app construction.
- `core/graph_nodes.py`: planner/brain node orchestration and execution guardrails.
- `core/graph_brain.py`: route-aware brain execution, response recovery, and repeated-signature enforcement.
- `core/graph_state_machine.py`: typed decision layer for routing, action recovery, retries, and signature policy.
- `core/error_codes.py`: canonical tool error codes used for structured failures and observability.
- `core/graph_filegen_policy.py`: deterministic file-generation verification and repair helpers.
- `core/graph_messages.py` and `core/graph_tool_events.py`: message normalization and tool-event extraction.
- `core/graph_response_formatters.py`: deterministic completion and info-tool response formatting.
- `tools/`: sandboxed file, execution, git, info, RAG, SAP, and SCADA tool implementations.
- `tests/`: focused unit and graph regression coverage for planner, brain, routing, capture, and tool behavior.

## Notes

- File and execution tools enforce sandbox boundaries relative to the selected workspace.
- Brain execution is split across `core/graph_brain.py` (orchestration and LLM invocation) and `core/graph_state_machine.py` (pure decision functions such as `decide_brain_execution`, `decide_action_recovery`, and `decide_repeated_signature`).
- `core/rag.py` caches search results per query and `top_k`, and `refresh()` clears that cache when the knowledge base changes.
- `core/graph_runner.py` logs run-level observability counters on completion, including node updates, tool-call counts, tool-result counts, duration, stop reason, and `error_counts` grouped by error code.
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
