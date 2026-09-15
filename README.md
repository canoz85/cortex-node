# CortexNode

Local-first AI agent built with LangGraph + Ollama. The current implementation is a controller-driven execution system: the planner proposes work, the controller decides what happens next, and the brain only executes the currently active step.

## Current execution model

The production graph is a thin persistence/presentation adapter around portable
Controller turns:

- `controller`: the sole registered production lifecycle node and semantic authority.
- `PortableExecutionRuntime`: prepares and reconciles portable turns.
- `ExecutionDriver`: applies Controller transitions and invokes the authorized worker.
- Planner and Brain: callable typed worker adapters inside the portable turn.
- Tool Runtime: direct `ToolRuntimePort` execution returning portable `ToolResult`.
- Finalizer: exclusive producer of `ExecutionSummary` and the final answer.

The effective loop is:

```text
START -> controller -> controller -> ... -> END
```

The older Planner/Brain/ToolNode/Capture/Summary node graph remains only as an
injected test/integration compatibility path; it is not production topology.

## What the current code does

### Planner / Controller / Brain split

The ownership boundaries are explicit in the active prompts and protocol contracts:

- Planner owns proposal generation; Controller owns the accepted plan and revision.
- Controller owns execution order, authorization, retries, and termination.
- Brain returns typed outcomes for one active step; Controller accepts their lifecycle
  meaning and binds completion provenance.
- Finalizer owns terminal summary and final-answer generation.

This is enforced in the runtime prompts and in `core/protocol/controller.py`, which validates exactly one worker result at a time and chooses the next legal transition.

### Active-step execution model

The brain operates in a strict active-step mode:

- It only works on the current `active_step`.
- It may request a tool call or return a step-level outcome.
- It is not allowed to re-order the plan or decide the final answer on its own.

The execution brief passed to the brain includes the full plan and highlights the active step. This keeps the model focused on the current objective instead of broad plan improvisation.

### Step completion

There is no separate YES/NO Brain completion checker in the production lifecycle.
Brain returns a typed completion, failure, replan, tool-request, or other supported
outcome. Controller validates completion coverage and binds accepted evidence
provenance before changing step state.

### Evidence semantics

The brain assembles cumulative execution evidence from `tool_execution_history`, including prior successful facts and prior failures. Important semantics in the current code:

- a later failed tool call does not invalidate an earlier successful result;
- evidence is cumulative across the active step;
- successful prior execution remains relevant unless newer evidence explicitly disproves it;
- a step is not considered complete simply because the last tool call failed or because only the latest output is examined.

This evidence is projected into Brain input and is bound to accepted completion
provenance by Controller.

The retained `CompletionService` performs deterministic completion-requirement and
coverage validation; it is not the removed checker. A failed tool alone does not mark
a step complete.

## Controller ownership

The controller is the execution owner in the current implementation:

- it enforces max reasoning limits;
- it decides when to request a planner rework;
- it turns brain `TOOL_REQUEST` into tool execution;
- it processes tool success/failure and routes back to the brain;
- it advances to the next step when a step is complete;
- it terminates on max-step or failure conditions.

The controller is also the location where tool result mismatches and invalid continuations are rejected. This is the authoritative state transition layer.

## Protocol / data contracts

The project has an explicit protocol layer under `core/protocol/`.

Core types include:

- `ExecutionPlan` and `ExecutionStep`
- `ExecutionCursor`
- `ToolRequest` and `ToolResult`
- `BrainInput` and `BrainResult`
- `ControllerInput` and `ControllerDecision`
- `ExecutionState` with `protocol_visible` and `working` sections

The key design choice is that `ExecutionState.protocol_visible` is the authoritative accepted state, while `working` holds runtime orchestration metadata. The controller writes the accepted-state transitions; the workers consume typed input contracts rather than ad hoc state dictionaries.

## Setup

### Requirements

- Python 3.10+
- Ollama running locally
- A model available in Ollama (default examples in the project use `qwen2.5-coder:14b` or `gpt-oss:20b` depending on settings)

### Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

Interactive mode:

```bash
python main.py
```

Single prompt:

```bash
python main.py --prompt "Create hello.py in the workspace and run it"
```

Optional config is still supported through the CLI and environment variables. See `main.py` and the project config handling for the current defaults.

Observe-only GPU telemetry is enabled by default for the local CLI. It records
VRAM, GPU utilization, visible GPU processes, and operation duration around
Planner, Brain, tool execution, summary, and async-provider polling boundaries.
It never unloads a model or changes scheduling. Disable it with
`--no-gpu-telemetry` or `CORTEX_GPU_TELEMETRY=false`.

## Tools and capabilities

CortexNode currently exposes a sandboxed tool set including:

- file system: `list_files`, `read_file`, `write_file`, `make_directory`
- Python execution: `run_python`, `install_package`
- git: `git_status`, `git_log`, `git_show`, `git_diff`
- runtime: `agent_info`, `token_usage`, `current_time`
- knowledge: `rag_search`, `rag_refresh_index`
- SAP / SCADA / vision tools depending on the active tool bundle

## Quality checks

Run the local test suite:

```bash
python -m pytest
```

Graph-oriented regression checks are also available in the project tests and are designed around the controller/planner/brain execution flow.

## Notes

This README reflects the implementation currently in the repository, not a planned future architecture. The active behavior is controller-owned execution with explicit protocol contracts and active-step completion checks.

- File and execution tools enforce sandbox boundaries relative to the selected workspace.
- The controller and protocol layer are the authoritative execution state path; the brain is intentionally scoped to the active step.
- The runtime still includes RAG, git, file, runtime, SAP, and SCADA tool bundles depending on the active setup.
- Current evidence handling is cumulative and explicit: failed later tool calls do not automatically invalidate earlier successful results for the same step.
- The project may still have legacy references in some prompts or historical notes, but the current execution logic is the protocol-driven controller model described above.
- **One task per prompt:** Bundle logically related steps, but avoid 5+ independent operations.
- **Be explicit:** State expected output format and verification steps clearly.
- **Break into steps:** If your prompt requires multiple independent scripts/files, consider running them separately.
- **Example good prompt:** `"Create sensor.py that reads temperature and saves to temp.json. Run it and show me the output."`
