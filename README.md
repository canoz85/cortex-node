# CortexNode

Local-first AI agent built with LangGraph + Ollama. The current implementation is a controller-driven execution system: the planner proposes work, the controller decides what happens next, and the brain only executes the currently active step.

## Current execution model

The live graph is centered on the protocol and controller, not a loose brain-led loop.

- `planner`: creates an `ExecutionPlan` and concrete `ExecutionStep` objects.
- `controller`: evaluates the latest protocol state and decides the next legal action: dispatch planner, dispatch brain, run tools, summarize, or terminate.
- `brain`: executes the single currently active step. It does not own planning, ordering, retries, or final user response.
- `tools`: invokes the actual tool calls requested by the brain.
- `capture_tool_output`: normalizes the tool result into structured protocol payloads.
- `summarize_memory`: terminal summary path after execution ends.

The effective loop is:

```text
planner -> controller -> brain -> tools -> capture_tool_output -> controller -> ...
```

`controller` decides when to continue, retry, advance to the next step, or stop. The controller is the authority for execution decisions, step transitions, and termination conditions.

## What the current code does

### Planner / Controller / Brain split

The ownership boundaries are explicit in the active prompts and protocol contracts:

- Planner owns the execution plan and step definitions.
- Controller owns execution order, iterations, retries, stopping conditions, and final user-visible completion logic.
- Brain owns only the current active step and returns structured outcomes for the controller to consume.

This is enforced in the runtime prompts and in `core/protocol/controller.py`, which validates exactly one worker result at a time and chooses the next legal transition.

### Active-step execution model

The brain operates in a strict active-step mode:

- It only works on the current `active_step`.
- It may request a tool call or return a step-level outcome.
- It is not allowed to re-order the plan or decide the final answer on its own.

The execution brief passed to the brain includes the full plan and highlights the active step. This keeps the model focused on the current objective instead of broad plan improvisation.

### Step Completion Checker

The project includes a dedicated completion check path driven by `STEP_COMPLETED_SYSTEM_PROMPT` in `core/graph_constants.py`.

Its job is to answer one question only: is the current active step complete or unreachable?

The checker is instructed to:

- evaluate the accumulated evidence across the active step, not only the newest result;
- treat prior successful tool results as valid unless later evidence directly contradicts them;
- return `YES` if the intent is satisfied or if it is demonstrably unreachable;
- return `NO` if the step is still incomplete or requires additional verification.

The check is intentionally narrow: it does not decide plan strategy or final messaging. The controller interprets that answer and advances or terminates execution.

### Evidence semantics

The brain assembles cumulative execution evidence from `tool_execution_history`, including prior successful facts and prior failures. Important semantics in the current code:

- a later failed tool call does not invalidate an earlier successful result;
- evidence is cumulative across the active step;
- successful prior execution remains relevant unless newer evidence explicitly disproves it;
- a step is not considered complete simply because the last tool call failed or because only the latest output is examined.

This is reflected in the `Execution evidence v1` block built in `core/graph_brain.py` and in the completion-checker prompt text.

### Completion vs. unreachability

The completion checker distinguishes two ways a step can be treated as terminal:

- `completed`: the original step intent has been satisfied;
- `unreachable`: the intent cannot be achieved under the current constraints and no meaningful allowed action remains.

Both are considered terminal states for the active step, but they are not the same outcome. A failed tool alone is not enough to mark a step as satisfied or unreachable.

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
