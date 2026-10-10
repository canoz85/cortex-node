from __future__ import annotations

from core.protocol.enums import BrainOutcome
from core.logging.node_update import NodeUpdate

from .console import (
    ANSI_CYAN,
    ANSI_GREEN,
    ANSI_LIGHT_BLUE,
    ANSI_RESET,
)

from .formatter import (
    format_accepted_plan,
    format_ai_message,
    format_planner_plan,
    format_tool_call_preview,
    format_tool_result,
)
from .live_status import current_live_status
from .live_status import LiveStatus


def render_node_update(node_update: NodeUpdate, *, verbose: bool = False) -> None:
    """Render a normalized node update."""
    status = current_live_status()
    if status is not None:
        with status.permanent_output(redraw=False):
            _render_node_update(node_update, verbose=verbose)
        changed = _update_live_status(node_update, verbose=verbose)
        if not changed:
            status.redraw()
        return
    _render_node_update(node_update, verbose=verbose)


def _render_node_update(node_update: NodeUpdate, *, verbose: bool = False) -> None:

    if node_update.accepted_plan is not None:
        _render_controller_event(node_update, "Plan ready", verbose=verbose)
        # _render_accepted_plan(node_update)
    elif node_update.planner_result is not None:
        if node_update.planner_failure_retryable:
            pass
        else:
            _render_planner(node_update)

    if node_update.tool_result is not None:
        _render_tool_result(node_update)
        _render_controller_events(node_update, prefix="Tool completed", verbose=verbose)

    # Accepted completion content remains protocol evidence for the Finalizer;
    # it is not a second user-facing answer.
    _render_controller_events(node_update, prefix="Step completed", verbose=verbose)

    if (
        node_update.brain_result is not None
        and node_update.brain_result.outcome == BrainOutcome.TOOL_REQUEST
    ):
        _render_tool_request(node_update)
        _render_controller_events(node_update, prefix="Tool requested", verbose=verbose)

    if node_update.finalization_result is not None:
        _render_controller_event(node_update, "Execution completed", verbose=verbose)
        _render_ai_text(node_update, label="finalizer")
        return

    if (
        node_update.brain_result is not None
        and node_update.brain_result.outcome in {
            BrainOutcome.FINAL_ANSWER,
            BrainOutcome.STEP_COMPLETED,
        }
    ):
        return

    if (
        node_update.ai_message is not None
        and node_update.tool_result is None
        and (
            node_update.brain_result is None
            or node_update.brain_result.outcome
            not in {
                BrainOutcome.TOOL_REQUEST,
                BrainOutcome.FINAL_ANSWER,
                BrainOutcome.STEP_COMPLETED,
            }
        )
    ):
        _render_ai_text(node_update)


def _update_live_status(node_update: NodeUpdate, *, verbose: bool) -> bool:
    """Map normalized semantic events to concise temporary progress text."""

    status = current_live_status()
    if status is None:
        return False

    if node_update.planner_failure_retryable:
        status.update("planner", node_update.planner_retry_detail)
    elif node_update.finalization_result is not None:
        status.update("finalizer")
    elif node_update.brain_result is not None:
        request = node_update.brain_result.tool_request
        if request is None:
            return False
        status.update(request.tool_name)
    elif node_update.planner_result is not None:
        status.update("planner", "planning")
    else:
        semantic_changed = False
        if not verbose or not node_update.controller_events:
            return semantic_changed
        event = node_update.controller_events[-1]
        detail = _controller_status_detail(event)
        if detail is None:
            return semantic_changed
        status.update("controller", detail)
        return True

    if not verbose or not node_update.controller_events:
        return True

    event = node_update.controller_events[-1]
    detail = _controller_status_detail(event)
    if detail is None:
        return True
    status.update("controller", detail)
    return True


def _controller_status_detail(event: str) -> str | None:
    if event == "Plan ready":
        return "plan ready"
    elif event.startswith("Tool requested"):
        return "dispatching tool"
    elif event.startswith("Tool completed"):
        return "evaluating evidence"
    elif event.startswith("Step completed"):
        return "step completed"
    elif event == "Execution completed":
        return "execution completed"
    return None


def render_system_message(message: str) -> None:
    """Render user-visible application/session status consistently."""
    status = current_live_status()
    if status is not None:
        with status.permanent_output():
            print("\n[system]")
            print(message)
    else:
        print("\n[system]")
        print(message)


def render_usage_summary(status: LiveStatus) -> None:
    """Render a completed turn's provider usage when at least one call occurred."""
    summary = status.format_usage_summary()
    if summary:
        print(f"\n[usage] {summary}")


def _render_controller_events(
    node_update: NodeUpdate, *, prefix: str, verbose: bool
) -> None:
    if not verbose:
        return
    for event in node_update.controller_events:
        if event.startswith(prefix):
            _print_controller_event(event)


def _render_controller_event(
    node_update: NodeUpdate, event: str, *, verbose: bool
) -> None:
    if verbose and event in node_update.controller_events:
        _print_controller_event(event)


def _print_controller_event(event: str) -> None:
    print("\n[controller]")
    print(event)


def _render_accepted_plan(node_update: NodeUpdate) -> None:
    print(f"\n{ANSI_GREEN}[planner]{ANSI_RESET}")
    print(format_accepted_plan(node_update.accepted_plan))
    print()


def _render_planner(node_update: NodeUpdate) -> None:

    planner = node_update.planner_result

    # # Successful plans are rendered once by the authoritative detailed Planner
    # # debug logger. Keep console rendering here for non-plan outcomes only.
    # if planner.proposed_plan is not None:
    #     return

    header = "[planner]"
    if planner.outcome:
        header = f"[planner:{planner.outcome.value}]"

    print(f"\n{ANSI_GREEN}{header}{ANSI_RESET}")
    print(
        format_accepted_plan(planner.proposed_plan)
        if planner.proposed_plan is not None
        else format_planner_plan(planner)
    )

    print()


def _render_tool_request(node_update: NodeUpdate) -> None:

    step_id = node_update.brain_result.step_id if node_update.brain_result is not None else None
    context = f" [{step_id}]" if step_id else ""
    print(f"\n{ANSI_CYAN}[brain]{context}{ANSI_RESET}")

    request = (
        node_update.brain_result.tool_request
        if node_update.brain_result is not None
        else None
    )
    if request is not None:
        print(f"Calling {request.tool_name}")
    elif node_update.ai_message is not None:
        print(format_tool_call_preview(node_update.ai_message))

    print()


def _render_tool_result(node_update: NodeUpdate) -> None:

    label = f"tool:{node_update.tool_name}" if node_update.tool_name else "tool"
    context = f" [{node_update.active_step_id}]" if node_update.active_step_id else ""
    print(f"\n{ANSI_CYAN}[{label}]{context}{ANSI_RESET}")
    print(format_tool_result(node_update.tool_result))
    print()


def _render_ai_text(node_update: NodeUpdate, *, label: str = "brain") -> None:

    text = format_ai_message(node_update.ai_message)

    if not text:
        return

    print(f"\n{ANSI_LIGHT_BLUE}[{label}]{ANSI_RESET}")
    print(f"{ANSI_LIGHT_BLUE}{text}{ANSI_RESET}")
    print()
