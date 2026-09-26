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


def render_node_update(node_update: NodeUpdate, *, verbose: bool = False) -> None:
    """Render a normalized node update."""

    if node_update.accepted_plan is not None:
        _render_controller_event(node_update, "Plan ready", verbose=verbose)
        _render_accepted_plan(node_update)
    elif node_update.planner_result is not None:
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


def render_system_message(message: str) -> None:
    """Render user-visible application/session status consistently."""
    print("\n[system]")
    print(message)


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

    # Successful plans are rendered once by the authoritative detailed Planner
    # debug logger. Keep console rendering here for non-plan outcomes only.
    if planner.proposed_plan is not None:
        return

    header = "[planner]"
    if planner.outcome:
        header = f"[planner:{planner.outcome.value}]"

    print(f"\n{ANSI_GREEN}{header}{ANSI_RESET}")
    print(format_planner_plan(planner))
    print()


def _render_tool_request(node_update: NodeUpdate) -> None:

    print(f"\n{ANSI_CYAN}[brain]{ANSI_RESET}")

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
    print(f"\n{ANSI_CYAN}[{label}]{ANSI_RESET}")
    print(format_tool_result(node_update.tool_result))
    print()


def _render_ai_text(node_update: NodeUpdate, *, label: str = "brain") -> None:

    text = format_ai_message(node_update.ai_message)

    if not text:
        return

    print(f"\n{ANSI_LIGHT_BLUE}[{label}]{ANSI_RESET}")
    print(f"{ANSI_LIGHT_BLUE}{text}{ANSI_RESET}")
    print()
