from __future__ import annotations

from core.protocol.models import BrainOutcome
from core.logging.node_update import NodeUpdate

from .console import (
    ANSI_CYAN,
    ANSI_GREEN,
    ANSI_LIGHT_BLUE,
    ANSI_RESET,
)

from .formatter import (
    format_ai_message,
    format_planner_plan,
    format_tool_call_preview,
    format_tool_result,
)


def render_node_update(node_update: NodeUpdate) -> None:
    """Render a normalized node update."""

    if node_update.planner_result is not None:
        _render_planner(node_update)
        return

    if (
        node_update.brain_result is not None
        and node_update.brain_result.outcome == BrainOutcome.TOOL_REQUEST
    ):
        _render_tool_request(node_update)
        return

    if node_update.tool_result is not None:
        _render_tool_result(node_update)
        return

    if node_update.ai_message is not None:
        _render_ai_text(node_update)


def _render_planner(node_update: NodeUpdate) -> None:

    planner = node_update.planner_result

    header = "[planner]"
    if planner.outcome:
        header = f"[planner:{planner.outcome.value}]"

    print(f"\n{ANSI_GREEN}{header}{ANSI_RESET}")
    print(format_planner_plan(planner))
    print()


def _render_tool_request(node_update: NodeUpdate) -> None:

    print(f"\n{ANSI_CYAN}[brain]{ANSI_RESET}")

    if node_update.ai_message is not None:
        print(format_tool_call_preview(node_update.ai_message))

    print()


def _render_tool_result(node_update: NodeUpdate) -> None:

    print(f"\n{ANSI_CYAN}[tools]{ANSI_RESET}")
    print(format_tool_result(node_update.tool_result))
    print()


def _render_ai_text(node_update: NodeUpdate) -> None:

    text = format_ai_message(node_update.ai_message)

    if not text:
        return

    print(f"\n{ANSI_LIGHT_BLUE}[brain]{ANSI_RESET}")
    print(f"{ANSI_LIGHT_BLUE}{text}{ANSI_RESET}")
    print()