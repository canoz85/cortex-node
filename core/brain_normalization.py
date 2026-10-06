"""One defensive boundary from model/provider output to domain BrainOutcome.

Exactly one native call is executable. Text and provider envelopes are never
parsed for tools or lifecycle outcomes. This module is internal to the provider
adapter; no raw response is returned by the Brain service.
"""

import hashlib
import json
from collections.abc import Mapping

from pydantic import ValidationError

from core.protocol.enums import BrainOutcomeKind as Kind, StepStatus
from core.protocol.models import (
    BrainInput, BrainOutcome, BrainUsage, ExactCollection, ReplanRequest,
    StepCompletionEvidence, ToolRequest,
)


class InvalidBrainOutput(ValueError):
    pass


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _only_fields(value: dict, allowed: set[str]):
    if value.keys() - allowed:
        raise InvalidBrainOutput("unexpected_envelope_fields")


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidBrainOutput(f"missing_or_invalid_{field}")
    return value


def _exact_collection(value) -> ExactCollection | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidBrainOutput("invalid_exact_collection")
    _only_fields(value, {"source_record_index", "data_path", "label"})
    source_record_index = value.get("source_record_index")
    if isinstance(source_record_index, bool) or not isinstance(source_record_index, int):
        raise InvalidBrainOutput("invalid_exact_collection_source_record_index")
    path = value.get("data_path")
    if (
        not isinstance(path, list)
        or any(
            isinstance(part, bool) or not isinstance(part, (str, int))
            for part in path
        )
    ):
        raise InvalidBrainOutput("invalid_exact_collection_data_path")
    label = value.get("label", "Results")
    if not isinstance(label, str) or not label.strip():
        raise InvalidBrainOutput("invalid_exact_collection_label")
    return ExactCollection(
        source_record_index=source_record_index,
        data_path=tuple(path),
        label=label.strip(),
    )


def _parse_call(call) -> tuple[str, dict]:
    if not isinstance(call, dict):
        raise InvalidBrainOutput("invalid_tool_call")
    _only_fields(call, {"id", "type", "name", "args"})
    name = _text(call.get("name"), "tool_name")
    arguments = call.get("args")
    if not isinstance(arguments, dict):
        raise InvalidBrainOutput("tool_arguments_must_be_object")
    return name, arguments

def _tool_request(
    name: str,
    arguments: dict,
    brain_input: BrainInput,
    allowed_tools: set[str],
) -> ToolRequest:
    if name not in allowed_tools:
        raise InvalidBrainOutput("unknown_tool")

    identity = {
        "execution": brain_input.identity.execution_id,
        "cursor": brain_input.cursor.model_dump(mode="json"),
        "history_length": len(brain_input.tool_execution_history),
        "retry": brain_input.retry.retry_count,
        "name": name,
        "arguments": arguments,
    }

    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:24]

    return ToolRequest(
        request_id=f"{brain_input.identity.execution_id}:tool:{digest}",
        tool_name=name,
        arguments=arguments,
    )

def _tool_requested_outcome(
    *,
    name: str,
    arguments: dict,
    brain_input: BrainInput,
    allowed_tools: set[str],
) -> BrainOutcome:
    if brain_input.direct_response or brain_input.active_step is None:
        raise InvalidBrainOutput("tool_call_requires_active_step")

    return BrainOutcome(
        outcome=Kind.TOOL_REQUESTED,
        step_id=brain_input.active_step.step_id,
        tool_request=_tool_request(
            name,
            arguments,
            brain_input,
            allowed_tools,
        ),
        message="Brain requested tool execution.",
    )


def _native_outcome(
    calls, brain_input: BrainInput, allowed_tools: set[str],
) -> BrainOutcome:
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        raise InvalidBrainOutput("exactly_one_tool_call_required")
    if brain_input.direct_response or brain_input.active_step is None:
        raise InvalidBrainOutput("native_call_requires_active_step")
    name, arguments, exact_collection = validate_native_call(calls[0], allowed_tools)
    if name in allowed_tools:
        return _tool_requested_outcome(
            name=name,
            arguments=arguments,
            brain_input=brain_input,
            allowed_tools=allowed_tools,
        )

    step_id = brain_input.active_step.step_id
    if name == "brain_step_completed":
        summary = arguments["message"]
        return BrainOutcome(
            outcome=Kind.STEP_COMPLETED, step_id=step_id, message=summary,
            completion_evidence=StepCompletionEvidence(
                step_id=step_id, summary=summary,
                exact_collection=exact_collection,
            ),
            proposed_step_status=StepStatus.COMPLETED,
        )
    if name == "brain_step_failed":
        return BrainOutcome(
            outcome=Kind.STEP_FAILED, step_id=step_id,
            message=arguments["message"],
            proposed_step_status=StepStatus.FAILED,
        )
    if name == "brain_replan_requested":
        reason = arguments["reason"]
        constraints = arguments["constraints"]
        return BrainOutcome(
            outcome=Kind.REPLAN_REQUESTED, step_id=step_id, message=reason,
            replan_request=ReplanRequest(
                reason=reason, failed_step_id=step_id, constraints=tuple(constraints),
            ),
        )
    raise InvalidBrainOutput("unknown_tool")


def validate_native_call(
    call, allowed_tools: set[str],
) -> tuple[str, dict, ExactCollection | None]:
    """Validate a candidate without creating an outcome or ToolRequest ID."""
    name, arguments = _parse_call(call)
    exact_collection = None
    if name in allowed_tools:
        return name, arguments, None
    if name == "brain_step_completed":
        collection_fields = {
            "exact_collection_source_record_index": "source_record_index",
            "exact_collection_data_path": "data_path",
            "exact_collection_label": "label",
        }
        _only_fields(arguments, {"message", *collection_fields})
        _text(arguments.get("message"), "message")
        proposal = {
            internal: arguments[external]
            for external, internal in collection_fields.items()
            if external in arguments
        }
        exact_collection = _exact_collection(proposal if proposal else None)
    elif name == "brain_step_failed":
        _only_fields(arguments, {"message"})
        _text(arguments.get("message"), "message")
    elif name == "brain_replan_requested":
        _only_fields(arguments, {"reason", "constraints"})
        _text(arguments.get("reason"), "reason")
        constraints = arguments.get("constraints")
        if not isinstance(constraints, list) or any(not isinstance(item, str) for item in constraints):
            raise InvalidBrainOutput("invalid_replan_constraints")
    else:
        raise InvalidBrainOutput("unknown_tool")
    return name, arguments, exact_collection


def normalize_brain_output(
    raw: object, brain_input: BrainInput, allowed_tools: set[str],
) -> BrainOutcome:
    """Consume only the provider's canonical native call channel; never recover text."""
    try:
        if getattr(raw, "invalid_tool_calls", None):
            raise InvalidBrainOutput("invalid_native_tool_call")
        if getattr(raw, "content", None):
            raise InvalidBrainOutput("unexpected_response_content")
        calls = getattr(raw, "tool_calls", None)
        if not calls:
            raise InvalidBrainOutput("native_tool_call_required")
        return _native_outcome(calls, brain_input, allowed_tools)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        code = str(exc) if isinstance(exc, InvalidBrainOutput) else "malformed_model_output"
        return BrainOutcome(
            outcome=Kind.INVALID_OUTPUT, error_code=code,
            step_id=brain_input.active_step.step_id if brain_input.active_step else None,
            message=f"Brain returned invalid output ({code}).",
        )


def normalize_brain_usage(raw: object) -> BrainUsage:
    """Ignore malformed accounting metadata without changing a valid outcome."""
    metadata = _field(raw, "response_metadata", {}) or {}
    usage = _field(raw, "usage_metadata", {}) or {}
    try:
        return BrainUsage(
            prompt_tokens=usage.get("input_tokens", metadata.get("prompt_eval_count", metadata.get("prompt_tokens", 0))),
            completion_tokens=usage.get("output_tokens", metadata.get("eval_count", metadata.get("completion_tokens", 0))),
        )
    except (ValidationError, AttributeError):
        return BrainUsage()
