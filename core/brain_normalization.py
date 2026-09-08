"""One defensive boundary from model/provider output to domain BrainOutcome.

Only complete, unambiguous envelopes are executable. Prose is never searched for
JSON, tool names, or lifecycle markers. This module is internal to the provider
adapter; no raw response is returned by the Brain service.
"""

import ast
import hashlib
import json
import re
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from core.brain_evidence import EvidenceSnapshot, evidence_scope
from core.protocol.enums import BrainOutcomeKind as Kind, StepStatus
from core.protocol.models import (
    BrainInput, BrainOutcome, BrainUsage, FinalAnswerDraft, ReplanRequest,
    StepCompletionEvidence, ToolRequest,
)


class InvalidBrainOutput(ValueError):
    pass


def _json(text: str):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidBrainOutput("duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise InvalidBrainOutput("non_finite_json_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_pairs,
            parse_constant=invalid_constant,
        )
    except Exception as e:
        print("TYPE:", type(e).__name__)
        print("ERROR:", e)
        print("TEXT REPR:", repr(text))
        raise

    # return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _only_fields(value: dict, allowed: set[str]):
    if value.keys() - allowed:
        raise InvalidBrainOutput("unexpected_envelope_fields")


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidBrainOutput(f"missing_or_invalid_{field}")
    return value


def _step_id(payload: dict, brain_input: BrainInput) -> str:
    step_id = _text(payload.get("step_id"), "step_id")
    if brain_input.active_step is None or step_id != brain_input.active_step.step_id:
        raise InvalidBrainOutput("step_id_does_not_match_active_step")
    return step_id


def _tool(call, brain_input: BrainInput, allowed_tools: set[str]) -> ToolRequest:
    if not isinstance(call, dict):
        raise InvalidBrainOutput("invalid_tool_call")
    if "function" in call:
        _only_fields(call, {"id", "type", "function"})
        call = call["function"]
        if not isinstance(call, dict):
            raise InvalidBrainOutput("invalid_tool_function")
    _only_fields(call, {"id", "type", "name", "arguments", "args"})
    name = _text(call.get("name"), "tool_name")
    if name not in allowed_tools:
        raise InvalidBrainOutput("unknown_tool")
    if "arguments" in call and "args" in call:
        raise InvalidBrainOutput("ambiguous_tool_arguments")
    arguments = call.get("arguments", call.get("args", {}))
    if isinstance(arguments, str):
        arguments = _json(arguments)
    if not isinstance(arguments, dict):
        raise InvalidBrainOutput("tool_arguments_must_be_object")
    # Domain IDs are independent of provider tool-call IDs and deterministic for
    # the same invocation, including retries and repeated calls in later turns.
    identity = {
        "execution": brain_input.identity.execution_id,
        "cursor": brain_input.cursor.model_dump(mode="json"),
        "history_length": len(brain_input.tool_execution_history),
        "retry": brain_input.retry.retry_count,
        "name": name, "arguments": arguments,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
    return ToolRequest(
        request_id=f"{brain_input.identity.execution_id}:tool:{digest}",
        tool_name=name, arguments=arguments,
    )


def _tool_outcome(calls, brain_input: BrainInput, allowed_tools: set[str]) -> BrainOutcome:
    if not isinstance(calls, (list, tuple)) or len(calls) != 1:
        raise InvalidBrainOutput("exactly_one_tool_call_required")
    if brain_input.direct_response or brain_input.active_step is None:
        raise InvalidBrainOutput("tool_call_requires_active_step")
    return BrainOutcome(
        outcome=Kind.TOOL_REQUESTED,
        step_id=brain_input.active_step.step_id,
        tool_request=_tool(calls[0], brain_input, allowed_tools),
        message="Brain requested tool execution.",
    )


def _structured(
    payload, brain_input: BrainInput, allowed_tools: set[str], *, allow_text_tool_calls: bool,
    evidence_snapshot: EvidenceSnapshot | None = None,
) -> BrainOutcome:
    if not isinstance(payload, dict):
        raise InvalidBrainOutput("outcome_must_be_object")
    if "kind" not in payload:
        if "tool_calls" in payload:
            if not allow_text_tool_calls:
                raise InvalidBrainOutput("native_tool_call_required")
            _only_fields(payload, {"tool_calls"})
            return _tool_outcome(payload["tool_calls"], brain_input, allowed_tools)
        if "name" in payload or "function" in payload:
            if not allow_text_tool_calls:
                raise InvalidBrainOutput("native_tool_call_required")
            return _tool_outcome([payload], brain_input, allowed_tools)
        raise InvalidBrainOutput("missing_outcome_kind")

    raw_kind = payload["kind"]
    if not isinstance(raw_kind, str):
        raise InvalidBrainOutput("invalid_outcome_kind")
    kind = Kind.__members__.get(raw_kind)
    if kind is None:
        kind = Kind(raw_kind)

    if kind == Kind.TOOL_REQUESTED:
        if not allow_text_tool_calls:
            raise InvalidBrainOutput("native_tool_call_required")
        _only_fields(payload, {"kind", "tool", "step_id"})
        if "step_id" in payload:
            _step_id(payload, brain_input)
        return _tool_outcome([payload.get("tool")], brain_input, allowed_tools)
    if kind == Kind.FINAL_ANSWER_READY:
        _only_fields(payload, {"kind", "answer"})
        if not brain_input.direct_response and not (
            brain_input.active_plan is not None and brain_input.active_step is None
        ):
            raise InvalidBrainOutput("final_answer_requires_final_answer_context")
        return BrainOutcome(
            outcome=kind, final_answer_draft=FinalAnswerDraft(text=_text(payload.get("answer"), "answer")),
        )
    if kind == Kind.STEP_COMPLETED:
        _only_fields(payload, {"kind", "step_id", "message", "evidence_refs"})
        step_id = _step_id(payload, brain_input)
        summary = _text(payload.get("message"), "message")
        refs = payload.get("evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            raise InvalidBrainOutput("invalid_evidence_refs")
        if evidence_snapshot is not None and evidence_snapshot.scope != evidence_scope(brain_input):
            raise InvalidBrainOutput("evidence_snapshot_scope_mismatch")
        bindings = dict(evidence_snapshot.bindings) if evidence_snapshot is not None else {}
        if set(refs) - bindings.keys():
            raise InvalidBrainOutput("unknown_step_evidence")
        ids = tuple(bindings[ref] for ref in refs)
        return BrainOutcome(
            outcome=kind, step_id=step_id, message=summary,
            completion_evidence=StepCompletionEvidence(step_id=step_id, summary=summary, tool_request_ids=ids),
            proposed_step_status=StepStatus.COMPLETED,
        )
    if kind == Kind.REPLAN_REQUESTED:
        _only_fields(payload, {"kind", "step_id", "reason", "constraints"})
        step_id = _step_id(payload, brain_input)
        reason = _text(payload.get("reason"), "reason")
        constraints = payload.get("constraints", [])
        if not isinstance(constraints, list) or any(not isinstance(item, str) for item in constraints):
            raise InvalidBrainOutput("invalid_replan_constraints")
        return BrainOutcome(
            outcome=kind, step_id=step_id, message=reason,
            replan_request=ReplanRequest(reason=reason, failed_step_id=step_id, constraints=tuple(constraints)),
        )
    if kind == Kind.STEP_FAILED:
        _only_fields(payload, {"kind", "step_id", "message"})
        return BrainOutcome(
            outcome=kind, step_id=_step_id(payload, brain_input),
            message=_text(payload.get("message"), "message"), proposed_step_status=StepStatus.FAILED,
        )
    if kind in {Kind.INVALID_OUTPUT, Kind.PROVIDER_FAILURE}:
        _only_fields(payload, {"kind", "message"})
        return BrainOutcome(outcome=kind, message=_text(payload.get("message"), "message"), error_code=kind.value)
    raise InvalidBrainOutput("unsupported_model_outcome")


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type", "text") not in {"text", "output_text"}:
                raise InvalidBrainOutput("unsupported_content_block")
            text = block.get("text", block.get("content", block.get("value")))
            if not isinstance(text, str):
                raise InvalidBrainOutput("invalid_text_block")
            parts.append(text)
        return "\n".join(parts)
    raise InvalidBrainOutput("unsupported_content")


def _text_outcome(
    text: str, brain_input: BrainInput, allowed_tools: set[str], *, allow_text_tool_calls: bool,
    evidence_snapshot: EvidenceSnapshot | None = None,
) -> BrainOutcome:
    stripped = text.strip()
    if not stripped:
        raise InvalidBrainOutput("empty_response")
    candidate = stripped
    if candidate.startswith("```"):
        # A whole fenced envelope is accepted; missing fences and trailing prose
        # are not repaired or silently discarded.
        match = re.fullmatch(r"```(?:json|python)?\s*\n?(.*?)\n?```", candidate, re.DOTALL | re.IGNORECASE)
        if match is None:
            raise InvalidBrainOutput("invalid_fenced_envelope")
        candidate = match.group(1).strip()
    if candidate.startswith(("{", "[")):
        payload = _json(candidate)
        has_contract_fields = isinstance(payload, dict) and bool(
            payload.keys() & {"kind", "tool_calls", "name", "function"}
        )
        if not has_contract_fields and (
            brain_input.direct_response
            or (brain_input.active_plan is not None and brain_input.active_step is None)
        ):
            # A JSON-formatted answer is still answer content in an explicitly
            # authorized answer mode; arbitrary data cannot request lifecycle.
            return BrainOutcome(outcome=Kind.FINAL_ANSWER_READY, final_answer_draft=FinalAnswerDraft(text=text))
        return _structured(payload, brain_input, allowed_tools, allow_text_tool_calls=allow_text_tool_calls, evidence_snapshot=evidence_snapshot)
    # Preserve complete keyword-only function text from non-tool-capable models.
    # Never search within prose or guess arguments from a partial call.
    if re.match(r"^[A-Za-z_]\w*\s*\(", candidate):
        if not allow_text_tool_calls:
            raise InvalidBrainOutput("native_tool_call_required")
        expression = ast.parse(candidate, mode="eval").body
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.args:
            raise InvalidBrainOutput("invalid_function_tool_call")
        args = {}
        for keyword in expression.keywords:
            if keyword.arg is None or keyword.arg in args:
                raise InvalidBrainOutput("ambiguous_function_arguments")
            args[keyword.arg] = ast.literal_eval(keyword.value)
        return _tool_outcome([{"name": expression.func.id, "args": args}], brain_input, allowed_tools)
    if brain_input.direct_response or (brain_input.active_plan is not None and brain_input.active_step is None):
        return BrainOutcome(outcome=Kind.FINAL_ANSWER_READY, final_answer_draft=FinalAnswerDraft(text=text))
    if brain_input.active_plan is None:
        # Preserve Stage 1's missing-plan classification, independently of text.
        return BrainOutcome(outcome=Kind.STEP_FAILED, message=text, error_code="missing_execution_plan")
    raise InvalidBrainOutput("expected_structured_outcome")


def normalize_brain_output(
    raw: object, brain_input: BrainInput, allowed_tools: set[str], *, allow_text_tool_calls: bool = False,
    evidence_snapshot: EvidenceSnapshot | None = None,
) -> BrainOutcome:
    """Normalize once. Text tool compatibility requires explicit provider opt-in."""
    try:
        if isinstance(raw, Mapping) and "parsing_error" in raw:
            if raw["parsing_error"] is not None or raw.get("parsed") is None:
                raise InvalidBrainOutput("structured_output_failure")
            parsed = raw["parsed"]
            if isinstance(parsed, BaseModel):
                parsed = parsed.model_dump(mode="json")
            if _field(raw.get("raw"), "tool_calls"):
                raise InvalidBrainOutput("ambiguous_structured_and_native_output")
            return _structured(parsed, brain_input, allowed_tools, allow_text_tool_calls=allow_text_tool_calls, evidence_snapshot=evidence_snapshot)
        if isinstance(raw, Mapping) and "choices" in raw:
            if len(raw["choices"]) != 1:
                raise InvalidBrainOutput("ambiguous_provider_choices")
            raw = raw["choices"][0]["message"]
        elif isinstance(raw, Mapping) and isinstance(raw.get("message"), Mapping):
            raw = raw["message"]
        if isinstance(raw, Mapping) and any(key in raw for key in ("kind", "name", "function")):
            return _structured(dict(raw), brain_input, allowed_tools, allow_text_tool_calls=allow_text_tool_calls, evidence_snapshot=evidence_snapshot)
        if _field(raw, "invalid_tool_calls"):
            raise InvalidBrainOutput("invalid_native_tool_call")
        native = _field(raw, "tool_calls")
        additional = _field(raw, "additional_kwargs", {}) or {}
        encoded_native = additional.get("tool_calls")
        # LangChain may mirror the original native calls in additional_kwargs.
        # Accept the mirror only if both representations normalize identically.
        if native and encoded_native:
            left = _tool_outcome(native, brain_input, allowed_tools)
            right = _tool_outcome(encoded_native, brain_input, allowed_tools)
            if left.tool_request != right.tool_request:
                raise InvalidBrainOutput("conflicting_native_tool_calls")
        calls = native or encoded_native
        content = raw if isinstance(raw, str) else _field(raw, "content", "")
        if calls:
            text = _content_text("" if content is None else content).strip()
            if text.startswith(("{", "[", "```")) or re.match(r"^[A-Za-z_]\w*\s*\(", text):
                raise InvalidBrainOutput("ambiguous_native_and_structured_output")
            return _tool_outcome(calls, brain_input, allowed_tools)
        return _text_outcome(
            _content_text(content), brain_input, allowed_tools, allow_text_tool_calls=allow_text_tool_calls,
            evidence_snapshot=evidence_snapshot,
        )
    except (ValueError, TypeError, KeyError, AttributeError, SyntaxError, RecursionError) as exc:
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
