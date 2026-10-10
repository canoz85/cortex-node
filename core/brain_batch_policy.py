"""Shared validation for explicitly enabled, independent synchronous batches."""

import json
from collections import Counter
from typing import Callable

from tools.registry import get_tool_argument_schema, get_tool_definition


MAX_BATCH_CALLS = 24


def _batch_execution_metadata(name, argument_schema_for):
    definition = get_tool_definition(name)
    if definition is None or definition.mutating or definition.max_batch_calls <= 1:
        raise ValueError("tool_not_batch_eligible")
    lookup = get_tool_argument_schema if argument_schema_for is None else argument_schema_for
    schema = lookup(name)
    if not callable(getattr(schema, "model_validate", None)):
        raise ValueError("batch_argument_schema_unavailable")
    return definition.max_batch_calls, schema


def batch_call_limit(name, *, argument_schema_for: Callable[[str], type | None] | None = None) -> int:
    """Read an eligible tool's limit; metadata and schemas never grant authority."""
    return _batch_execution_metadata(name, argument_schema_for)[0]


def normalized_batch_arguments(
    name, arguments, *, argument_schema_for: Callable[[str], type | None] | None = None,
):
    _, schema = _batch_execution_metadata(name, argument_schema_for)
    return schema.model_validate(arguments).model_dump(mode="json")


def validate_read_only_batch(
    candidates, authorized_names, *, argument_schema_for: Callable[[str], type | None] | None = None,
):
    """Validate the entire ordered group; return default-normalized arguments.

    Explicit metadata opts tools into independent synchronous execution.
    Caller supplies the accepted execution's capability ceiling and schemas.
    Returned arguments are for validation, never replacements for requests.
    """
    if not isinstance(candidates, (list, tuple)) or len(candidates) < 2:
        raise ValueError("invalid_batch_size")
    normalized = _normalized_read_only_batch_members(
        candidates, authorized_names, argument_schema_for=argument_schema_for,
    )
    if _exceeds_batch_limits(candidates, argument_schema_for=argument_schema_for):
        raise ValueError("invalid_batch_size")
    return normalized


def is_oversized_read_only_batch(
    candidates, authorized_names, *, argument_schema_for: Callable[[str], type | None] | None = None,
):
    """Correction classification only: size is the group's sole disqualifier.

    Validate every member without relaxing the executable batch size limit.
    This predicate never accepts or authorizes a batch for execution.
    """
    if not isinstance(candidates, (list, tuple)) or len(candidates) < 2:
        return False
    try:
        _normalized_read_only_batch_members(
            candidates, authorized_names, argument_schema_for=argument_schema_for,
        )
        return _exceeds_batch_limits(candidates, argument_schema_for=argument_schema_for)
    except (ValueError, TypeError, KeyError):
        return False


def _exceeds_batch_limits(candidates, *, argument_schema_for):
    return len(candidates) > MAX_BATCH_CALLS or any(
        count > batch_call_limit(name, argument_schema_for=argument_schema_for)
        for name, count in Counter(name for name, _ in candidates).items()
    )


def _normalized_read_only_batch_members(candidates, authorized_names, *, argument_schema_for):
    if any(not isinstance(name, str) or not isinstance(arguments, dict)
           for name, arguments in candidates):
        raise ValueError("malformed_batch_invocation")
    names = {name for name, _ in candidates}
    if not names <= set(authorized_names):
        raise ValueError("batch_tool_not_authorized")
    normalized = []
    seen = set()
    for name, arguments in candidates:
        effective = normalized_batch_arguments(name, arguments, argument_schema_for=argument_schema_for)
        identity = (name, json.dumps(effective, sort_keys=True, allow_nan=False))
        if identity in seen:
            raise ValueError("duplicate_effective_batch_invocation")
        seen.add(identity)
        normalized.append(effective)
    return tuple(normalized)
