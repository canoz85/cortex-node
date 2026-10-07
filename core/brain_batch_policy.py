"""Fail-closed policy for independent, synchronous homogeneous read_file actions."""

import json
import os

from tools.registry import ToolRegistry, get_tool_argument_schema, get_tool_definition

# A complete batch fits within MAX_CURRENT_ATTEMPT_RECORDS, leaving room for
# preceding attempts. Full provenance remains independent of display.
MAX_READ_FILE_BATCH = 24
_BATCH_ELIGIBLE_TOOLS = frozenset({"read_file"})


def normalized_batch_arguments(name, arguments, *, registry: ToolRegistry | None = None):
    definition = get_tool_definition(name)
    if name not in _BATCH_ELIGIBLE_TOOLS or definition is None or definition.mutating:
        raise ValueError("tool_not_batch_eligible")
    schema = get_tool_argument_schema(name, registry=registry)
    if not callable(getattr(schema, "model_validate", None)):
        raise ValueError("batch_argument_schema_unavailable")
    return schema.model_validate(arguments).model_dump(mode="json")


def validate_read_only_batch(candidates, authorized_names, *, registry: ToolRegistry | None = None):
    """Validate the entire ordered group; return default-normalized arguments.

    read_file accepts only literal path/offset/limit values, with no result
    references or async input. Thus each validated invocation is independent.
    Caller supplies the accepted execution's capability ceiling and scope.
    """
    if not 2 <= len(candidates) <= MAX_READ_FILE_BATCH:
        raise ValueError("invalid_batch_size")
    return _normalized_read_only_batch_members(candidates, authorized_names, registry=registry)


def is_oversized_read_only_batch(candidates, authorized_names, *, registry: ToolRegistry | None = None):
    """Correction classification only: size is the group's sole disqualifier.

    Validate every member without relaxing the executable batch size limit.
    This predicate never accepts or authorizes a batch for execution.
    """
    if len(candidates) <= MAX_READ_FILE_BATCH:
        return False
    try:
        _normalized_read_only_batch_members(candidates, authorized_names, registry=registry)
    except (ValueError, TypeError, KeyError):
        return False
    return True


def _normalized_read_only_batch_members(candidates, authorized_names, *, registry):
    names = {name for name, _ in candidates}
    if len(names) != 1 or not names <= set(authorized_names):
        raise ValueError("batch_tool_not_authorized_or_homogeneous")
    normalized = []
    seen = set()
    for name, arguments in candidates:
        effective = normalized_batch_arguments(name, arguments, registry=registry)
        identity_args = dict(effective)
        identity_args["path"] = os.path.normcase(os.path.normpath(effective["path"]))
        identity = json.dumps(identity_args, sort_keys=True, allow_nan=False)
        if identity in seen:
            raise ValueError("duplicate_effective_batch_invocation")
        seen.add(identity)
        normalized.append(effective)
    return tuple(normalized)
