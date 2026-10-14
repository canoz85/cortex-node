"""Deterministic rejection of already successful step-local tool invocations."""

import json

from tools.registry import get_tool_argument_schema


DUPLICATE_SUCCESSFUL_CALL = "duplicate_successful_call"


class DuplicateSuccessfulCall(ValueError):
    pass


def _argument_identity(name, arguments, argument_schema_for):
    lookup = argument_schema_for or get_tool_argument_schema
    schema = lookup(name)
    effective = arguments
    if callable(getattr(schema, "model_validate", None)):
        try:
            validated = schema.model_validate(arguments).model_dump(mode="json")
        except (ValueError, TypeError):
            # Malformed/legacy arguments retain exact native comparison; this
            # guard never relaxes or replaces executable argument validation.
            pass
        else:
            # Same effective representation used by batch validation and the
            # executable schema, including defaults/coercions. Requests stay intact.
            effective = validated
    return json.dumps(effective, sort_keys=True, allow_nan=False)


def reject_duplicate_successful_calls(candidates, context, *, argument_schema_for=None):
    """Reject the whole proposed action before any member is authorized.

    Scope excludes attempts: a retry cannot erase already successful evidence.
    All continuation arguments participate in identity, so changing an offset,
    cursor, token or range remains allowed without tool-name-specific rules.
    An unfinished async observation is still pollable under existing lifecycle
    policy; a terminal successful observation is ordinary completed evidence.
    """
    plan, step = context.active_plan, context.active_step
    if plan is None or step is None:
        return
    scope = (context.identity.execution_id, plan.plan_id, plan.revision, step.step_id)
    records = [record for record in context.tool_execution_history
               if (record.execution_id, record.plan_id, record.plan_revision, record.step_id) == scope
               and record.result.success
               and not (record.result.is_async_job and not record.result.async_terminal)]
    for name, arguments in candidates:
        matching = [record for record in records if record.tool_name == name]
        if not matching:
            continue
        identity = _argument_identity(name, arguments, argument_schema_for)
        if any(_argument_identity(name, record.arguments, argument_schema_for) == identity
               for record in matching):
            raise DuplicateSuccessfulCall(
                f"{DUPLICATE_SUCCESSFUL_CALL}: The identical {name} call already succeeded "
                "on this active step, including any empty result. Reuse the prior evidence. "
                "Choose a different tool/query/arguments, complete the step, or request replan. "
                "The entire proposed action was rejected; no tool was executed."
            )
