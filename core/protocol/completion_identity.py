"""Domain-neutral identities used to reject stale completion assessments."""

import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def requirement_scope(identity, plan, step):
    return fingerprint({
        "execution": identity.execution_id, "plan": plan.plan_id,
        "revision": plan.revision, "step": step.step_id,
        "requirement": step.completion_requirement.model_dump(mode="json"),
    })


def evidence_identity(records):
    return fingerprint([record.model_dump(mode="json") for record in records])


def plan_validation_identity(plan):
    return fingerprint(plan.model_dump(mode="json"))


def eligible_records(identity, plan, records):
    return tuple(r for r in records if r.execution_id == identity.execution_id
                 and r.plan_id == plan.plan_id and r.plan_revision == plan.revision)


def accepted_step(plan, step, cursor=None):
    if plan is None:
        raise ValueError("active step requires accepted plan")
    matches = [s for s in plan.steps if s.step_id == step.step_id]
    if len(matches) != 1 or matches[0] != step:
        raise ValueError("active step disagrees with accepted plan")
    if cursor is not None and (cursor.step_id != step.step_id or
            (cursor.plan_revision is not None and cursor.plan_revision != plan.revision)):
        raise ValueError("active step revision/cursor mismatch")
    return matches[0]


def binding_for(identity, plan, step, bindings):
    matches = [b for b in bindings if (b.execution_id, b.plan_id, b.plan_revision, b.step_id)
               == (identity.execution_id, plan.plan_id, plan.revision, step.step_id)]
    return matches[0] if len(matches) == 1 else None
