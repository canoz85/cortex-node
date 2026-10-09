import pytest
from core.planner import PLANNER_SYSTEM_PROMPT
from core.planner_contract import PlannerProposal, PlannerProposalResultType, ProposedStep


def test_all_semantic_variants_validate_without_failure_category():
    step = ProposedStep(
        step_id="step1",
        title="Inspect files",
        description="Inspect the workspace files.",
        primary_tool="list_files",
    )

    assert PlannerProposal(
        result=PlannerProposalResultType.PLAN_PROPOSED,
        steps=(step,),
    ).steps == (step,)
    assert PlannerProposal(
        result=PlannerProposalResultType.NO_PLAN_REQUIRED,
        message="The direct answer.",
    ).steps == ()
    assert PlannerProposal(
        result=PlannerProposalResultType.NEEDS_INPUT,
        message="Which workspace should I inspect?",
    ).steps == ()
    assert PlannerProposal(
        result=PlannerProposalResultType.PLANNING_FAILED,
        message="No available runtime capability can perform the request.",
    ).steps == ()


@pytest.mark.parametrize(
    "result",
    [
        PlannerProposalResultType.NO_PLAN_REQUIRED,
        PlannerProposalResultType.NEEDS_INPUT,
        PlannerProposalResultType.PLANNING_FAILED,
    ],
)
def test_message_bearing_variants_reject_empty_messages(result):
    with pytest.raises(ValueError, match="requires a non-empty message"):
        PlannerProposal(result=result, message="   ")


def test_proposal_extra_fields_are_forbidden():
    with pytest.raises(ValueError, match="failure_category"):
        PlannerProposal.model_validate({
            "result": "PLANNING_FAILED",
            "message": "The request cannot be planned with available tools.",
            "failure_category": "UNPLANNABLE",
        })


def test_system_policy_exposes_only_semantic_proposal_outcomes():
    prompt = PLANNER_SYSTEM_PROMPT.format(
        route="info",
        available_capabilities='[{"name":"list_files"}]',
        capability_guidance="",
    )

    assert "Canonical shape:" not in prompt
    for result in PlannerProposalResultType:
        assert f"- {result.value}:" in prompt
    assert "bound schema" in prompt
    assert "message contains the direct answer" in prompt
    assert "message contains the concrete question" in prompt
    assert "Provider/invalid-output failures are handled externally" in prompt
    assert "For the three non-plan outcomes, steps must be empty" in prompt


@pytest.mark.parametrize("payload", [
    {"result": "PLAN_PROPOSED", "steps": []},
    {"result": "NEEDS_INPUT", "message": "Which file?", "steps": [{"step_id": "s",
        "title": "Read", "description": "Read the file", "primary_tool": "read_file"}]},
    {"result": "INVALID_OUTPUT", "message": "Bad output"},
])
def test_invalid_proposal_variant_shapes_are_rejected(payload):
    with pytest.raises(ValueError):
        PlannerProposal.model_validate(payload)


@pytest.mark.parametrize("tool", [None, ""])
def test_executable_step_requires_a_primary_tool(tool):
    with pytest.raises(ValueError):
        ProposedStep(step_id="s", title="Read", description="Read the file", primary_tool=tool)


def test_executable_step_rejects_missing_tool_and_extra_arguments():
    payload = {"step_id": "s", "title": "Read", "description": "Read the file"}
    with pytest.raises(ValueError):
        ProposedStep.model_validate(payload)
    with pytest.raises(ValueError, match="extra"):
        ProposedStep.model_validate({**payload, "primary_tool": "read_file", "arguments": {"path": "a"}})
