from types import SimpleNamespace

import pytest

from core.planner import PLANNER_SYSTEM_PROMPT
from core.planner_contract import PlannerProposal, PlannerProposalResultType, ProposedStep
from core.planner_provider import _extract_planner_proposal


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


def test_removed_failure_category_is_forbidden_extra_input():
    with pytest.raises(ValueError, match="failure_category"):
        PlannerProposal.model_validate({
            "result": "PLANNING_FAILED",
            "message": "The request cannot be planned with available tools.",
            "failure_category": "UNPLANNABLE",
        })


def test_raw_schema_valid_json_fallback_remains_supported():
    exchange = {
        "raw": SimpleNamespace(content=(
            '{"result":"NO_PLAN_REQUIRED","objective":"","steps":[],'
            '"message":"The direct answer."}'
        )),
        "parsed": None,
        "parsing_error": ValueError("native parser failed"),
    }

    proposal = _extract_planner_proposal(exchange)

    assert proposal.result == PlannerProposalResultType.NO_PLAN_REQUIRED
    assert proposal.message == "The direct answer."


def test_result_contract_includes_every_canonical_shape_and_ownership_rule():
    prompt = PLANNER_SYSTEM_PROMPT.format(
        route="info",
        available_tools="- list_files",
        capability_guidance="",
    )

    assert prompt.count("Canonical shape:") == 4
    assert '"result": "PLAN_PROPOSED"' in prompt
    assert '"result": "NO_PLAN_REQUIRED"' in prompt
    assert '"result": "NEEDS_INPUT"' in prompt
    assert '"result": "PLANNING_FAILED"' in prompt
    assert "Provider failures and invalid model output are not PLANNING_FAILED" in prompt
    assert "Do not add fields outside the schema." in prompt
