"""Provider-facing structured Planner proposal contract."""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from core.protocol.models import MAX_PLANNER_DIRECT_RESPONSE_CHARS


class PlannerProposalResultType(str, Enum):
    PLAN_PROPOSED = "PLAN_PROPOSED"
    NO_PLAN_REQUIRED = "NO_PLAN_REQUIRED"
    NEEDS_INPUT = "NEEDS_INPUT"
    PLANNING_FAILED = "PLANNING_FAILED"


class ProposedStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    step_id: str = Field(min_length=1)
    title: str = Field(
        min_length=1,
        description="Controller-accepted executable step scope. Include a known resolved value here or in description when execution requires it; do not leave it only in the plan objective or Planner-only context.",
    )
    description: str = Field(
        min_length=1,
        description="What this step must accomplish, including any already-known resolved context required by the worker. This becomes Controller-accepted active-step semantics, not tool arguments or a prompt.",
    )

    primary_tool: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()


class PlannerProposal(BaseModel):
    """One strict schema envelope with four explicit result variants."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    result: PlannerProposalResultType
    objective: str = ""
    steps: tuple[ProposedStep, ...] = ()
    message: str = Field(
        default="",
        description=(
            "For NEEDS_INPUT, the concrete non-empty question to ask the user. "
            f"For NO_PLAN_REQUIRED, the direct answer, at most {MAX_PLANNER_DIRECT_RESPONSE_CHARS} characters. "
            "For other result variants, optional explanatory text."
        ),
    )
    @model_validator(mode="after")
    def variant_shape(self):
        if self.result == PlannerProposalResultType.PLAN_PROPOSED:
            if not self.steps:
                raise ValueError("PLAN_PROPOSED requires at least one step")
        elif self.steps:
            raise ValueError(f"{self.result.value} cannot contain steps")
        if self.result in {
            PlannerProposalResultType.NO_PLAN_REQUIRED,
            PlannerProposalResultType.NEEDS_INPUT,
            PlannerProposalResultType.PLANNING_FAILED,
        } and not self.message.strip():
            raise ValueError(f"{self.result.value} requires a non-empty message")
        if self.result == PlannerProposalResultType.NO_PLAN_REQUIRED:
            # The variant's answer has the same bound as the downstream result.
            if len(self.message.strip()) > MAX_PLANNER_DIRECT_RESPONSE_CHARS:
                raise ValueError(f"NO_PLAN_REQUIRED message exceeds the {MAX_PLANNER_DIRECT_RESPONSE_CHARS}-character limit")
        return self


class PlannerInvalidOutputError(ValueError):
    """Provider data did not satisfy PlannerProposal."""


def authorized_planner_schema(
    available_tools: tuple[str, ...], *, max_steps: int,
) -> type[PlannerProposal]:
    """Restrict the provider contract to this request's authorized capability set."""
    names = tuple(sorted(set(available_tools)))
    if names:
        step = create_model(
            "AuthorizedProposedStep", __base__=ProposedStep,
            primary_tool=(Literal.__getitem__(names), Field(...)),
        )
        steps_type = tuple[step, ...]
        steps_field = Field(default=(), max_length=max_steps)
    else:
        # No executable step is possible; the other result variants remain valid.
        steps_type = tuple[()]
        steps_field = Field(default=())
    return create_model(
        "AuthorizedPlannerProposal", __base__=PlannerProposal,
        steps=(steps_type, steps_field),
    )
