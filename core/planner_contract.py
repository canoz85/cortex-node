"""Provider-facing structured Planner proposal contract."""

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannerProposalResultType(str, Enum):
    PLAN_PROPOSED = "PLAN_PROPOSED"
    NO_PLAN_REQUIRED = "NO_PLAN_REQUIRED"
    NEEDS_INPUT = "NEEDS_INPUT"
    PLANNING_FAILED = "PLANNING_FAILED"


class ProposalFailureCategory(str, Enum):
    INVALID_OUTPUT = "INVALID_OUTPUT"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    UNPLANNABLE = "UNPLANNABLE"


class ProposedStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    step_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    primary_tool: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()


class PlannerProposal(BaseModel):
    """One strict schema envelope with four explicit result variants."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    result: PlannerProposalResultType
    objective: str = ""
    steps: tuple[ProposedStep, ...] = ()
    message: str = ""
    failure_category: ProposalFailureCategory | None = None

    @model_validator(mode="after")
    def variant_shape(self):
        if self.result == PlannerProposalResultType.PLAN_PROPOSED:
            if self.failure_category is not None:
                raise ValueError("PLAN_PROPOSED cannot contain failure_category")
        elif self.steps:
            raise ValueError(f"{self.result.value} cannot contain steps")
        if self.result == PlannerProposalResultType.PLANNING_FAILED:
            if self.failure_category is None:
                raise ValueError("PLANNING_FAILED requires failure_category")
        elif self.failure_category is not None:
            raise ValueError(f"{self.result.value} cannot contain failure_category")
        return self


class PlannerInvalidOutputError(ValueError):
    """Provider data did not satisfy PlannerProposal."""
