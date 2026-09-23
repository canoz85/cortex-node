from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    action: Literal["created", "modified", "deleted"]