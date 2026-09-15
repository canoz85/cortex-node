"""Framework-neutral semantic wake contracts for asynchronous execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AsyncWakeIntent(str, Enum):
    """Why an external scheduler is waking a suspended execution."""

    POLL_DUE = "poll_due"


@dataclass(frozen=True, slots=True)
class AsyncExecutionWake:
    """Correlate a semantic async wake without carrying runtime authorization."""

    execution_id: str
    async_job_id: str
    intent: AsyncWakeIntent = AsyncWakeIntent.POLL_DUE

    def __post_init__(self) -> None:
        if not self.execution_id:
            raise ValueError("Async wake requires execution_id.")
        if not self.async_job_id:
            raise ValueError("Async wake requires async_job_id.")


__all__ = ["AsyncExecutionWake", "AsyncWakeIntent"]
