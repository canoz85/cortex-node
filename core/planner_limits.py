"""Planner generation policy, independent of other model roles."""

# Includes hidden reasoning: leave headroom above the observed 1,148-token plan.
MAX_PLANNER_GENERATION_TOKENS = 4096

PLANNER_LENGTH_DIAGNOSTIC = (
    "Planner response reached the permitted generation limit before completing "
    "the required structured proposal. Return the required proposal directly and concisely."
)
