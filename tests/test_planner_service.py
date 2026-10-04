"""Planner service, normalization and authorized context tests."""

import json
import os
import pytest
import subprocess
import sys
from core.planner import AmbientRetrievalEligibility, PlannerService, ambient_retrieval_eligibility
from core.planner_contract import PlannerInvalidOutputError
from core.planner_normalization import normalize_planner_proposal
from core.protocol.enums import (
    PlannerOutcome,
    PlanningFailureCategory,
    PlanningOperation,
    ReplanTrigger,
    StepStatus,
    WorkerRole,
)
from core.protocol.models import (
    ExecutionContext,
    ExecutionIdentity,
    ExecutionPlan,
    ExecutionStep,
    PlannerResult,
    PlanningCapabilities,
    PlanningRequest,
    RetryMetadata,
)
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


VALID = {
    "result": "PLAN_PROPOSED",
    "objective": "Inspect then write",
    "steps": [
        {
            "step_id": "inspect",
            "title": "Inspect",
            "description": "Inspect workspace",
            "primary_tool": "list_files",
            "dependencies": [],
        },
        {
            "step_id": "write",
            "title": "Write",
            "description": "Create file",
            "primary_tool": "write_file",
            "dependencies": ["inspect"],
        },
    ],
}


def planner_input(**updates):
    plan = updates.pop("active_plan", None)
    request_context = updates.pop("context", None) or ExecutionContext(
        user_request="create a file",
        role=WorkerRole.PLANNER,
    )

    return PlanningRequest(
        request_id="fixture-request",
        episode_id="fixture-episode",
        sequence=1,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation=(
            PlanningOperation.REVISE
            if plan
            else PlanningOperation.CREATE
        ),
        base_plan=plan,
        base_plan_id=plan.plan_id if plan else None,
        base_revision=plan.revision if plan else None,
        trigger=ReplanTrigger.BRAIN_REQUESTED if plan else None,
        reason="fixture revision" if plan else "",
        
        capabilities=PlanningCapabilities(
            available_tools=(
                "list_files",
                "read_file",
                "write_file",
                "current_time",
                "agent_info",
                "token_usage",
            ),
            unavailable_tools=("blocked_tool",),
        ),
        identity=ExecutionIdentity(
            execution_id="p3",
            protocol_version="1",
        ),
        context=request_context,
        **updates,
    )


class FakeRouter:
    def __init__(
        self,
        route: str = "action",
        *,
        error: Exception | None = None,
    ):
        self.route_value = route
        self.error = error
        self.calls = []

    def route(self, user_request: str):
        self.calls.append(user_request)
        if self.error is not None:
            raise self.error

        return SimpleNamespace(route=self.route_value)


class FakeProvider:
    def __init__(
        self,
        content=VALID,
        *,
        error_at: str | None = None,
    ):
        self.content = content
        self.error_at = error_at
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)

        if self.error_at == "generate":
            raise RuntimeError("generate failed")

        if self.error_at == "invalid":
            raise PlannerInvalidOutputError("invalid output")

        return self.content


def service(
    provider,
    *,
    route: str = "action",
    route_error: Exception | None = None,
):
    return PlannerService(
        provider=provider,
        router=FakeRouter(
            route=route,
            error=route_error,
        ),
        mutating_tools={"write_file"},
    )


def test_valid_dependent_plan():
    provider = FakeProvider()

    result = service(provider).run(
        planner_input(),
        retrieve=lambda _: ("retrieved",),
    )

    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[1].depends_on_step_ids == ("inspect",)
    assert result.proposed_plan.steps[0].primary_tool == "list_files"
    assert [
        step.primary_tool
        for step in result.proposed_plan.steps
    ] == ["list_files", "write_file"]
    assert "Add prerequisite inspection" in provider.messages[0][0].content


def test_preserved_clarification_route_skips_router_and_plans_original_request():
    provider = FakeProvider({
        "result": "PLAN_PROPOSED",
        "objective": "Read and summarize README.md",
        "steps": [{
            "step_id": "read",
            "title": "Read and summarize README.md",
            "description": "Read README.md and produce the requested summary.",
            "primary_tool": "read_file",
            "dependencies": [],
        }],
    })
    router = FakeRouter(error=AssertionError("clarification must not be rerouted"))
    planner = PlannerService(
        provider=provider,
        router=router,
        mutating_tools={"write_file"},
    )
    request = planner_input(
        planner_route="info",
        context=ExecutionContext(
            user_request="Workspace içindeki şu dosyayı oku ve özetle.",
            clarification_question="Hangi dosyayı okumalıyım?",
            clarification="Readme.md",
            role=WorkerRole.PLANNER,
        ),
    )

    result = planner.run(request)

    assert router.calls == []
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.planner_route == "info"
    assert result.proposed_plan.steps[0].primary_tool == "read_file"
    assert provider.messages[0][-1].content == request.context.user_request
    structured_context = provider.messages[0][-2].content
    context_payload = json.loads(structured_context.split("\n", 1)[1])["context"]
    assert "user_request" not in context_payload
    assert context_payload["clarification_question"] == "Hangi dosyayı okumalıyım?"
    assert context_payload["clarification"] == "Readme.md"
    assert "human message is the original Controller-authorized request" in structured_context
    assert "context.clarification_question is the Planner's previous NEEDS_INPUT" in structured_context
    assert "context.clarification is the user's answer to that question" in structured_context
    assert "Interpret the human request, clarification_question, and clarification together" in structured_context
    assert "do not ask again for information it supplies" in structured_context
    assert "does not authorize unrelated or expanded work" in structured_context
    assert "route, capabilities, operation, and Controller constraints" in structured_context
    assert "only if the combined original request and clarification are still insufficient" in structured_context


def test_new_request_and_existing_replan_still_route_normally():
    router = FakeRouter(route="action")
    planner = PlannerService(
        provider=FakeProvider(), router=router, mutating_tools={"write_file"},
    )

    planner.run(planner_input())
    base = ExecutionPlan(
        plan_id="accepted",
        revision=1,
        steps=(ExecutionStep(step_id="old", title="Old"),),
    )
    planner.run(planner_input(active_plan=base))

    assert router.calls == ["create a file", "create a file"]
    assert [call[-1].content for call in planner.provider.messages] == [
        "create a file",
        "create a file",
    ]


def _planner_facing_tools(provider: FakeProvider) -> set[str]:
    messages = provider.messages[0]
    prompt_section = messages[0].content.split(
        "AVAILABLE TOOLS FOR THIS REQUEST", 1
    )[1].split("STEP SEMANTICS:", 1)[0]
    prompt_tools = {
        line.removeprefix("- ").strip()
        for line in prompt_section.splitlines()
        if line.startswith("- ") and "No tool access" not in line
    }
    context = json.loads(messages[-2].content.split("\n", 1)[1])
    assert "capabilities" not in context
    return prompt_tools


def test_info_planner_prompt_and_validation_use_one_authorized_tool_set():
    provider = FakeProvider({
        "result": "PLAN_PROPOSED",
        "objective": "Inspect",
        "steps": [{
            "step_id": "inspect",
            "title": "Inspect files",
            "description": "Inspect workspace files",
            "primary_tool": "list_files",
            "dependencies": [],
        }],
    })

    request = planner_input()
    result = service(provider, route="info").run(request)

    prompt_tools = _planner_facing_tools(provider)
    assert "write_file" not in prompt_tools
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.available_tools == tuple(sorted(prompt_tools))
    assert request.capabilities.available_tools == planner_input().capabilities.available_tools


def test_info_planner_cannot_validate_tool_outside_authorized_set():
    provider = FakeProvider({
        "result": "PLAN_PROPOSED",
        "objective": "Write",
        "steps": [{
            "step_id": "write",
            "title": "Write a file",
            "description": "Create the requested file",
            "primary_tool": "write_file",
            "dependencies": [],
        }],
    })

    result = service(provider, route="info").run(planner_input())

    prompt_tools = _planner_facing_tools(provider)
    assert "write_file" not in prompt_tools
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.INVALID_OUTPUT
    assert "primary_tool 'write_file' is unknown" in result.message


def test_empty_authorized_set_is_consistent_and_can_return_unplannable():
    provider = FakeProvider({
        "result": "PLANNING_FAILED",
        "message": "No authorized capability can satisfy the request.",
    })
    request = planner_input().model_copy(update={
        "capabilities": PlanningCapabilities(available_tools=("write_file",)),
    })

    result = service(provider, route="info").run(request)

    assert _planner_facing_tools(provider) == set()
    assert result.outcome == PlannerOutcome.FAILED
    assert result.failure_category == PlanningFailureCategory.UNPLANNABLE


def test_knowledge_request_and_uncertain_request_remain_ambient_rag_eligible():
    cases = (
        (
            "inspect the CortexNode checkpoint architecture",
            "info",
        ),
        (
            "investigate the project behavior",
            "action",
        ),
        (
            "list files and explain the checkpoint architecture",
            "info",
        ),
    )

    for user_request, route in cases:
        provider = FakeProvider(VALID)

        request = planner_input().model_copy(
            update={
                "context": ExecutionContext(
                    user_request=user_request,
                    role=WorkerRole.PLANNER,
                )
            }
        )

        retrieval_calls = []

        service(
            provider,
            route=route,
        ).run(
            request,
            retrieve=lambda query: (
                retrieval_calls.append(query)
                or ("architecture context",)
            ),
        )

        assert retrieval_calls == [user_request]
        assert (
            provider.messages[0][1].content
            == "architecture context"
        )


def test_runtime_intent_without_matching_authorized_capability_keeps_retrieval():
    request = planner_input().model_copy(
        update={
            "context": ExecutionContext(
                user_request="git status",
                role=WorkerRole.PLANNER,
            ),
            "capabilities": PlanningCapabilities(
                available_tools=("rag_search",),
            ),
        }
    )

    assert (
        ambient_retrieval_eligibility(
            request,
            route="info",
        )
        == AmbientRetrievalEligibility.KNOWLEDGE
    )


def test_revise_preserves_direct_route_and_skips_ambient_retrieval():
    base = ExecutionPlan(
        plan_id="accepted",
        revision=3,
        steps=(
            ExecutionStep(
                step_id="prior",
                title="Prior",
                status=StepStatus.COMPLETED,
            ),
        ),
    )

    request = planner_input(
        active_plan=base,
    ).model_copy(
        update={
            "context": ExecutionContext(
                user_request=(
                    "inspect the CortexNode checkpoint architecture"
                ),
                role=WorkerRole.PLANNER,
            )
        }
    )

    provider = FakeProvider(VALID)
    retrieval_calls = []

    service(
        provider,
        route="conversation",
    ).run(
        request,
        retrieve=lambda query: (
            retrieval_calls.append(query)
            or ("revision knowledge",)
        ),
    )

    assert retrieval_calls == []


@pytest.mark.parametrize(
    "route",
    ["conversation", "clarify"],
)
def test_direct_routes_remain_ambient_rag_ineligible(route):
    assert (
        ambient_retrieval_eligibility(
            planner_input(),
            route=route,
        )
        == AmbientRetrievalEligibility.NONE
    )


def test_planner_context_includes_retrieved_background_once():
    provider = FakeProvider()
    service(provider).run(planner_input(), retrieve=lambda _: ("background",))
    messages = provider.messages[0]
    assert sum(m.content == "background" for m in messages) == 1
    assert "retrieval_messages" not in json.loads(messages[-2].content.split("\n", 1)[1])["context"]


def test_valid_independent_steps():
    value = {
        **VALID,
        "steps": [
            {
                **VALID["steps"][0],
            },
            {
                **VALID["steps"][1],
                "dependencies": [],
            },
        ],
    }

    result = normalize_planner_proposal(
        value,
        planner_input(),
        route="action",
    )

    assert [
        step.depends_on_step_ids
        for step in result.proposed_plan.steps
    ] == [(), ()]


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (
            lambda value: value["steps"].__setitem__(
                1,
                {
                    **value["steps"][1],
                    "step_id": "inspect",
                },
            ),
            "unique",
        ),
        (
            lambda value: value["steps"][1].update(
                dependencies=["missing"]
            ),
            "unknown dependency",
        ),
        (
            lambda value: (
                value["steps"][0].update(
                    dependencies=["write"]
                ),
                value["steps"][1].update(
                    dependencies=["inspect"]
                ),
            ),
            "cyclic",
        ),
        (
            lambda value: value["steps"][0].update(
                primary_tool="blocked_tool"
            ),
            "unavailable",
        ),
        (
            lambda value: value["steps"][0].update(
                primary_tool="invented"
            ),
            "unknown",
        ),
        (
            lambda value: value["steps"][0].update(primary_tool="   "),
            "primary_tool",
        ),
    ],
)
def test_invalid_plan_constraints_rejected(
    mutate,
    needle,
):
    import copy

    value = copy.deepcopy(VALID)
    mutate(value)

    result = normalize_planner_proposal(
        value,
        planner_input(),
        route="action",
    )

    assert result.outcome == PlannerOutcome.FAILED
    assert (
        result.failure_category
        == PlanningFailureCategory.INVALID_OUTPUT
    )
    assert needle in result.message


def test_malformed_structured_output_is_invalid():
    content = "numbered prose"
    result = service(
        FakeProvider(content)
    ).run(
        planner_input()
    )

    assert (
        result.failure_category
        == PlanningFailureCategory.INVALID_OUTPUT
    )


@pytest.mark.parametrize(
    "payload,outcome",
    [
        (
            {
                "result": "NO_PLAN_REQUIRED",
                "message": "direct",
            },
            PlannerOutcome.DIRECT_RESPONSE,
        ),
        (
            {
                "result": "NEEDS_INPUT",
                "message": "need path",
            },
            PlannerOutcome.CLARIFICATION_REQUIRED,
        ),
        (
            {
                "result": "PLANNING_FAILED",
                "message": "impossible",
            },
            PlannerOutcome.FAILED,
        ),
    ],
)
def test_explicit_result_variants(
    payload,
    outcome,
):
    result = service(
        FakeProvider(payload),
        route="conversation",
    ).run(
        planner_input()
    )

    assert result.outcome == outcome

    if outcome == PlannerOutcome.FAILED:
        assert (
            result.failure_category
            == PlanningFailureCategory.UNPLANNABLE
        )


def test_revise_preserves_request_and_versions_candidate():
    base = ExecutionPlan(
        plan_id="accepted",
        revision=3,
        steps=(
            ExecutionStep(
                step_id="done",
                title="Done",
                status=StepStatus.COMPLETED,
            ),
        ),
    )

    request = planner_input(
        active_plan=base,
        completed_step_ids=("done",),
        completed_steps=base.steps,
        retry=RetryMetadata(
            step_id="failed",
            retry_count=1,
            max_retries=2,
        ),
    )

    before = request.model_dump_json()
    provider = FakeProvider()

    result = service(provider).run(request)

    assert request.model_dump_json() == before
    assert (
        result.proposed_plan.plan_id,
        result.proposed_plan.revision,
    ) == (
        "accepted",
        4,
    )

    context = provider.messages[0][-2].content

    payload = json.loads(context.split("\n", 1)[1])
    assert payload["base_plan"]["revision"] == 3
    assert payload["base_plan"]["steps"][0]["step_id"] == "done"


def test_planner_results_are_bound_and_outcome_payloads_are_strict():
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match="request_id",
    ):
        PlannerResult(
            outcome=PlannerOutcome.DIRECT_RESPONSE
        )

    with pytest.raises(
        ValidationError,
        match="requires a plan",
    ):
        PlannerResult(
            outcome=PlannerOutcome.EXECUTION_PLAN,
            request_id="request",
        )

    with pytest.raises(
        ValidationError,
        match="failure category",
    ):
        PlannerResult(
            outcome=PlannerOutcome.FAILED,
            request_id="request",
        )


def test_planning_request_requires_explicit_episode_identity():
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match="episode_id",
    ):
        PlanningRequest(
            request_id="request",
            sequence=1,
            created_at_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            operation=PlanningOperation.CREATE,
            capabilities=PlanningCapabilities(),
            identity=ExecutionIdentity(
                execution_id="p6",
                protocol_version="1",
            ),
            context=ExecutionContext(
                user_request="work",
                role=WorkerRole.PLANNER,
            ),
        )


def test_comfy_guidance_uses_action_route_and_authorized_capability_only():
    capabilities = PlanningCapabilities(
        available_tools=(
            "list_files",
            "run_comfy_workflow",
            "get_comfy_history",
            "download_comfy_output_image",
        )
    )

    ordinary = FakeProvider()

    ordinary_request = planner_input().model_copy(
        update={
            "capabilities": capabilities,
            "context": ExecutionContext(
                user_request="Generate a cat image and save it.",
                role=WorkerRole.PLANNER,
            ),
        }
    )

    service(
        ordinary,
        route="info",
    ).run(
        ordinary_request
    )

    assert (
        "CAPABILITY-SPECIFIC GUIDANCE"
        not in ordinary.messages[0][0].content
    )

    image = FakeProvider()

    image_request = planner_input().model_copy(
        update={
            "capabilities": capabilities,
            "context": ExecutionContext(
                user_request="Generate a cat image and save it.",
                role=WorkerRole.PLANNER,
            ),
        }
    )

    service(
        image,
        route="action",
    ).run(
        image_request
    )

    prompt = image.messages[0][0].content

    assert (
        "CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION"
        in prompt
    )
    assert "run_comfy_workflow" in prompt

    fragment = prompt.split(
        "CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION:",
        1,
    )[1]

    submission = fragment.index(
        "run_comfy_workflow"
    )
    history = fragment.index(
        "get_comfy_history",
        submission,
    )
    download = fragment.index(
        "download_comfy_output_image",
        history,
    )

    assert submission < history < download

    unrelated_action = FakeProvider()

    unrelated_request = planner_input().model_copy(
        update={
            "capabilities": capabilities,
            "context": ExecutionContext(
                user_request="perform the authorized action",
                role=WorkerRole.PLANNER,
            ),
        }
    )

    service(
        unrelated_action,
        route="action",
    ).run(
        unrelated_request
    )

    assert (
        "CAPABILITY-SPECIFIC GUIDANCE — COMFYUI GENERATION"
        in unrelated_action.messages[0][0].content
    )


def test_service_runs_with_framework_imports_blocked():
    script = '''
import importlib.abc
import sys

sys.path.insert(0, "tests")


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith((
            "langchain",
            "langgraph",
            "ollama",
            "core.graph",
            "core.planner_provider",
            "core.planner_routing",
        )):
            raise AssertionError(fullname)


sys.meta_path.insert(0, Block())

from test_planner_service import FakeProvider, planner_input, service
from core.protocol.enums import PlannerOutcome

assert (
    service(FakeProvider())
    .run(planner_input())
    .outcome
    == PlannerOutcome.EXECUTION_PLAN
)
'''

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )

    assert (
        completed.returncode == 0
    ), completed.stdout + completed.stderr


@pytest.mark.parametrize("user_request,tool_name", [
    ("list files", "list_files"),
    ("current git status", "git_status"),
    ("what time is it", "current_time"),
])
def test_runtime_discovery_bypasses_ambient_retrieval_with_scripted_provider(user_request, tool_name):
    provider = FakeProvider({"result": "PLAN_PROPOSED", "steps": [{
        "step_id": "discover", "title": "Discover", "description": user_request,
        "primary_tool": tool_name,
    }]})
    request = planner_input(context=ExecutionContext(user_request=user_request)).model_copy(update={
        "capabilities": PlanningCapabilities(available_tools=(tool_name, "rag_search")),
    })
    def unexpected_retrieval(_query):
        raise AssertionError("Live workspace discovery must not query ambient retrieval")
    result = service(provider, route="info").run(request, retrieve=unexpected_retrieval)
    assert result.outcome == PlannerOutcome.EXECUTION_PLAN
    assert result.proposed_plan.steps[0].primary_tool == tool_name
    assert result.planner_route == "info"
