"""File coverage contract, real transport capture, and bound lifecycle checks."""

from copy import deepcopy

import pytest

from core.completion import CompletionService, EvidenceSnapshot, immutable
from core.completion_providers.filesystem import FileReadCollectionProvider, PROVIDER_ID
from core.graph_capture import _build_tool_result
from core.graph_controller import create_controller_node
from core.models import ListFilesResult, ReadFileResult
from core.protocol.controller import CortexController
from core.protocol.enums import BrainOutcomeKind, ExecutionPhase, StepStatus
from core.protocol.models import (
    BrainOutcome, ControllerInput, CoverageRequirement, ExecutionContext,
    ExecutionCursor, ExecutionIdentity, ExecutionPlan, ExecutionState, ExecutionStep,
    ProtocolVisibleState, ResolvedCoverage, ToolExecutionRecord, ToolRequest, WorkingState,
)


def spec():
    return {"root": ".", "discovery_step_ids": ["discover"],
            "processing_step_ids": ["read"],
            "selection": {"kind": "files", "suffix": ".py", "recursive": False},
            "processing": "full_content_read"}


def capture(name, step, request_id, arguments, payload):
    request = ToolRequest(request_id=request_id, tool_name=name, arguments=arguments)
    result = _build_tool_result(request=request, raw_content=payload.to_tool_output())
    return ToolExecutionRecord(execution_id="e", plan_id="p", plan_revision=1,
                               step_id=step, tool_name=name, arguments=arguments, result=result)


def listing(entries=("a.py", "b.py", "c.py"), root=".", **changes):
    payload = ListFilesResult(success=True, message="Listing", path=root,
                              entries=list(entries), is_file=False).model_copy(update=changes)
    return capture("list_files", "discover", "listing", {"path": root}, payload)


def read(path="a.py", content="hello", offset=0, total=None, truncated=False, **changes):
    payload = ReadFileResult(success=True, message="Read", path=path, content=content,
                             offset=offset, total_chars=len(content) if total is None else total,
                             read_chars=len(content), is_truncated=truncated).model_copy(update=changes)
    return capture("read_file", "read", f"read-{path}", {"path": path, "offset": offset}, payload)


def snapshot(*records):
    return EvidenceSnapshot("scope", "evidence", tuple(immutable(r.model_dump(mode="json")) for r in records))


def resolved(provider, specification, *records):
    resolution = provider.resolve(immutable(specification), snapshot(*records))
    assert resolution.required_item_ids is not None
    return ResolvedCoverage(scope_id="scope", provider_version=provider.contract_version,
                            required_item_ids=resolution.required_item_ids,
                            source_evidence_ids=resolution.source_evidence_ids)


def assess(*records, specification=None, discovery=None):
    provider = FileReadCollectionProvider()
    specification = specification or spec()
    membership = resolved(provider, specification, discovery or listing())
    return provider.assess(immutable(specification), membership, snapshot(*records))


def change_data(record, **changes):
    data = {**record.result.data, **changes}
    return record.model_copy(update={"result": record.result.model_copy(update={"data": data})})


def test_three_members_two_reads_then_final_read():
    provider = FileReadCollectionProvider()
    membership = resolved(provider, spec(), listing())
    assert membership.required_item_ids == ("a.py", "b.py", "c.py")
    assert membership.source_evidence_ids == ("listing",)
    first = assess(read(), read("b.py"))
    assert first.status == "missing" and first.missing_item_ids == ("c.py",)
    assert assess(read(), read("b.py"), read("c.py")).status == "satisfied"


@pytest.mark.parametrize("entries", [(), ("notes.txt", "something.py/", "docs/")])
def test_proven_empty_collection(entries):
    assert assess(discovery=listing(entries)).status == "satisfied"


def test_filters_and_deduplicates_in_stable_order():
    membership = resolved(FileReadCollectionProvider(), spec(), listing(
        ("b.py", "a.py", "a.py", "notes.txt", "folder.py/", "A.PY")))
    assert membership.required_item_ids == ("a.py", "b.py")


@pytest.mark.parametrize("records", [
    (), (listing(success=False),), (listing(is_file=True),),
    (listing().model_copy(update={"step_id": "wrong"}),),
    (listing().model_copy(update={"tool_name": "wrong"}),), (listing(root="elsewhere"),),
])
def test_no_qualifying_discovery_is_unresolved(records):
    assert FileReadCollectionProvider().resolve(spec(), snapshot(*records)).required_item_ids is None


@pytest.mark.parametrize("field,value", [("entries", None), ("entries", "a.py"),
                                          ("is_file", 0), ("path", "other")])
def test_malformed_matching_discovery_raises(field, value):
    with pytest.raises(ValueError):
        FileReadCollectionProvider().resolve(spec(), snapshot(change_data(listing(), **{field: value})))


def test_missing_discovery_fields_never_invent_empty_membership():
    record = listing()
    record = record.model_copy(update={"result": record.result.model_copy(update={"data": {"path": "."}})})
    with pytest.raises(ValueError):
        FileReadCollectionProvider().resolve(spec(), snapshot(record))


@pytest.mark.parametrize("entry", ["../a.py", "/a.py", "src/a.py", "./a.py", "C:a.py", "", ".", 1])
def test_invalid_direct_child_entry_raises(entry):
    with pytest.raises(ValueError):
        FileReadCollectionProvider().resolve(spec(), snapshot(change_data(listing(), entries=[entry])))


def test_incomplete_listing_is_unresolved_and_first_complete_listing_wins():
    first = listing()
    first = first.model_copy(update={"result": first.result.model_copy(update={
        "integrity": first.result.integrity.model_copy(update={"is_truncated": True})})})
    provider = FileReadCollectionProvider()
    assert provider.resolve(spec(), snapshot(first)).required_item_ids is None
    result = provider.resolve(spec(), snapshot(first, listing(("b.py",)), listing(())))
    assert result.required_item_ids == ("b.py",)


@pytest.mark.parametrize("record", [
    read().model_copy(update={"step_id": "wrong"}),
    read().model_copy(update={"tool_name": "wrong"}),
    read("unrelated.py", content="bad", read_chars=99),
    read(success=False), read(offset=1, total=6), read(total=10),
    read(content="he\n\n--- [TRUNCATED] ---", total=5, truncated=True, read_chars=2),
])
def test_unrelated_failed_and_partial_reads_remain_missing(record):
    assert assess(record).missing_item_ids == ("a.py", "b.py", "c.py")


@pytest.mark.parametrize("changes", [
    {"content": "bad"}, {"read_chars": True}, {"total_chars": -1},
    {"is_truncated": 0}, {"offset": 1}, {"path": "b.py"},
])
def test_malformed_matching_read_raises(changes):
    with pytest.raises(ValueError):
        assess(change_data(read(), **changes))


@pytest.mark.parametrize("content", ["", "ç漢😀\n", "hello"])
def test_full_reads_and_duplicates(content):
    record = read(content=content)
    result = assess(record, record, discovery=listing(("a.py",)))
    assert result.status == "satisfied" and result.satisfied_item_ids == ("a.py",)


def test_continuation_or_integrity_flags_block_full_read():
    record = read()
    for update in ({"integrity": record.result.integrity.model_copy(update={"stdout_truncated": True})},
                   {"pagination": record.result.pagination.model_copy(update={"has_more": True})}):
        flagged = record.model_copy(update={"result": record.result.model_copy(update=update)})
        assert assess(flagged).satisfied_item_ids == ()


def test_partial_chunks_are_never_aggregated():
    first = read(content="he\n\n--- [TRUNCATED] ---", total=5, truncated=True, read_chars=2)
    last = read(content="llo", offset=2, total=5)
    assert assess(first, last, discovery=listing(("a.py",))).missing_item_ids == ("a.py",)


def test_empty_membership_does_not_inspect_unrelated_read_targets():
    unrelated = read().model_copy(update={"arguments": {"path": "/unsupported"}})
    assert assess(unrelated, discovery=listing(())).status == "satisfied"


def test_listing_documented_default_argument_is_supported():
    record = listing(("a.py",)).model_copy(update={"arguments": {}})
    assert resolved(FileReadCollectionProvider(), spec(), record).required_item_ids == ("a.py",)


def test_lexical_normalization_and_case_preservation():
    specification = spec()
    specification["root"] = ".\\src//."
    discovery = listing(("a.py", "folder.py\\"), root="src/")
    record = change_data(read(".\\src\\a.py"), path="src/./a.py")
    result = assess(record, specification=specification, discovery=discovery)
    assert result.status == "satisfied" and result.satisfied_item_ids == ("src/a.py",)
    assert assess(read("src/A.py"), specification=specification, discovery=discovery).status == "missing"


@pytest.mark.parametrize("path", ["", "../a.py", "src/../a.py", "/a.py", "C:\\a.py",
                                 "C:a.py", "\\\\server\\a.py", "a.py:stream", "a.py.", "a.py ", "a\x00.py"])
def test_unsupported_paths_rejected(path):
    specification = spec()
    specification["root"] = path
    with pytest.raises(ValueError):
        FileReadCollectionProvider().validate(specification)
    record = read().model_copy(update={"arguments": {"path": path}})
    with pytest.raises(ValueError):
        assess(record)


@pytest.mark.parametrize("changes", [
    {"extra": True}, {"root": 1}, {"processing": "explain"},
    {"discovery_step_ids": []}, {"discovery_step_ids": "discover"},
    {"processing_step_ids": [False]}, {"processing_step_ids": ["read", "read"]},
    {"selection": {"kind": "files", "suffix": ".py", "recursive": True}},
    {"selection": {"kind": "files", "suffix": ".py", "recursive": 0}},
    {"selection": {"kind": "directories", "suffix": ".py", "recursive": False}},
    {"selection": {"kind": "files", "suffix": "", "recursive": False}},
    {"selection": {"kind": "files", "suffix": "*.py", "recursive": False}},
    {"selection": {"kind": "files", "suffix": ".py", "recursive": False, "extra": 1}},
])
def test_strict_specification(changes):
    with pytest.raises(ValueError):
        FileReadCollectionProvider().validate({**spec(), **changes})


def test_no_mutation_and_reads_before_discovery_count():
    specification = spec()
    records = (read(), listing(("a.py",)))
    before = deepcopy(specification), tuple(r.model_dump_json() for r in records)
    provider = FileReadCollectionProvider()
    provider.validate(specification)
    membership = resolved(provider, specification, *records)
    frozen_before = membership.model_dump_json()
    result = provider.assess(immutable(specification), membership, snapshot(*records))
    assert result.status == "satisfied"
    assert (specification, tuple(r.model_dump_json() for r in records)) == before
    assert membership.model_dump_json() == frozen_before


@pytest.mark.parametrize("key", tuple(spec()))
def test_specification_requires_every_field(key):
    specification = spec()
    del specification[key]
    with pytest.raises(ValueError):
        FileReadCollectionProvider().validate(specification)


@pytest.mark.parametrize("key", ("path", "content", "total_chars", "offset", "read_chars", "is_truncated"))
def test_read_requires_every_structured_field(key):
    record = read()
    data = dict(record.result.data)
    del data[key]
    record = record.model_copy(update={"result": record.result.model_copy(update={"data": data})})
    with pytest.raises(ValueError):
        assess(record)


def bound_context(records, requirement=True):
    service = CompletionService({PROVIDER_ID: FileReadCollectionProvider()})
    step = ExecutionStep(step_id="read", title="Read files", status=StepStatus.ACTIVE,
                         completion_requirement=CoverageRequirement(provider_id=PROVIDER_ID,
                             specification=spec()) if requirement else None)
    plan = ExecutionPlan(plan_id="p", revision=1, steps=(
        ExecutionStep(step_id="discover", title="Discover", status=StepStatus.COMPLETED), step))
    identity = ExecutionIdentity(execution_id="e", protocol_version="1")
    _, error, bindings = service.bind_plan(identity, plan)
    assert error is None
    return service, ControllerInput(identity=identity,
        cursor=ExecutionCursor(phase=ExecutionPhase.EXECUTING, step_id="read", plan_revision=1),
        context=ExecutionContext(user_request="Read all Python files"),
        active_plan=plan, active_step=step, accepted_requirements=bindings,
        tool_execution_history=records,
        brain_result=BrainOutcome(outcome=BrainOutcomeKind.STEP_COMPLETED, step_id="read"))


def evaluate(service, context, frozen=()):
    return service.evaluate(context.identity, context.active_plan, context.active_step,
                            context.tool_execution_history, frozen, bindings=context.accepted_requirements)


@pytest.mark.parametrize("outcome", [BrainOutcomeKind.STEP_COMPLETED, BrainOutcomeKind.FINAL_ANSWER_READY])
def test_bound_completion_and_final_answer_gate(outcome):
    service, context = bound_context((listing(), read(), read("b.py")))
    context = context.model_copy(update={"brain_result": BrainOutcome(outcome=outcome, step_id="read")})
    assessment, frozen = evaluate(service, context)
    assert assessment.missing_item_ids == ("c.py",)
    controller = CortexController(20)
    assert controller.decide(context.model_copy(update={"coverage_assessment": assessment})).completed_step_id is None
    context = context.model_copy(update={"tool_execution_history": (*context.tool_execution_history, read("c.py"), listing(()))})
    assessment, after = evaluate(service, context, frozen)
    assert after == frozen and assessment.status == "satisfied"
    assert controller.decide(context.model_copy(update={"coverage_assessment": assessment})).completed_step_id == "read"


@pytest.mark.parametrize("foreign", [{"execution_id": "other"}, {"plan_id": "other"}, {"plan_revision": 2}])
def test_service_excludes_foreign_discovery_and_processing(foreign):
    service, context = bound_context((listing().model_copy(update=foreign),))
    assert evaluate(service, context)[0].status == "unresolved"
    context = context.model_copy(update={"tool_execution_history": (
        listing(("a.py",)), read().model_copy(update=foreign))})
    assert evaluate(service, context)[0].missing_item_ids == ("a.py",)


@pytest.mark.parametrize("records", [(change_data(listing(), entries=None),),
                                      (listing(), change_data(read(), content="bad"))])
def test_service_maps_malformed_evidence_to_provider_error(records):
    service, context = bound_context(records)
    assessment, _ = evaluate(service, context)
    assert assessment.status == "error" and assessment.reason == "completion_provider_error"


def test_no_requirement_behavior_unchanged():
    service, context = bound_context((), requirement=False)
    assert evaluate(service, context) == (None, ())
    assert CortexController(20).decide(context).completed_step_id == "read"


def test_service_read_captured_while_unresolved_counts_after_discovery():
    service, context = bound_context((read(),))
    assessment, frozen = evaluate(service, context)
    assert assessment.status == "unresolved" and frozen == ()
    context = context.model_copy(update={"tool_execution_history": (read(), listing(("a.py",)))})
    assessment, frozen = evaluate(service, context, frozen)
    assert assessment.status == "satisfied" and frozen[0].required_item_ids == ("a.py",)


def test_composition_registers_versioned_provider(monkeypatch):
    from unittest.mock import Mock
    from core import graph_nodes

    factory = Mock()
    monkeypatch.setattr(graph_nodes, "create_controller_node", factory)
    for name in ("create_planner_node", "create_capture_tool_output_node",
                 "create_summarize_memory_node", "create_brain_node"):
        monkeypatch.setattr(graph_nodes, name, Mock())
    graph_nodes.create_graph_nodes(
        brain_llm=None, tool_brain_llm=None, planner_llm=None, rag_service=None,
        rag_top_k=0, agent_system_prompt="",
        casual_system_prompt="", sap_system_prompt=None,
        tools_set=set(), show_raw_llm=False,
    )
    service = factory.call_args.kwargs["completion_service"]
    _, context = bound_context(())
    _, error, bindings = service.bind_plan(context.identity, context.active_plan)
    assert error is None
    assert bindings[0].provider_id == PROVIDER_ID
    assert bindings[0].provider_version == "1"


def test_graph_checkpoint_freezes_membership_and_discovery_reference():
    service, context = bound_context((read(), listing()))
    state = ExecutionState(protocol_visible=ProtocolVisibleState(
        identity=context.identity, cursor=context.cursor, active_plan=context.active_plan,
        active_step=context.active_step, accepted_requirements=context.accepted_requirements),
        working=WorkingState(tool_execution_history=context.tool_execution_history))
    node = create_controller_node(completion_service=service)
    output = node({"execution_state": state, "brain_result": context.brain_result})
    restored = ExecutionState.model_validate_json(output["execution_state"].model_dump_json())
    membership = restored.protocol_visible.resolved_coverages
    assert membership[0].source_evidence_ids == ("listing",)
    assert membership[0].required_item_ids == ("a.py", "b.py", "c.py")
    assert restored.working.coverage_assessment.missing_item_ids == ("b.py", "c.py")
    assert membership[0].evidence_id == restored.working.coverage_assessment.evidence_id
    restored = restored.model_copy(update={"working": restored.working.model_copy(update={
        "tool_execution_history": (*restored.working.tool_execution_history, listing(()))})})
    output = node({"execution_state": restored, "brain_result": context.brain_result})
    assert output["execution_state"].protocol_visible.resolved_coverages == membership
