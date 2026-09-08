"""Completion-provider orchestration. Providers receive immutable value snapshots only.

The graph's serialized state transition commits returned resolutions atomically.
The service owns no authoritative in-memory cache and performs no tool execution.
"""

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from core.protocol.completion_identity import evidence_identity, requirement_scope, plan_validation_identity, eligible_records, accepted_step, binding_for
from core.protocol.models import CoverageAssessment, ResolvedCoverage, AcceptedRequirement


def immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({key: immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(item) for item in value)
    return value


@dataclass(frozen=True)
class EvidenceSnapshot:
    scope_id: str
    evidence_id: str
    records: tuple[Mapping, ...]


@dataclass(frozen=True)
class Resolution:
    # None means unresolved; () is a legitimate resolved empty set.
    required_item_ids: tuple[str, ...] | None = None
    source_evidence_ids: tuple[str, ...] = ()
    reason: str = ""


class CompletionProvider(Protocol):
    contract_version: str

    def validate(self, specification: Mapping) -> None:
        """Raise ValueError for an invalid specification. Must be side-effect-free."""

    def resolve(self, specification: Mapping, evidence: EvidenceSnapshot) -> Resolution:
        """Resolve opaque identities from captured records, or report unresolved."""

    def assess(self, specification: Mapping, resolved: ResolvedCoverage,
               evidence: EvidenceSnapshot) -> CoverageAssessment:
        """Assess frozen membership against this exact evidence snapshot."""


class CompletionService:
    def __init__(self, providers: Mapping[str, CompletionProvider] | None = None):
        self._providers = dict(providers or {})

    def validate_plan(self, plan):
        """Return a plan-bound validation receipt or a non-accepting error."""
        for step in plan.steps:
            requirement = step.completion_requirement
            if requirement is None:
                continue
            provider = self._providers.get(requirement.provider_id)
            if provider is None:
                return None, "unknown_completion_provider"
            try:
                if not isinstance(provider.contract_version, str) or not provider.contract_version:
                    return None, "invalid_provider_version"
                provider.validate(immutable(requirement.specification))
            except ValueError:
                return None, "invalid_completion_specification"
            except Exception:
                return None, "completion_provider_error"
        return plan_validation_identity(plan), None

    def bind_plan(self, identity, plan, existing=()):
        receipt, error = self.validate_plan(plan)
        if error:
            return receipt, error, existing
        additions = []
        for step in plan.steps:
            binding = binding_for(identity, plan, step, existing)
            requirement = step.completion_requirement
            if binding and (requirement is None or binding.scope_id != requirement_scope(identity, plan, step)):
                return None, "accepted_requirement_changed", existing
            if requirement is None:
                continue
            if binding:
                if binding.provider_version != self._providers[requirement.provider_id].contract_version:
                    return None, "completion_provider_version_changed", existing
                continue
            additions.append(AcceptedRequirement(
                execution_id=identity.execution_id, plan_id=plan.plan_id, plan_revision=plan.revision,
                step_id=step.step_id, scope_id=requirement_scope(identity, plan, step),
                provider_id=requirement.provider_id,
                provider_version=self._providers[requirement.provider_id].contract_version,
                specification_json=json.dumps(requirement.specification, sort_keys=True),
            ))
        if any(b.execution_id == identity.execution_id and b.plan_id == plan.plan_id
               and b.plan_revision == plan.revision and b.step_id not in {s.step_id for s in plan.steps}
               for b in existing):
            return None, "accepted_requirement_changed", existing
        return receipt, None, (*existing, *additions)

    def evaluate(self, identity, plan, step, records, frozen=(), previous=None, bindings=()):
        """Return (assessment, authoritative resolutions); never mutate inputs."""
        if step is None:
            return None, frozen
        accepted_step(plan, step)
        binding = binding_for(identity, plan, step, bindings)
        if step.completion_requirement is None and binding is None:
            return None, frozen
        records = eligible_records(identity, plan, records)
        scope = binding.scope_id if binding else requirement_scope(identity, plan, step)
        evidence = EvidenceSnapshot(scope, evidence_identity(records), tuple(
            immutable(record.model_dump(mode="json")) for record in records))
        def blocked(reason, status="error"):
            return CoverageAssessment(scope_id=scope, evidence_id=evidence.evidence_id,
                                      status=status, reason=reason), frozen
        requirement = step.completion_requirement
        if binding is None:
            return blocked("accepted_requirement_binding_missing")
        if requirement is None or binding.scope_id != requirement_scope(identity, plan, step):
            return blocked("accepted_requirement_changed")
        provider = self._providers.get(binding.provider_id)
        if provider is None:
            return blocked("completion_provider_unavailable")
        if provider.contract_version != binding.provider_version:
            return blocked("completion_provider_version_changed")
        specification = immutable(json.loads(binding.specification_json))
        try:
            matches = [item for item in frozen if item.scope_id == scope]
            if len(matches) > 1:
                return blocked("ambiguous_frozen_coverage")
            resolved = matches[0] if matches else None
            if resolved is not None and resolved.provider_version != provider.contract_version:
                return blocked("completion_provider_version_changed")
            if resolved is None:
                if (previous is not None and previous.status == "unresolved"
                        and previous.scope_id == scope and previous.evidence_id == evidence.evidence_id):
                    return previous, frozen
                provider.validate(specification)
                resolution = provider.resolve(specification, evidence)
                if not isinstance(resolution, Resolution):
                    return blocked("invalid_provider_resolution")
                if resolution.required_item_ids is None:
                    return blocked(resolution.reason, "unresolved")
                items = resolution.required_item_ids
                refs = resolution.source_evidence_ids
                if (not isinstance(items, tuple) or any(not isinstance(i, str) or not i for i in items)
                        or len(items) != len(set(items))
                        or not isinstance(refs, tuple)
                        or any(not isinstance(i, str) for i in refs)
                        or set(refs) - {r.result.request_id for r in records}):
                    return blocked("invalid_provider_resolution")
                resolved = ResolvedCoverage(scope_id=scope, provider_version=provider.contract_version,
                    required_item_ids=items, source_evidence_ids=refs, evidence_id=evidence.evidence_id)
                frozen = (*frozen, resolved)
            # Providers get a separate frozen value, never the checkpoint object.
            assessment = provider.assess(specification, resolved.model_copy(deep=True), evidence)
            if not isinstance(assessment, CoverageAssessment):
                return blocked("invalid_provider_assessment")
            if assessment.scope_id != scope or assessment.evidence_id != evidence.evidence_id:
                return blocked("stale_provider_assessment", "stale")
            required = set(resolved.required_item_ids)
            satisfied = set(assessment.satisfied_item_ids)
            missing = set(assessment.missing_item_ids)
            if (len(satisfied) != len(assessment.satisfied_item_ids)
                    or len(missing) != len(assessment.missing_item_ids)
                    or satisfied - required or missing - required):
                return blocked("invalid_provider_assessment")
            if assessment.status in {"satisfied", "missing"}:
                if missing != required - satisfied or (assessment.status == "satisfied") != (not missing):
                    return blocked("inconsistent_provider_assessment")
            return assessment, frozen
        except Exception:
            return blocked("completion_provider_error")
