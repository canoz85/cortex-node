# CortexNode Implementation Specification (CIS)

- Specification Family: CortexNode Implementation Specification (CIS)
- Layer: Layer 3 (Implementation Specification)
- Status: Active Index Document

## 1. Purpose
The CortexNode Implementation Specification (CIS) series describes how the current CortexNode runtime realizes the protocol defined by the CortexNode Execution Protocol (CEP).

CEP defines protocol semantics.

CIS defines runtime realization.

Protocol authority always belongs to CEP.

Runtime implementation may evolve while preserving CEP semantics.

### 1.1 What CIS Is Not

CIS does not define protocol behavior.

CIS does not introduce new execution semantics.

CIS does not replace CEP.

CIS exists only to describe how the current CortexNode runtime realizes the protocol defined by CEP.

## 2. Layer Relationship
CortexNode documentation is organized as a layered architecture set.

Layer 1:
- RFC
- Architecture vision
- Project direction

Layer 2:
- CEP
- Protocol specification
- Normative behavior

Layer 3:
- CIS
- Runtime realization
- Current implementation mapping

Interpretation guide:
- RFC answers: Why does CortexNode exist?
- CEP answers: What is CortexNode?
- CIS answers: How is CortexNode currently implemented?

## 3. Reading Order
Recommended reading order:
1. RFC documents
2. CEP documents
3. CIS documents

For implementation work:
- read CEP first
- then read CIS
- never the reverse

## 4. CIS Document Overview
| Document | Purpose |
| --- | --- |
| CIS-001 Execution Mapping | Maps CEP concepts to current runtime roles, services, and modules. |
| CIS-002 Runtime State Model | Describes Protocol-visible State and Working State runtime realization. |
| CIS-003 LangGraph Execution Graph | Describes current execution graph realization and controller-governed routing realization. |
| CIS-004 Checkpoint & Recovery | Describes checkpoint, replay, resume, and recovery runtime realization. |
| CIS-005 Worker Runtime | Describes runtime realization of worker roles and supporting runtime services. |
| CIS-006 Testing & Verification | Reserved for implementation verification and protocol conformance validation. |

## 5. Relationship with CEP
Every CIS document references one or more CEP documents.

CEP remains normative.

CIS never modifies protocol semantics.

Change policy:
- If implementation changes while protocol behavior stays the same, update CIS.
- If protocol behavior changes, update CEP first, then update CIS.

## 6. When To Update CIS
Update CIS when:
- runtime modules change in ways that alter architectural responsibility mapping
- worker realization changes
- execution graph realization changes
- runtime state realization changes
- checkpoint realization changes
- recovery realization changes

Do not update CIS for:
- protocol rule changes alone without implementation realization change
- architecture vision changes that belong to RFC layer
- implementation bugs that do not alter architectural realization

## 7. Evolution Policy
Runtime implementation may evolve.

Modules may split.

Modules may merge.

Technologies may change.

Frameworks may change.

Observable CEP behavior MUST remain unchanged unless CEP is updated.

CIS documents SHOULD always reflect the current runtime implementation.

## 8. Design Principles
- Controller owns orchestration authority.
- CEP defines protocol semantics.
- CIS documents runtime realization.
- Workers remain role-scoped.
- Runtime Services never become protocol authorities.
- Protocol-visible State remains authoritative for protocol behavior.
- Accepted Event History remains append-only.
- Replay remains deterministic.
- Resume preserves completed work.
- Runtime evolution preserves protocol-equivalent observable behavior.

## 9. Contributing
Recommended documentation workflow:
1. Read relevant CEP document(s).
2. Read relevant CIS document(s).
3. Modify runtime implementation.
4. Update affected CIS document(s) when runtime realization changes.
5. Update CEP only when protocol semantics change.

Contribution rule:
- keep CIS implementation-focused
- avoid duplicating normative CEP semantics
- preserve CEP authority boundaries in all CIS updates

## 10. Future Work
The CIS series is intended to remain implementation-focused.

Future CIS documents may describe:

- implementation verification
- protocol conformance testing
- runtime performance validation
- deployment realization
- distributed execution realization

Additional CIS documents should continue to map runtime implementation without redefining protocol semantics.

## 11. Documentation Rule

When changing CortexNode:

1. Determine whether the change affects protocol semantics or only implementation.
2. Update CEP first if protocol semantics change.
3. Update CIS whenever runtime realization changes.
4. Keep RFC documents focused on long-term architectural direction.

Protocol semantics must never be inferred from CIS.
