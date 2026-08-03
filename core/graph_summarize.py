import hashlib
import json
import re

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.graph_constants import MAX_SUMMARY_CHARS, MAX_SUMMARY_TURNS
from core.graph_messages import recent_human_turn_messages, recent_turn_slice
from core.graph_tool_events import current_turn_tool_events
from core.state import AgentState


def rolling_summary_message(summary: str) -> list[SystemMessage]:
    compact = (summary or "").strip()
    if not compact:
        return []
    return [
        SystemMessage(
            content=(
                "Rolling memory from earlier turns (user profile and stable goals may be used directly; "
                "workspace/system/runtime facts must still be verified with tools):\n"
                f"{compact}"
            )
        )
    ]


def create_summarize_memory_node(*, summarize_llm: ChatOllama):
    USER_CATEGORIES = {"profile", "preferences", "constraints", "goals"}
    PROJECT_CATEGORIES = {"repo", "architecture", "environment", "tooling", "workflow", "domain"}
    SOURCE_TYPES = {"human", "tool", "assistant_inferred"}
    QUESTION_STATUS = {"open", "resolved"}
    QUESTION_PRIORITY = {"low", "medium", "high"}

    def _empty_payload() -> dict:
        return {
            "schema_version": 3,
            "user_profile": [],
            "project_facts": [],
            "active_task_state": {
                "objective": "",
                "status": "unknown",
                "last_completed_step": "",
                "next_step": "",
                "confidence": 0.0,
                "source_type": "assistant_inferred",
                "source_turn": 0,
                "source_tool_signature": "",
            },
            "open_questions": [],
            "meta": {
                "updated_at_turn": 0,
                "last_task_update_turn": 0,
            },
        }

    def _extract_json_object(text: str) -> str | None:
        compact = (text or "").strip()
        if not compact:
            return None

        if compact.startswith("```"):
            compact = re.sub(r"^```(?:json)?\s*", "", compact)
            compact = re.sub(r"\s*```$", "", compact).strip()

        if compact.startswith("{") and compact.endswith("}"):
            return compact

        first = compact.find("{")
        last = compact.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        return compact[first : last + 1]

    def _stable_id(*parts: str) -> str:
        key = "|".join((part or "").strip().lower() for part in parts)
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def _normalize_source_type(value: object, *, default: str) -> str:
        text = str(value or "").strip().lower()
        if text in SOURCE_TYPES:
            return text
        return default

    def _normalize_memory_entry(item: object, *, default_category: str, default_source: str) -> dict | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            category = default_category
            return {
                "id": _stable_id(category, text),
                "text": text,
                "category": category,
                "confidence": 0.7,
                "source_type": default_source,
                "source_turn": 0,
                "source_tool_signature": "",
            }

        if not isinstance(item, dict):
            return None

        text = str(item.get("text", "")).strip()
        if not text:
            return None

        category = str(item.get("category", default_category)).strip().lower() or default_category
        if category in USER_CATEGORIES:
            category_set = USER_CATEGORIES
        elif category in PROJECT_CATEGORIES:
            category_set = PROJECT_CATEGORIES
        else:
            category_set = USER_CATEGORIES if default_category in USER_CATEGORIES else PROJECT_CATEGORIES
            category = default_category

        if category not in category_set:
            category = default_category

        confidence = item.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        confidence = max(0.0, min(float(confidence), 1.0))

        fid = str(item.get("id", "")).strip() or _stable_id(category, text)
        source_type = _normalize_source_type(item.get("source_type"), default=default_source)
        source_turn = item.get("source_turn", 0)
        if not isinstance(source_turn, int) or source_turn < 0:
            source_turn = 0
        source_tool_signature = str(item.get("source_tool_signature", "") or "").strip()

        return {
            "id": fid,
            "text": text,
            "category": category,
            "confidence": confidence,
            "source_type": source_type,
            "source_turn": source_turn,
            "source_tool_signature": source_tool_signature,
        }

    def _normalize_question(item: object) -> dict | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {
                "text": text,
                "status": "open",
                "priority": "medium",
                "source_type": "human",
                "source_turn": 0,
            }

        if not isinstance(item, dict):
            return None

        text = str(item.get("text", "")).strip()
        if not text:
            return None

        status = str(item.get("status", "open")).strip().lower() or "open"
        if status not in QUESTION_STATUS:
            status = "open"

        priority = str(item.get("priority", "medium")).strip().lower() or "medium"
        if priority not in QUESTION_PRIORITY:
            priority = "medium"

        source_type = _normalize_source_type(item.get("source_type"), default="human")
        source_turn = item.get("source_turn", 0)
        if not isinstance(source_turn, int) or source_turn < 0:
            source_turn = 0

        return {
            "text": text,
            "status": status,
            "priority": priority,
            "source_type": source_type,
            "source_turn": source_turn,
        }

    def _normalize_task_state(item: object) -> dict:
        empty = _empty_payload()["active_task_state"]
        if not isinstance(item, dict):
            return empty

        objective = str(item.get("objective", "") or "").strip()
        status = str(item.get("status", "unknown") or "unknown").strip().lower()
        last_completed_step = str(item.get("last_completed_step", "") or "").strip()
        next_step = str(item.get("next_step", "") or "").strip()
        confidence = item.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(float(confidence), 1.0))

        source_type = _normalize_source_type(item.get("source_type"), default="assistant_inferred")
        source_turn = item.get("source_turn", 0)
        if not isinstance(source_turn, int) or source_turn < 0:
            source_turn = 0

        source_tool_signature = str(item.get("source_tool_signature", "") or "").strip()

        return {
            "objective": objective,
            "status": status,
            "last_completed_step": last_completed_step,
            "next_step": next_step,
            "confidence": confidence,
            "source_type": source_type,
            "source_turn": source_turn,
            "source_tool_signature": source_tool_signature,
        }

    def _normalize_summary_payload(payload: dict | None) -> dict:
        normalized = _empty_payload()
        if not isinstance(payload, dict):
            return normalized

        version = payload.get("schema_version", 1)
        if isinstance(version, int):
            normalized["schema_version"] = max(1, version)

        # Backward compatibility: schema v2 stores all facts in one array.
        legacy_facts = payload.get("facts", []) if normalized["schema_version"] < 3 else []

        user_profile = payload.get("user_profile", [])
        if isinstance(user_profile, list):
            seen_ids: set[str] = set()
            for raw in user_profile:
                fact = _normalize_memory_entry(raw, default_category="profile", default_source="human")
                if not fact or fact["category"] not in USER_CATEGORIES:
                    continue
                if fact["id"] in seen_ids:
                    continue
                seen_ids.add(fact["id"])
                normalized["user_profile"].append(fact)

        project_facts = payload.get("project_facts", [])
        if isinstance(project_facts, list):
            seen_ids: set[str] = set()
            for raw in project_facts:
                fact = _normalize_memory_entry(raw, default_category="architecture", default_source="tool")
                if not fact or fact["category"] not in PROJECT_CATEGORIES:
                    continue
                if fact["id"] in seen_ids:
                    continue
                seen_ids.add(fact["id"])
                normalized["project_facts"].append(fact)

        if isinstance(legacy_facts, list):
            existing = {f["id"] for f in normalized["user_profile"]}
            for raw in legacy_facts:
                fact = _normalize_memory_entry(raw, default_category="profile", default_source="human")
                if not fact or fact["id"] in existing:
                    continue
                normalized["user_profile"].append(fact)
                existing.add(fact["id"])

        normalized["active_task_state"] = _normalize_task_state(payload.get("active_task_state"))

        oq = payload.get("open_questions", [])
        if isinstance(oq, list):
            seen_q: set[str] = set()
            for raw in oq:
                q = _normalize_question(raw)
                if not q:
                    continue
                key = q["text"].lower()
                if key in seen_q:
                    continue
                seen_q.add(key)
                normalized["open_questions"].append(q)

        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            turn = meta.get("updated_at_turn", 0)
            if isinstance(turn, int) and turn >= 0:
                normalized["meta"]["updated_at_turn"] = turn

            task_turn = meta.get("last_task_update_turn", 0)
            if isinstance(task_turn, int) and task_turn >= 0:
                normalized["meta"]["last_task_update_turn"] = task_turn

        normalized["schema_version"] = max(3, normalized["schema_version"])
        return normalized

    def _parse_summary_payload(text: str) -> dict | None:
        blob = _extract_json_object(text)
        if not blob:
            return None
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return _normalize_summary_payload(parsed)

    def _payload_to_text(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _is_personal_fact(text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if "?" in t:
            return False
        lower = t.lower()
        blocked = ("run ", "create ", "delete ", "list ", "what is ", "how to ")
        if any(lower.startswith(b) for b in blocked):
            return False
        return True

    def _is_project_fact(text: str) -> bool:
        t = text.strip()
        if not t or "?" in t:
            return False
        lower = t.lower()
        blocked = ("i am ", "my name is ", "call me ", "please ")
        if any(lower.startswith(b) for b in blocked):
            return False
        return True

    def _merge_fact_lists(
        base_facts: list[dict],
        incoming_facts: list[dict],
        *,
        filter_fn,
    ) -> list[dict]:
        by_id = {f["id"]: f for f in base_facts}
        for item in incoming_facts:
            if not filter_fn(item.get("text", "")):
                continue

            existing = by_id.get(item["id"])
            if existing is None:
                by_id[item["id"]] = item
                continue

            # Prefer tool-grounded entries and higher confidence.
            prefer_new = False
            if item.get("source_type") == "tool" and existing.get("source_type") != "tool":
                prefer_new = True
            elif item.get("confidence", 0.0) > existing.get("confidence", 0.0):
                prefer_new = True
            elif item.get("source_turn", 0) > existing.get("source_turn", 0):
                prefer_new = True

            if prefer_new:
                by_id[item["id"]] = item
            else:
                existing["confidence"] = max(existing.get("confidence", 0.0), item.get("confidence", 0.0))

        return list(by_id.values())

    def _merge_task_state(existing_task: dict, candidate_task: dict, turn_index: int) -> tuple[dict, int]:
        prev = _normalize_task_state(existing_task)
        new = _normalize_task_state(candidate_task)

        meaningful = any(
            [
                bool(new.get("objective")),
                bool(new.get("last_completed_step")),
                bool(new.get("next_step")),
                new.get("status", "unknown") not in {"", "unknown"},
            ]
        )
        if not meaningful:
            return prev, 0

        merged = {
            "objective": new.get("objective") or prev.get("objective", ""),
            "status": new.get("status") or prev.get("status", "unknown"),
            "last_completed_step": new.get("last_completed_step") or prev.get("last_completed_step", ""),
            "next_step": new.get("next_step") or prev.get("next_step", ""),
            "confidence": max(prev.get("confidence", 0.0), new.get("confidence", 0.0)),
            "source_type": new.get("source_type") or prev.get("source_type", "assistant_inferred"),
            "source_turn": max(new.get("source_turn", 0), turn_index),
            "source_tool_signature": new.get("source_tool_signature") or prev.get("source_tool_signature", ""),
        }
        return merged, turn_index

    def _merge(existing: dict, candidate: dict, turn_index: int) -> dict:
        out = _normalize_summary_payload(existing)
        inc = _normalize_summary_payload(candidate)

        out["user_profile"] = _merge_fact_lists(
            out["user_profile"],
            inc["user_profile"],
            filter_fn=_is_personal_fact,
        )

        out["project_facts"] = _merge_fact_lists(
            out["project_facts"],
            inc["project_facts"],
            filter_fn=_is_project_fact,
        )

        merged_task, task_updated_turn = _merge_task_state(
            out.get("active_task_state", {}),
            inc.get("active_task_state", {}),
            turn_index,
        )
        out["active_task_state"] = merged_task

        by_text = {q["text"].lower(): q for q in out["open_questions"]}
        for q in inc["open_questions"]:
            key = q["text"].lower()
            by_text[key] = q
        out["open_questions"] = list(by_text.values())

        out["schema_version"] = 3
        out["meta"]["updated_at_turn"] = max(turn_index, out["meta"].get("updated_at_turn", 0))
        out["meta"]["last_task_update_turn"] = max(
            task_updated_turn,
            out["meta"].get("last_task_update_turn", 0),
        )
        return out

    def _enforce_budget(payload: dict, max_chars: int = MAX_SUMMARY_CHARS) -> str:
        # Keep JSON valid by pruning objects, never by string clipping.
        work = _normalize_summary_payload(payload)

        def _emit() -> str:
            return _payload_to_text(work)

        text = _emit()
        if len(text) <= max_chars:
            return text

        # Priority keep order: active_task_state > open_questions > project_facts > user_profile
        # Prune lowest-priority classes first.
        work["user_profile"].sort(key=lambda x: (x.get("confidence", 0.0), len(x.get("text", ""))))
        while work["user_profile"] and len(_emit()) > max_chars:
            work["user_profile"].pop(0)

        work["project_facts"].sort(key=lambda x: (x.get("confidence", 0.0), len(x.get("text", ""))))
        while work["project_facts"] and len(_emit()) > max_chars:
            work["project_facts"].pop(0)

        while work["open_questions"] and len(_emit()) > max_chars:
            work["open_questions"].pop()

        if len(_emit()) > max_chars:
            work["active_task_state"] = _empty_payload()["active_task_state"]

        # Guaranteed valid JSON even if mostly empty.
        return _emit()

    def update_rolling_summary(
        summarize_llm: ChatOllama,
        existing_summary: str,
        facts_history: list,
        questions_history: list,
        tool_events: list[dict],
        turn_index: int,
    ) -> str:
        if not facts_history and not existing_summary:
            return ""

        existing_payload = _parse_summary_payload(existing_summary) or _empty_payload()

        def _fmt_messages(msgs: list) -> str:
            parts = []
            for m in msgs:
                role = getattr(m, "type", "unknown")
                content = str(getattr(m, "content", "") or "").strip()
                if content:
                    parts.append(f"[{role}]: {content}")
            return "\n".join(parts) or "(none)"

        facts_block = _fmt_messages(facts_history)
        questions_block = _fmt_messages(questions_history)

        successful_tools = [event for event in tool_events if event.get("success")]

        def _fmt_tool_events(events: list[dict]) -> str:
            if not events:
                return "(none)"
            lines: list[str] = []
            for event in events:
                name = str(event.get("name", "unknown"))
                signature = str(event.get("signature", ""))
                unwrapped = event.get("unwrapped")
                if isinstance(unwrapped, dict):
                    message = str(unwrapped.get("message", "") or "").strip()
                    data = unwrapped.get("data")
                else:
                    message = ""
                    data = None
                data_preview = ""
                if isinstance(data, dict):
                    keys = ",".join(sorted(str(k) for k in data.keys())[:6])
                    if keys:
                        data_preview = f" data_keys={keys}"
                lines.append(f"- tool={name} signature={signature} message={message}{data_preview}".strip())
            return "\n".join(lines)

        tools_block = _fmt_tool_events(successful_tools)

        summarization_prompt = (
            "You are an internal memory updater.\n"
            "Return ONLY a raw JSON object, no markdown, no commentary.\n"
            "Schema:\n"
            "{"
            "\"schema_version\":3,"
            "\"user_profile\":[{\"id\":\"...\",\"text\":\"...\",\"category\":\"profile|preferences|constraints|goals\",\"confidence\":0.7,\"source_type\":\"human|assistant_inferred\",\"source_turn\":0,\"source_tool_signature\":\"\"}],"
            "\"project_facts\":[{\"id\":\"...\",\"text\":\"...\",\"category\":\"repo|architecture|environment|tooling|workflow|domain\",\"confidence\":0.7,\"source_type\":\"tool|assistant_inferred\",\"source_turn\":0,\"source_tool_signature\":\"\"}],"
            "\"active_task_state\":{\"objective\":\"...\",\"status\":\"in_progress|blocked|done|unknown\",\"last_completed_step\":\"...\",\"next_step\":\"...\",\"confidence\":0.7,\"source_type\":\"tool|assistant_inferred\",\"source_turn\":0,\"source_tool_signature\":\"\"},"
            "\"open_questions\":[{\"text\":\"...\",\"status\":\"open|resolved\",\"priority\":\"low|medium|high\",\"source_type\":\"human|assistant_inferred\",\"source_turn\":0}],"
            "\"meta\":{\"updated_at_turn\":0,\"last_task_update_turn\":0}"
            "}\n\n"
            "Section rules:\n"
            "HUMAN_FACT_EVIDENCE contains user messages only.\n"
            "  - Add durable user profile/preferences/constraints/goals to user_profile.\n"
            "  - Do NOT store transient commands/tasks as user_profile facts.\n"
            "TOOL_EVIDENCE contains successful tool outcomes from this turn.\n"
            "  - Add durable repository/workspace/tooling facts to project_facts.\n"
            "  - If deriving from tool evidence, set source_type=tool and source_tool_signature.\n"
            "QUESTION_EVIDENCE contains user + AI messages.\n"
            "  - New open_questions must originate from user messages.\n"
            "  - Mark resolved only when AI clearly answered the user's question in this turn.\n"
            "TASK_STATE rules:\n"
            "  - Keep concise objective and next_step for continuity.\n"
            "  - status should be one of in_progress|blocked|done|unknown.\n"
            "  - Prefer source_type=tool when task progress is validated by successful tool output.\n"
            "GENERAL rules:\n"
            "  - Never invent facts unsupported by evidence blocks.\n"
            "  - Keep entries concise and deduplicated.\n"
            "If no new info found, return Existing Summary unchanged."
        )

        summary_messages = [
            SystemMessage(
                content=(
                    f"{summarization_prompt}\n\n"
                    f"Existing summary:\n{_payload_to_text(existing_payload)}\n\n"
                    f"--- HUMAN_FACT_EVIDENCE (user messages only) ---\n"
                    f"{facts_block}\n\n"
                    f"--- TOOL_EVIDENCE (successful tools this turn) ---\n"
                    f"{tools_block}\n\n"
                    f"--- QUESTION_EVIDENCE (user + AI messages) ---\n"
                    f"{questions_block}\n\n"
                    "Output JSON only:"
                )
            ),
        ]

        response = summarize_llm.invoke(summary_messages)
        text = str(getattr(response, "content", "") or "").strip()

        if not text:
            return _enforce_budget(existing_payload)

        candidate = _parse_summary_payload(text)
        if candidate is None:
            return _enforce_budget(existing_payload)

        merged = _merge(existing_payload, candidate, turn_index=turn_index)
        return _enforce_budget(merged)

    def summarize_memory_node(state: AgentState):
        history = state.get("messages", [])
        facts_history = recent_human_turn_messages(history, max_turns=MAX_SUMMARY_TURNS)
        questions_history = recent_turn_slice(history, max_turns=MAX_SUMMARY_TURNS, include_ai=True)
        tool_events = current_turn_tool_events(history)
        previous_summary = state.get("rolling_summary", "")
        turn_index = sum(1 for m in history if getattr(m, "type", "") == "human")

        updated_summary = update_rolling_summary(
            summarize_llm=summarize_llm,
            existing_summary=previous_summary,
            facts_history=facts_history,
            questions_history=questions_history,
            tool_events=tool_events,
            turn_index=turn_index,
        )

        return {
            "rolling_summary": updated_summary,
            "messages": [
                AIMessage(content=state["final_answer"])
            ],
            "final_answer": "",
        }

    return summarize_memory_node