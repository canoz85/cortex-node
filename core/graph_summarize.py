import json
import re
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from core.state import AgentState
from core.graph_constants import MAX_SUMMARY_CHARS, MAX_SUMMARY_TURNS, RECENT_MESSAGE_WINDOW
from core.graph_messages import recent_turn_slice, recent_human_turn_messages


def rolling_summary_message(summary: str) -> list[SystemMessage]:
    compact = (summary or "").strip()
    if not compact:
        return []
    return [
        SystemMessage(
            content=(
                "Rolling summary from earlier turns (context hints, verify file facts with tools):\n"
                f"{compact}"
            )
        )
    ]


def create_summarize_memory_node(*, summarize_llm: ChatOllama):
    def _empty_payload() -> dict:
        return {
            "schema_version": 2,
            "facts": [],
            "open_questions": [],
            "meta": {"updated_at_turn": 0},
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

    def _fact_id(text: str, category: str) -> str:
        base = f"{category}|{text.strip().lower()}"
        return str(abs(hash(base)))

    def _normalize_fact(item: object) -> dict | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            category = "profile"
            return {
                "id": _fact_id(text, category),
                "text": text,
                "category": category,
                "confidence": 0.7,
            }

        if not isinstance(item, dict):
            return None

        text = str(item.get("text", "")).strip()
        if not text:
            return None

        category = str(item.get("category", "profile")).strip() or "profile"
        confidence = item.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        confidence = max(0.0, min(float(confidence), 1.0))

        fid = str(item.get("id", "")).strip() or _fact_id(text, category)
        return {
            "id": fid,
            "text": text,
            "category": category,
            "confidence": confidence,
        }

    def _normalize_question(item: object) -> dict | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"text": text, "status": "open", "priority": "medium"}

        if not isinstance(item, dict):
            return None

        text = str(item.get("text", "")).strip()
        if not text:
            return None

        status = str(item.get("status", "open")).strip() or "open"
        priority = str(item.get("priority", "medium")).strip() or "medium"
        return {"text": text, "status": status, "priority": priority}

    def _normalize_summary_payload(payload: dict | None) -> dict:
        normalized = _empty_payload()
        if not isinstance(payload, dict):
            return normalized

        version = payload.get("schema_version", 1)
        if isinstance(version, int):
            normalized["schema_version"] = max(1, version)

        facts = payload.get("facts", [])
        if isinstance(facts, list):
            seen_ids: set[str] = set()
            for raw in facts:
                fact = _normalize_fact(raw)
                if not fact:
                    continue
                if fact["id"] in seen_ids:
                    continue
                seen_ids.add(fact["id"])
                normalized["facts"].append(fact)

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

    def _merge(existing: dict, candidate: dict, turn_index: int) -> dict:
        out = _normalize_summary_payload(existing)
        inc = _normalize_summary_payload(candidate)

        by_id = {f["id"]: f for f in out["facts"]}
        for f in inc["facts"]:
            if not _is_personal_fact(f["text"]):
                continue
            if f["id"] in by_id:
                prev = by_id[f["id"]]
                prev["confidence"] = max(prev["confidence"], f["confidence"])
            else:
                by_id[f["id"]] = f

        out["facts"] = list(by_id.values())

        by_text = {q["text"].lower(): q for q in out["open_questions"]}
        for q in inc["open_questions"]:
            key = q["text"].lower()
            by_text[key] = q
        out["open_questions"] = list(by_text.values())

        out["schema_version"] = 2
        out["meta"]["updated_at_turn"] = max(turn_index, out["meta"].get("updated_at_turn", 0))
        return out

    def _enforce_budget(payload: dict, max_chars: int = MAX_SUMMARY_CHARS) -> str:
        # Keep JSON valid by pruning objects, never by string clipping.
        work = _normalize_summary_payload(payload)

        def _emit() -> str:
            return _payload_to_text(work)

        text = _emit()
        if len(text) <= max_chars:
            return text

        # Trim open questions first
        while work["open_questions"] and len(_emit()) > max_chars:
            work["open_questions"].pop()

        # Then trim lowest-confidence facts
        work["facts"].sort(key=lambda x: (x.get("confidence", 0.0), len(x.get("text", ""))))
        while work["facts"] and len(_emit()) > max_chars:
            work["facts"].pop(0)

        # Guaranteed valid JSON even if mostly empty.
        return _emit()

    def update_rolling_summary(
        summarize_llm: ChatOllama,
        existing_summary: str,
        facts_history: list,
        questions_history: list,
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

        summarization_prompt = (
            "You are an internal memory updater.\n"
            "Return ONLY a raw JSON object, no markdown, no commentary.\n"
            "Schema:\n"
            "{"
            "\"schema_version\":2,"
            "\"facts\":[{\"id\":\"...\",\"text\":\"...\",\"category\":\"profile|preferences|constraints|goals\",\"confidence\":0.0}],"
            "\"open_questions\":[{\"text\":\"...\",\"status\":\"open|resolved\",\"priority\":\"low|medium|high\"}],"
            "\"meta\":{\"updated_at_turn\":0}"
            "}\n\n"
            "Section rules:\n"
            "FACT_EVIDENCE contains user messages only.\n"
            "  - Extract durable personal profile facts (name, location, role, preferences).\n"
            "  - Do NOT extract questions, commands, tool syntax, or transient tasks as facts.\n"
            "QUESTION_EVIDENCE contains user + AI messages.\n"
            "  - New open_questions must come from user messages only.\n"
            "  - AI replies may change an existing question status to 'resolved' if it was clearly answered.\n"
            "  - Do NOT create new facts from AI messages.\n"
            "If no new info found, return Existing Summary unchanged."
        )

        summary_messages = [
            SystemMessage(
                content=(
                    f"{summarization_prompt}\n\n"
                    f"Existing summary:\n{_payload_to_text(existing_payload)}\n\n"
                    f"--- FACT_EVIDENCE (user messages only) ---\n"
                    f"{facts_block}\n\n"
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
        previous_summary = state.get("rolling_summary", "")
        turn_index = sum(1 for m in history if getattr(m, "type", "") == "human")

        updated_summary = update_rolling_summary(
            summarize_llm=summarize_llm,
            existing_summary=previous_summary,
            facts_history=facts_history,
            questions_history=questions_history,
            turn_index=turn_index,
        )
        return {"rolling_summary": updated_summary}

    return summarize_memory_node