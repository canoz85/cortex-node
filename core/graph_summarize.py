
import json
import re
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from core.state import AgentState
from core.graph_constants import MAX_SUMMARY_CHARS, RECENT_MESSAGE_WINDOW
from core.graph_messages import recent_messages


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

def create_summarize_memory_node(
    *,
    summarize_llm: ChatOllama):

    
    def _clip_summary(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."


    def _extract_json_object(text: str) -> str | None:
        compact = (text or "").strip()
        if not compact:
            return None

        if compact.startswith("```"):
            compact = re.sub(r"^```(?:json)?\\s*", "", compact)
            compact = re.sub(r"\\s*```$", "", compact).strip()

        if compact.startswith("{") and compact.endswith("}"):
            return compact

        first = compact.find("{")
        last = compact.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        return compact[first : last + 1]


    def _normalize_summary_payload(payload: dict | None) -> dict[str, list[str]]:
        normalized = {"facts": [], "open_questions": []}
        if not isinstance(payload, dict):
            return normalized

        for key in ("facts", "open_questions"):
            value = payload.get(key)
            if not isinstance(value, list):
                continue

            cleaned: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                entry = item.strip()
                if not entry:
                    continue
                if entry not in cleaned:
                    cleaned.append(entry)

            normalized[key] = cleaned

        return normalized


    def _parse_summary_payload(text: str) -> dict[str, list[str]] | None:
        json_blob = _extract_json_object(text)
        if not json_blob:
            return None

        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return None

        return _normalize_summary_payload(parsed)


    def _payload_to_summary_text(payload: dict[str, list[str]]) -> str:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


    def _recent_user_statements(recent_history: list) -> list[str]:
        statements: list[str] = []
        for message in recent_history:
            message_type = getattr(message, "type", "")
            if message_type != "human":
                continue

            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            if content not in statements:
                statements.append(content)
        return statements


    def _fallback_summary(existing_summary: str, recent_history: list) -> str:
        """Return a summary when the LLM fails to produce a valid payload.

        The original implementation appended every user statement to the
        ``facts`` list, which caused non‑fact statements (e.g. greetings or
        questions) to appear in the summary.  To honour the system prompt
        that requires *only* permanent personal details, we now ignore
        recent user statements entirely and simply return the existing
        summary unchanged.
        """
        # Preserve the existing summary if it is already a valid JSON
        # payload.  If it is empty or malformed, return an empty payload.
        parsed = _parse_summary_payload(existing_summary)
        if parsed is None:
            return _clip_summary(_payload_to_summary_text({"facts": [], "open_questions": []}))
        return _clip_summary(_payload_to_summary_text(parsed))
    
    def update_rolling_summary(
        summarize_llm: ChatOllama,
        existing_summary: str,
        recent_history: list,
    ) -> str:
        if not recent_history and not existing_summary:
            return ""

        recent_history = [m for m in recent_history if not isinstance(m, ToolMessage)]

        # summarization_prompt = (
        #     "Extract durable memory from conversation messages.\n"
        #     "Output ONLY a single JSON object with this exact schema:\n"
        #     "{\"facts\":[\"...\"],\"open_questions\":[\"...\"]}\n"
        #     "Rules:\n"
        #     "1) facts MUST be exact user statements copied verbatim from history\n"
        #     "2) open_questions must be subset of user statements that are unresolved questions\n"
        #     "3) no paraphrasing, no added commentary, no markdown, no code fences\n"
        #     "4) if nothing applies, return empty arrays\n"
        # )

        # summarization_prompt = (
        #     "Extract durable memory from the recent conversation messages.\n"
        #     "Output ONLY a single JSON object with this exact schema:\n"
        #     "{\"facts\":[\"...\"],\"open_questions\":[\"...\"]}\n"
        #     "Rules:\n"
        #     "1) facts MUST be long-term personal profile details (e.g., name, location, job, core preferences) "
        #     "copied verbatim as exact user statements from the history.\n"
        #     "2) Do NOT extract temporary chit-chat, math equations, code logic, or transient inquiries (e.g., '2 + 2', 'hello', 'test').\n"
        #     "3) open_questions must be a subset of user statements representing unresolved, major personal needs or goals.\n"
        #     "4) no paraphrasing, no added commentary, no markdown, no code fences.\n"
        #     "5) If the recent messages contain no new durable profile data or important unresolved questions, "
        #     "leave the arrays completely empty or preserve the existing summary exactly as it was.\n"
        # )

        # summarization_prompt = (
        #     "You are the memory layer of an AI assistant. Analyze the recent conversation turn "
        #     "to update the long-term profile summary of the user.\n"
        #     "Output ONLY a single JSON object with this exact schema:\n"
        #     "{\"facts\":[\"...\"],\"open_questions\":[\"...\"]}\n"
        #     "Rules:\n"
        #     "1) facts MUST be new, clear, declarative sentences summarizing permanent personal details "
        #     "about the user (e.g., 'User name is Can', 'User was born in Mersin').\n"
        #     "2) CRITICAL: Do NOT extract user questions, math, or terminal commands as facts. "
        #     "If the user asks a question (like 'what is my brother name') and the AI response does not answer it, "
        #     "this is NOT a fact. Do not add it to the facts array.\n"
        #     "3) open_questions should only capture major, unresolved long-term user goals or project needs, "
        #     "NOT simple trivia questions the user is testing you with.\n"
        #     "4) No paraphrasing the schema, no added commentary, no markdown, no code fences.\n"
        #     "5) If the recent turn contains no new profile information, leave the arrays completely empty "
        #     "or return the existing summary exactly as it was.\n"
        # )

        summarization_prompt = (
            "CRITICAL: You are an internal background system database layer, NOT a chat assistant. "
            "Do NOT talk to a user. Do NOT say 'Hello', 'Sure', or 'Feel free to ask'. "
            "Your output must ONLY be raw JSON text.\n\n"
            "Analyze the recent conversation turn to update the long-term profile summary of the user.\n"
            "Output ONLY a single JSON object with this exact schema:\n"
            "{\"facts\":[\"...\"],\"open_questions\":[\"...\"]}\n"
            "Rules:\n"
            "1) facts MUST be new, clear, declarative sentences summarizing permanent personal details "
            "about the user (e.g., 'User name is Can', 'User was born in Mersin').\n"
            "2) CRITICAL: Do NOT extract user questions, math, or terminal commands as facts.\n"
            "3) open_questions should only capture major, unresolved long-term user goals or project needs.\n"
            "4) Absolutely no paraphrasing the schema, no added commentary, no markdown, no code fences.\n"
            "5) If the recent turn contains no new profile information, return the 'Existing Summary' exactly as it was.\n"
            "STOP: Check your output. If it contains conversational language, you have failed the task. Output ONLY JSON."
        )

        summary_messages = [
            SystemMessage(content=f"{summarization_prompt}\n\n"
                                f"Existing summary:\n{existing_summary or '(none)'}\n\n"
                                f"--- START RECENT CONVERSATION TURN TO ANALYZE ---"),
            *recent_history,
            SystemMessage(content="--- END RECENT CONVERSATION TURN ---\n"
                                "Analyze the turn above and output the JSON now:")
        ]

        # summary_messages = [
        #     SystemMessage(content=f"{summarization_prompt}\n\n"
        #                 f"Existing summary:\n{existing_summary or '(none)'}"),
        #     *recent_history,
        # ]
        response = summarize_llm.invoke(summary_messages)
        text = str(getattr(response, "content", "") or "").strip()
        if not text:
            return _fallback_summary(existing_summary or "", recent_history)

        parsed = _parse_summary_payload(text)
        if parsed is None:
            return _fallback_summary(existing_summary or "", recent_history)
        return _clip_summary(_payload_to_summary_text(parsed))

    
    def summarize_memory_node(state: AgentState):
        """Summarize the conversation history and update the rolling summary."""
        history = state.get("messages", [])
        recent_history = recent_messages(history, RECENT_MESSAGE_WINDOW)
        previous_summary = state.get("rolling_summary", "")
        updated_summary = update_rolling_summary(
            summarize_llm=summarize_llm,
            existing_summary=previous_summary,
            recent_history=recent_history,
        )

        return {
            "rolling_summary": updated_summary,
        }

    return summarize_memory_node