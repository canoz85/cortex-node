import argparse
import json
import os
from pathlib import Path
import sys

from langchain_core.messages import messages_from_dict, messages_to_dict

from core.application_session import ApplicationSession, bounded_recent_conversation
from core.conversation_compaction import CompactionLimits, compact_recent_conversation
from core.conversation_memory_updater import LLMMemoryUpdater, build_memory_update_request
from core.memory import ConversationMemory
from core.memory.policy import merge_memory
from core.memory.terminal import extract_memory_update
from core.planner_memory import project_planner_memory
from core.logging_utils import configure_logging, get_logger

from core.graph import build_app, run_prompt
from core.runtime.gpu_resources import GpuResourceMode, GpuResourcePolicy


DEFAULT_SETTINGS = {
    "workspace": "workspace",
    "knowledge_dir": "knowledge",
    "model": "gemma4:26b", #"gemma4:26b", #"qwen2.5-coder:14b", #
    "model_planner": "gpt-oss:20b", # qwen2.5:7b
    "embedding_model": "nomic-embed-text",
    "rag_top_k": 4,
    "raw_llm": True,
    "show_summary": False,
    "log_level": "INFO",
    "json_logs": False,
    "gpu_telemetry": True,
    "gpu_handoff": True,
    "session_file": ".cortex_session.json",
    "memory_llm_enabled": True,
    "memory_compaction_target_turns": 12,
    "memory_minimum_verbatim_turns": 8,
}

def _load_config_file(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix not in {".json"}:
        raise ValueError("Only JSON config files are supported (use .json)")
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Config root must be an object")
    return parsed


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _build_settings(args: argparse.Namespace) -> dict:
    settings = dict(DEFAULT_SETTINGS)

    env_overrides = {
        "workspace": os.getenv("CORTEX_WORKSPACE"),
        "knowledge_dir": os.getenv("CORTEX_KNOWLEDGE_DIR"),
        "model": os.getenv("CORTEX_MODEL"),
        "model_planner": os.getenv("CORTEX_MODEL_PLANNER"),
        "embedding_model": os.getenv("CORTEX_EMBEDDING_MODEL"),
        "rag_top_k": os.getenv("CORTEX_RAG_TOP_K"),
        "log_level": os.getenv("CORTEX_LOG_LEVEL"),
        "raw_llm": _env_bool("CORTEX_RAW_LLM"),
        "show_summary": _env_bool("CORTEX_SHOW_SUMMARY"),
        "json_logs": _env_bool("CORTEX_JSON_LOGS"),
        "gpu_telemetry": _env_bool("CORTEX_GPU_TELEMETRY"),
        "gpu_handoff": _env_bool("CORTEX_GPU_HANDOFF"),
        "memory_llm_enabled": _env_bool("CORTEX_MEMORY_LLM_ENABLED"),
    }
    for key, value in env_overrides.items():
        if value is not None:
            settings[key] = value

    if args.config:
        file_settings = _load_config_file(args.config)
        for key in DEFAULT_SETTINGS:
            if key in file_settings:
                settings[key] = file_settings[key]

    cli_overrides = {
        "workspace": args.workspace,
        "knowledge_dir": args.knowledge_dir,
        "model": args.model,
        "model_planner": args.model_planner,
        "embedding_model": args.embedding_model,
        "rag_top_k": args.rag_top_k,
        "raw_llm": args.raw_llm,
        "show_summary": args.show_summary,
        "log_level": args.log_level,
        "json_logs": args.json_logs,
        "gpu_telemetry": args.gpu_telemetry,
        "gpu_handoff": args.gpu_handoff,
        "memory_llm_enabled": args.memory_llm_enabled,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            settings[key] = value

    settings["rag_top_k"] = int(settings["rag_top_k"])
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CortexNode local-first AI software agent")

    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON config file path. Merged as: defaults -> env -> config -> CLI.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--workspace",
        default=None,
        help="Sandbox directory used by file and execution tools",
    )
    runtime_group.add_argument(
        "--knowledge-dir",
        default=None,
        help="Folder containing .md and .json knowledge sources for RAG",
    )
    runtime_group.add_argument(
        "--model",
        default=None,
        help="Ollama model name",
    )

    runtime_group.add_argument(
        "--model-planner",
        default=None,
        help="Ollama model name for the planner (can be same as --model)",
    )

    runtime_group.add_argument(
        "--embedding-model",
        default=None,
        help="Ollama embedding model used for knowledge retrieval",
    )
    runtime_group.add_argument(
        "--rag-top-k",
        type=int,
        default=None,
        help="Top-k knowledge chunks to retrieve per query.",
    )

    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--prompt",
        default="",
        help="Single prompt to run. If omitted, interactive mode starts.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--raw-llm",
        dest="raw_llm",
        action="store_true",
        default=None,
        help="Enable raw LLM responses (debug view) in red/italic ANSI output.",
    )
    output_group.add_argument(
        "--no-raw-llm",
        dest="raw_llm",
        action="store_false",
        help="Disable raw LLM response output.",
    )
    output_group.add_argument(
        "--show-summary",
        dest="show_summary",
        action="store_true",
        default=None,
        help="Show rolling summary output in blue after each run.",
    )
    output_group.add_argument(
        "--no-show-summary",
        dest="show_summary",
        action="store_false",
        help="Disable rolling summary output.",
    )
    output_group.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    output_group.add_argument(
        "--json-logs",
        dest="json_logs",
        action="store_true",
        default=None,
        help="Emit structured JSON logs.",
    )
    output_group.add_argument(
        "--no-json-logs",
        dest="json_logs",
        action="store_false",
        help="Emit plain text logs.",
    )

    output_group.add_argument(
        "--gpu-telemetry",
        dest="gpu_telemetry",
        action="store_true",
        default=None,
        help="Observe GPU/VRAM around Planner, Brain, tool, summary, and async poll operations.",
    )
    output_group.add_argument(
        "--no-gpu-telemetry",
        dest="gpu_telemetry",
        action="store_false",
        help="Disable observe-only GPU runtime telemetry.",
    )
    output_group.add_argument(
        "--gpu-handoff",
        dest="gpu_handoff",
        action="store_true",
        default=None,
        help="Require verified Ollama/ComfyUI single-GPU handoff.",
    )
    output_group.add_argument(
        "--no-gpu-handoff",
        dest="gpu_handoff",
        action="store_false",
        help="Disable active Ollama/ComfyUI GPU handoff.",
    )

    output_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Disable resuming from the session file.",
    )
    output_group.add_argument(
        "--memory-llm",
        dest="memory_llm_enabled",
        action="store_true",
        default=None,
        help="Enable optional structured durable-memory extraction after completed turns.",
    )
    output_group.add_argument(
        "--no-memory-llm",
        dest="memory_llm_enabled",
        action="store_false",
        help="Disable optional durable-memory LLM extraction.",
    )


    return parser.parse_args()


def create_optional_memory_updater(settings: dict) -> LLMMemoryUpdater | None:
    if not settings["memory_llm_enabled"]:
        return None
    from langchain_ollama import ChatOllama
    from core.memory_provider import LangChainMemoryProposalProvider

    llm = ChatOllama(
        model=str(settings["model_planner"]), temperature=0,
        num_predict=512, client_kwargs={"timeout": 30},
    )
    return LLMMemoryUpdater(LangChainMemoryProposalProvider(llm))


def load_session(session_path: str) -> ApplicationSession:
    """Load application context; reject invalid memory independently of history."""
    session_path = Path(session_path)
    if session_path.exists():
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("session root must be an object")
            version = data.get("session_schema_version", 1)
            if version not in (1, 2):
                raise ValueError(f"unsupported session schema version: {version}")
            try:
                if version == 2 and "conversation_memory" not in data:
                    raise ValueError("conversation_memory is missing")
                memory = ConversationMemory.model_validate_json(
                    json.dumps(data.get("conversation_memory", {}))
                )
            except Exception as exc:
                print(f"[Warning] Invalid conversation memory; using empty memory: {exc}", file=sys.stderr)
                memory = ConversationMemory()
            try:
                serialized = data.get("recent_conversation", data.get("history", []))
                if not isinstance(serialized, list):
                    raise ValueError("recent conversation must be a list")
                recent = bounded_recent_conversation(messages_from_dict(serialized))
            except Exception as exc:
                print(f"[Warning] Invalid recent conversation; using empty history: {exc}", file=sys.stderr)
                recent = ()
            legacy_summary = data.get("rolling_summary", "")
            if not isinstance(legacy_summary, str):
                legacy_summary = ""
            known_turns = [len([m for m in recent if m.type == "human"])]
            known_turns.extend(fact.source.turn_index for fact in memory.facts)
            known_turns.extend(question.source_turn_index for question in memory.questions)
            if memory.continuity.source_turn_index is not None:
                known_turns.append(memory.continuity.source_turn_index)
            count = data.get("completed_turn_count", max(known_turns))
            if not isinstance(count, int) or count < max(known_turns):
                print("[Warning] Invalid completed turn count; using known turns", file=sys.stderr)
                count = max(known_turns)
            start = data.get("maintenance_start_turn")
            end = data.get("maintenance_end_turn")
            if not (
                (start is None and end is None)
                or (isinstance(start, int) and isinstance(end, int)
                    and 1 <= start <= end <= count)
            ):
                print("[Warning] Invalid memory maintenance window; ignoring it", file=sys.stderr)
                start = end = None
            enrichment_status = data.get("last_enrichment_status", "not_run")
            if enrichment_status not in {
                "not_run", "disabled", "accepted", "no_op", "failed_safe",
            }:
                enrichment_status = "not_run"
            print(f"[Info] Resumed previous session from {session_path}")
            return ApplicationSession(
                memory, recent, legacy_summary, count, start, end, enrichment_status,
            )
        except Exception as e:
            print(f"[Warning] Failed to load session file: {e}", file=sys.stderr)
    return ApplicationSession()

def save_session(
    session_path: str,
    session: ApplicationSession,
    debug: dict | None = None,
):
    """Save conversation state and optional runtime debug information."""

    session_path = Path(session_path)

    try:
        session_path.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "session_schema_version": 2,
            "conversation_memory": session.conversation_memory.model_dump(mode="json"),
            "recent_conversation": messages_to_dict(
                list(bounded_recent_conversation(session.recent_conversation))
            ),
            # Temporary old-graph compatibility; not the memory representation.
            "rolling_summary": session.legacy_rolling_summary,
            "completed_turn_count": session.completed_turn_count,
            "maintenance_start_turn": session.maintenance_start_turn,
            "maintenance_end_turn": session.maintenance_end_turn,
            "last_enrichment_status": session.last_enrichment_status,
        }

        if debug is not None:
            session_data["debug"] = debug

        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(
                session_data,
                f,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        print(f"\n[Info] Session state saved to {session_path}")

    except Exception as e:
        print(
            f"\n[Warning] Failed to save session state: {e}",
            file=sys.stderr,
        )

def main():
    args = parse_args()
    settings = _build_settings(args)
    configure_logging(level=str(settings["log_level"]), json_logs=bool(settings["json_logs"]))
    logger = get_logger(__name__)

    app = build_app(
        workspace_dir=str(settings["workspace"]),
        model=str(settings["model"]),
        model_planner=str(settings["model_planner"]),
        knowledge_dir=str(settings["knowledge_dir"]),
        embedding_model=str(settings["embedding_model"]),
        rag_top_k=int(settings["rag_top_k"]),
        show_raw_llm=bool(settings["raw_llm"]),
        gpu_resource_policy=GpuResourcePolicy(
            mode=(
                GpuResourceMode.OBSERVE_ONLY
                if bool(settings["gpu_telemetry"])
                else GpuResourceMode.DISABLED
            ),
            handoff_enabled=bool(settings["gpu_handoff"]),
        ),
    )

    print("--- CortexNode initialized ---")
    print(f"Model: {settings['model']}")
    print(f"Planner Model: {settings['model_planner']}")
    print(f"Sandbox: {settings['workspace']}")
    print(f"Knowledge: {settings['knowledge_dir']}")
    print(
        "GPU telemetry: "
        f"{'observe-only' if settings['gpu_telemetry'] else 'disabled'}"
    )
    print(
        "GPU handoff: "
        f"{'single-GPU verified' if settings['gpu_handoff'] else 'disabled'}"
    )
    logger.info(
        "CortexNode initialized",
        extra={
            "event_name": "app_initialized",
            "model": settings["model"],
            "planner_model": settings["model_planner"],
            "workspace": settings["workspace"],
            "knowledge_dir": settings["knowledge_dir"],
            "session_file": settings["session_file"],
            "rag_top_k": settings["rag_top_k"],
            "gpu_telemetry": settings["gpu_telemetry"],
            "gpu_handoff": settings["gpu_handoff"],
        },
    )

    session = load_session(settings["session_file"]) if args.resume else ApplicationSession()
    compaction_limits = CompactionLimits(
        target_turns=int(settings["memory_compaction_target_turns"]),
        minimum_verbatim_turns=int(settings["memory_minimum_verbatim_turns"]),
    )
    try:
        memory_updater = create_optional_memory_updater(settings)
    except Exception as exc:
        logger.warning("Optional memory updater unavailable: %s", type(exc).__name__)
        memory_updater = None

    def complete_turn(user_prompt: str) -> None:
        nonlocal session
        terminal_evidence = []
        history, legacy_summary = run_prompt(
            app,
            user_prompt,
            history=list(session.recent_conversation),
            rolling_summary=session.legacy_rolling_summary,
            show_summary=bool(settings["show_summary"]),
            completed_turn_evidence=terminal_evidence,
            turn_index=session.completed_turn_count + 1,
            planner_memory_context=project_planner_memory(
                session.conversation_memory,
                current_turn_index=session.completed_turn_count + 1,
            ),
        )
        memory = session.conversation_memory
        completed_count = session.completed_turn_count
        maintenance_start = session.maintenance_start_turn
        maintenance_end = session.maintenance_end_turn
        enrichment_status = session.last_enrichment_status
        maintained = False
        if terminal_evidence:
            completed_count += 1
            try:
                update = extract_memory_update(terminal_evidence[0])
                memory = merge_memory(memory, update)
                maintained = True
            except Exception as exc:
                logger.warning("Deterministic memory update failed: %s", type(exc).__name__)
            if maintained:
                if settings["memory_llm_enabled"]:
                    if memory_updater is None:
                        enrichment_status = "failed_safe"
                    else:
                        logger.info("Memory updater invoked")
                        try:
                            request = build_memory_update_request(terminal_evidence[0], memory)
                            proposal = memory_updater.propose(request)
                            if proposal.facts:
                                memory = merge_memory(memory, proposal)
                                enrichment_status = "accepted"
                                logger.info("Memory proposal accepted: facts=%s", len(proposal.facts))
                            else:
                                enrichment_status = "no_op"
                        except Exception as exc:
                            enrichment_status = "failed_safe"
                            logger.warning(
                                "Memory proposal rejected: %s: %s",
                                type(exc).__name__,
                                exc,
                            )
                else:
                    enrichment_status = "disabled"
                if maintenance_end == session.completed_turn_count and maintenance_start is not None:
                    maintenance_end = completed_count
                else:
                    maintenance_start = maintenance_end = completed_count
        recent = bounded_recent_conversation(history)
        if maintained:
            compacted = compact_recent_conversation(
                recent, completed_turn_count=completed_count,
                maintenance_start_turn=maintenance_start,
                maintenance_end_turn=maintenance_end,
                questions=memory.questions, limits=compaction_limits,
            )
            removed = sum(message.type == "human" for message in recent) - sum(
                message.type == "human" for message in compacted
            )
            if removed:
                logger.info("Conversation turns compacted: count=%s", removed)
            recent = compacted
        session = ApplicationSession(
            conversation_memory=memory,
            recent_conversation=recent,
            legacy_rolling_summary=legacy_summary,
            completed_turn_count=completed_count,
            maintenance_start_turn=maintenance_start,
            maintenance_end_turn=maintenance_end,
            last_enrichment_status=enrichment_status,
        )
        save_session(settings["session_file"], session)

    try:

        if args.prompt:
            complete_turn(args.prompt)
            return

        print("Interactive mode: type 'exit' to quit.")
        while True:
            try:
                user_prompt = input("\nYou> ").strip()
                if user_prompt.lower() in {"exit", "quit"}:
                    print("Stopping CortexNode.")
                    break
                if not user_prompt:
                    continue
                complete_turn(user_prompt)
            except KeyboardInterrupt:
                print("\nStopping CortexNode.")
                break
    finally:
        save_session(settings["session_file"], session)


if __name__ == "__main__":
    main()
