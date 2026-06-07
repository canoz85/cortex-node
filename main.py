import argparse
import json
import os
from pathlib import Path

from core.logging_utils import configure_logging, get_logger

from core.graph import build_app, run_prompt


DEFAULT_SETTINGS = {
    "workspace": "workspace",
    "knowledge_dir": "knowledge",
    "model": "qwen2.5-coder:14b",
    "embedding_model": "nomic-embed-text",
    "rag_top_k": 4,
    "raw_llm": True,
    "show_summary": False,
    "log_level": "INFO",
    "json_logs": False,
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
        "embedding_model": os.getenv("CORTEX_EMBEDDING_MODEL"),
        "rag_top_k": os.getenv("CORTEX_RAG_TOP_K"),
        "log_level": os.getenv("CORTEX_LOG_LEVEL"),
        "raw_llm": _env_bool("CORTEX_RAW_LLM"),
        "show_summary": _env_bool("CORTEX_SHOW_SUMMARY"),
        "json_logs": _env_bool("CORTEX_JSON_LOGS"),
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
        "embedding_model": args.embedding_model,
        "rag_top_k": args.rag_top_k,
        "raw_llm": args.raw_llm,
        "show_summary": args.show_summary,
        "log_level": args.log_level,
        "json_logs": args.json_logs,
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
    return parser.parse_args()


def main():
    args = parse_args()
    settings = _build_settings(args)
    configure_logging(level=str(settings["log_level"]), json_logs=bool(settings["json_logs"]))
    logger = get_logger(__name__)

    app = build_app(
        workspace_dir=str(settings["workspace"]),
        model=str(settings["model"]),
        knowledge_dir=str(settings["knowledge_dir"]),
        embedding_model=str(settings["embedding_model"]),
        rag_top_k=int(settings["rag_top_k"]),
    )

    print("--- CortexNode initialized ---")
    print(f"Model: {settings['model']}")
    print(f"Sandbox: {settings['workspace']}")
    print(f"Knowledge: {settings['knowledge_dir']}")
    logger.info(
        "CortexNode initialized",
        extra={
            "event_name": "app_initialized",
            "model": settings["model"],
            "workspace": settings["workspace"],
            "knowledge_dir": settings["knowledge_dir"],
            "rag_top_k": settings["rag_top_k"],
        },
    )

    if args.prompt:
        run_prompt(
            app,
            args.prompt,
            show_raw_llm=bool(settings["raw_llm"]),
            show_summary=bool(settings["show_summary"]),
        )
        return

    print("Interactive mode: type 'exit' to quit.")
    history = []
    rolling_summary = ""
    while True:
        user_prompt = input("\nYou> ").strip()
        if user_prompt.lower() in {"exit", "quit"}:
            print("Stopping CortexNode.")
            break
        if not user_prompt:
            continue
        history, rolling_summary = run_prompt(
            app,
            user_prompt,
            history=history,
            rolling_summary=rolling_summary,
            show_raw_llm=bool(settings["raw_llm"]),
            show_summary=bool(settings["show_summary"]),
        )


if __name__ == "__main__":
    main()
