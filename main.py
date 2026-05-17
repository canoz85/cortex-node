import argparse

from core.graph import build_app, run_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="CortexNode local-first AI software agent")
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Sandbox directory used by file and execution tools",
    )
    parser.add_argument(
        "--knowledge-dir",
        default="knowledge",
        help="Folder containing .md and .json knowledge sources for RAG",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model used for knowledge retrieval",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Single prompt to run. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--raw-llm",
        action="store_true",
        default=True,
        help="Show raw LLM responses (debug view) in red/italic ANSI output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(
        workspace_dir=args.workspace,
        model=args.model,
        knowledge_dir=args.knowledge_dir,
        embedding_model=args.embedding_model,
    )

    print("--- CortexNode initialized ---")
    print(f"Model: {args.model}")
    print(f"Sandbox: {args.workspace}")
    print(f"Knowledge: {args.knowledge_dir}")

    if args.prompt:
        run_prompt(app, args.prompt, show_raw_llm=args.raw_llm)
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
            show_raw_llm=args.raw_llm,
        )


if __name__ == "__main__":
    main()