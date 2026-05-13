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
        "--model",
        default="qwen2.5:7b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Single prompt to run. If omitted, interactive mode starts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(workspace_dir=args.workspace, model=args.model)

    print("--- CortexNode initialized ---")
    print(f"Model: {args.model}")
    print(f"Sandbox: {args.workspace}")

    if args.prompt:
        run_prompt(app, args.prompt)
        return

    print("Interactive mode: type 'exit' to quit.")
    history = []
    while True:
        user_prompt = input("\nYou> ").strip()
        if user_prompt.lower() in {"exit", "quit"}:
            print("Stopping CortexNode.")
            break
        if not user_prompt:
            continue
        history = run_prompt(app, user_prompt, history=history)


if __name__ == "__main__":
    main()