# Example Knowledge Rules

This folder demonstrates a simple RAG knowledge base for code generation.

When the user asks for code:
- Prefer the requested language and existing project style.
- Reuse examples from the knowledge base when they are relevant.
- Keep generated code minimal and testable.

When the user asks for a CLI:
- Use `argparse` in Python.
- Keep the entry point in `main()`.
- Print concise startup information.
