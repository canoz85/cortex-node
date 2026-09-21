import json
import os
from pathlib import Path


def save_raw_llm(worker: str, section: str, value: object, *, execution_id: str | None = None) -> None:
    file_path = os.getenv("CORTEX_RAW_LLM_FILE")

    if not file_path:
        return

    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # log_planner/log_finalizer may already have converted
        # structured values to JSON strings.
        # Convert those strings back to structured JSON for the file.
        file_value = value

        if isinstance(value, str):
            try:
                file_value = json.loads(value)
            except json.JSONDecodeError:
                # Normal prompt/text content: keep as string.
                pass

        entry = {
            "execution_id": execution_id,
            "worker": worker,
            "section": section,
            "value": value,
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    except Exception:
        # Debug logging must never break Cortex.
        pass