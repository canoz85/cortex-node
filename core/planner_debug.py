"""Planner stdout diagnostics, gated by the same show_raw_llm flag as Brain."""

import json
import re

from core.debug import save_raw_llm


def log_planner(
    section: str,
    value: object,
    *,
    enabled: bool = True,
    execution_id: str | None = None,
) -> None:
    # Diagnostics must never break Planner execution.
    try:
        # Serialize temporarily only for sanitization.
        if isinstance(value, str):
            text = value
            structured = False
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
            structured = True

        text = re.sub(
            r'(?i)\bBearer\s+[^\s"\']+',
            'Bearer [REDACTED]',
            text,
        )

        text = re.sub(
            r'(?i)(["\']?(?:api[_-]?key|password|passwd|secret|access[_-]?token|'
            r'refresh[_-]?token|authorization|credential)["\']?\s*[:=]\s*)'
            r'(?:"[^"\n]*"|\'[^\'\n]*\'|[^\s,;}]+)',
            r'\1[REDACTED]',
            text,
        )

        text = re.sub(
            r'(?i)(https?://)[^\s/@]+:[^\s/@]+@',
            r'\1[REDACTED]@',
            text,
        )

        text = re.sub(
            r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
            '[REDACTED]',
            text,
            flags=re.DOTALL,
        )

        # Restore structured JSON when possible.
        file_value: object = text

        if structured:
            try:
                file_value = json.loads(text)
            except json.JSONDecodeError:
                file_value = text

        save_raw_llm(
            "planner",
            section,
            file_value,
            execution_id=execution_id,
        )

        if not enabled:
            return

        console_text = (
            text
            if isinstance(file_value, str)
            else json.dumps(
                file_value,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        print(f"[planner:{section}]\n{console_text}")

    except Exception:
        # Diagnostics must not turn a successful invocation into a failure.
        pass