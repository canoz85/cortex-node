"""Finalizer stdout diagnostics, gated by the same show_raw_llm flag as Brain."""

import json
import re

from core.debug import save_raw_llm


def log_finalizer(section: str, value: object, *, enabled: bool = True, execution_id: str | None = None,
) -> None:
    # Sanitize only the display copy; never inspect provider configuration or env.
    # Avoid arbitrary object reprs, which can expose transport credentials.
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
        text = re.sub(r'(?i)\bBearer\s+[^\s"\']+', 'Bearer [REDACTED]', text)
        text = re.sub(
            r'(?i)([\"\']?(?:api[_-]?key|password|passwd|secret|access[_-]?token|'
            r'refresh[_-]?token|authorization|credential)[\"\']?\s*[:=]\s*)'
            r'(?:"[^"\n]*"|\'[^\'\n]*\'|[^\s,;}]+)',
            r'\1[REDACTED]', text,
        )
        text = re.sub(r'(?i)(https?://)[^\s/@]+:[^\s/@]+@', r'\1[REDACTED]@', text)
        text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
                      '[REDACTED]', text, flags=re.DOTALL)

        # File logging is independent from console logging
        save_raw_llm(
            "finalizer",
            section,
            text,
            execution_id=execution_id
        )

        if not enabled:
            return
    
        print(f"[finalizer:{section}]\n{text}")
    except Exception:
        # Diagnostics must not turn a successful invocation into a failure.
        pass
