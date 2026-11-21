"""Display limits for captured Brain evidence. All sizes are serialized JSON characters."""

import json
from itertools import islice

MAX_CURRENT_ATTEMPT_RECORDS = 32
MAX_VALUE_CHARS = 2048
# Shared across arguments, message, data, and artifacts of all visible records.
PAYLOAD_CHARS = 96000


def _size(value):
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def fits_display_budget(observation, budget):
    """Measure the same JSON emitted in the prompt, stopping once it cannot fit."""
    for chunk in json.JSONEncoder(ensure_ascii=True, separators=(",", ":")).iterencode(observation):
        budget -= len(chunk)
        if budget < 0:
            return False
    return True


def compact_value(value, budget):
    """Keep small JSON values verbatim. Nest previews at their original field paths."""
    if isinstance(value, str):
        if len(value) <= 512 and _size(value) <= budget:
            return value
        preview = value[:192]
        marker = {"display_compacted": True, "type": "string", "captured_chars": len(value)}
        while _size({**marker, "preview": preview}) > budget and preview:
            preview = preview[:len(preview) // 2]
        return {**marker, "preview": preview}

    if not isinstance(value, (dict, list, tuple)):
        return value

    mapping = isinstance(value, dict)
    preview = {} if mapping else []
    marker = {"display_compacted": True, "type": "object" if mapping else "array",
              "captured_count": len(value), "preview": preview}
    # Keep room for the marker when omission is already certain. Otherwise
    # allow even a deep, small structure to use the entire allowance intact.
    remaining = budget - (_size(marker) if len(value) > 32 else 2)
    entries = islice(value.items() if mapping else enumerate(value), 32)
    if mapping:
        # Leave room for short siblings (counts, cursors, status) before bodies.
        entries = sorted(entries, key=lambda pair: isinstance(pair[1], (dict, list, tuple))
                         or (isinstance(pair[1], str) and len(pair[1]) > 128))
    for key, item in entries:
        cost = (_size(key) + 1 if mapping else 0) + bool(preview)
        available = remaining - cost
        if available < 2:
            break
        child = compact_value(item, available)
        remaining = available - _size(child)
        if remaining < 0:
            break
        if mapping:
            preview[key] = child
        else:
            preview.append(child)
    if len(preview) == len(value):
        return preview
    while preview and _size(marker) > budget:
        if mapping:
            preview.popitem()
        else:
            preview.pop()
    return marker
