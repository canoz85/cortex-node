"""Standalone Ollama comparison for analysis of one unstructured financial note.

Run with: python benchmarks/financial_note_ollama.py
Results are written to benchmarks/results/financial-note-<timestamp>.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODELS = ("devstral-small-2:24b", "qwen3.8:27b", "gemma4:26b")
EVIDENCE = """2026-01-03
Akbank vadeli hesap açıldı.
500.000 TL yatırıldı.
Faiz oranı %42.

2026-01-18
Batu okul için 18.500 TL ödeme.
Hangi hesaptan çıktığı not edilmemiş.

Şubat başı
Akbank vadeden dönen para: 515.200 TL.
60.000 TL ayrıca hesaba eklendi.
Not: "eklenmedi" yazılmış ama toplam bakiye hesabında dahil edilmiş görünüyor.

2026-02-10
Dayıma 25.000 TL gönderildi.

Antalya
Yaklaşık 12.000 TL harcama.
Tarih yazılmamış.

Mart
Akbank faiz oranı %39'a düştü.
Başka bankada %44 teklif görüldü.
Para taşınmadı."""

PROMPT = f"""Analyze the supplied evidence only.

Build the most plausible chronological financial history.
Identify inconsistencies, unresolved questions, and financially relevant missed opportunities.

Do not invent:
- dates,
- recipients,
- purposes,
- account sources,
- amounts.

Keep explicit facts separate from inference.
If something is unknown, say it is unknown.

Evidence:
{EVIDENCE}"""

CHECKLIST = (
    'Did it incorrectly attach 2026-01-18 to the "Şubat başı" event?',
    'Did it turn "Dayıma" into another purpose such as alimony?',
    'Did it invent a date for the Antalya expense?',
    'Did it reverse the meaning of the 60,000 TL balance note?',
)


def query_model(model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.load(response)
        error = None
        text = data.get("message", {}).get("content", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        data = {}
        text = ""
        error = str(exc)
    elapsed = time.perf_counter() - started
    return {
        "model": model,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": data.get("prompt_eval_count"),
        "response_tokens": data.get("eval_count"),
        "response": text,
        "error": error,
        "manual_review_checklist": list(CHECKLIST),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(__file__).resolve().parent / "results" / f"financial-note-{timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Shared prompt/evidence; reasoning disabled where supported (think=false).\nSaved: {output_path}\n")
    with output_path.open("w", encoding="utf-8") as output:
        for model in MODELS:
            result = query_model(model)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"=== {model} ===")
            print(f"Elapsed: {result['elapsed_seconds']} s")
            print(f"Tokens: prompt={result['prompt_tokens']}, response={result['response_tokens']}")
            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                print(result["response"])
            print("Manual review:")
            for question in CHECKLIST:
                print(f"- [ ] {question}")
            print()


if __name__ == "__main__":
    main()
