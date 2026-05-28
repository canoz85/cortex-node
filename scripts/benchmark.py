import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkCase:
    case_id: str
    prompt: str
    expected_substrings: list[str]
    tags: list[str]


@dataclass
class BenchmarkResult:
    case_id: str
    mode: str
    success: bool
    latency_ms: float
    matched_substrings: list[str]
    missing_substrings: list[str]
    output_excerpt: str
    error: str


def load_cases(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark file must contain a JSON array")

    cases: list[BenchmarkCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case index {idx} must be an object")

        case_id = str(item.get("id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        expected = item.get("expected_substrings", [])
        tags = item.get("tags", [])

        if not case_id:
            raise ValueError(f"Case index {idx} is missing a non-empty 'id'")
        if not prompt:
            raise ValueError(f"Case '{case_id}' is missing a non-empty 'prompt'")
        if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
            raise ValueError(f"Case '{case_id}' has invalid 'expected_substrings'")
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            raise ValueError(f"Case '{case_id}' has invalid 'tags'")

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                prompt=prompt,
                expected_substrings=[x.strip() for x in expected if x.strip()],
                tags=[x.strip() for x in tags if x.strip()],
            )
        )

    return cases


def evaluate_case_output(case: BenchmarkCase, output: str) -> tuple[bool, list[str], list[str]]:
    normalized_output = (output or "").lower()
    matched: list[str] = []
    missing: list[str] = []

    for token in case.expected_substrings:
        if token.lower() in normalized_output:
            matched.append(token)
        else:
            missing.append(token)

    return len(missing) == 0, matched, missing


def run_case_live(case: BenchmarkCase, timeout: int) -> BenchmarkResult:
    command = [sys.executable, "main.py", "--prompt", case.prompt]
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        success, matched, missing = evaluate_case_output(case, output)
        if completed.returncode != 0:
            return BenchmarkResult(
                case_id=case.case_id,
                mode="live",
                success=False,
                latency_ms=elapsed_ms,
                matched_substrings=matched,
                missing_substrings=missing,
                output_excerpt=output[-1200:],
                error=f"Non-zero exit code: {completed.returncode}",
            )

        return BenchmarkResult(
            case_id=case.case_id,
            mode="live",
            success=success,
            latency_ms=elapsed_ms,
            matched_substrings=matched,
            missing_substrings=missing,
            output_excerpt=output[-1200:],
            error="",
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return BenchmarkResult(
            case_id=case.case_id,
            mode="live",
            success=False,
            latency_ms=elapsed_ms,
            matched_substrings=[],
            missing_substrings=case.expected_substrings,
            output_excerpt="",
            error=f"Timed out after {timeout} seconds",
        )


def run_case_dry(case: BenchmarkCase) -> BenchmarkResult:
    return BenchmarkResult(
        case_id=case.case_id,
        mode="dry",
        success=True,
        latency_ms=0.0,
        matched_substrings=[],
        missing_substrings=[],
        output_excerpt="Dry-run: case schema validated.",
        error="",
    )


def summarize(results: list[BenchmarkResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    avg_latency = round(sum(r.latency_ms for r in results) / total, 2) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        "avg_latency_ms": avg_latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CortexNode benchmark skeleton runner")
    parser.add_argument(
        "--cases",
        default="benchmarks/scenarios.json",
        help="Path to benchmark case JSON file",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/latest.json",
        help="Path to write JSON benchmark report",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Run only the first N cases (0 means all)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live prompts through main.py instead of dry-run schema validation",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per live benchmark case",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_path = Path(args.cases).resolve()
    output_path = Path(args.output).resolve()

    cases = load_cases(case_path)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise ValueError("No benchmark cases found")

    results: list[BenchmarkResult] = []
    for case in cases:
        result = run_case_live(case, timeout=args.timeout) if args.live else run_case_dry(case)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"[{status}] {result.case_id} ({result.mode}) {result.latency_ms}ms")
        if result.error:
            print(f"  error: {result.error}")

    summary = summarize(results)
    generated_at_utc = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "mode": "live" if args.live else "dry",
        "generated_at_utc": generated_at_utc,
        "git_sha": os.getenv("GITHUB_SHA", ""),
        "summary": summary,
        "results": [r.__dict__ for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Benchmark summary: "
        f"total={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} pass_rate={summary['pass_rate']}%"
    )
    print(f"Report written to: {output_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
