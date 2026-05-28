import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    expected_substrings: list[str]
    max_latency_ms: float
    weight_content: float
    weight_latency: float
    tags: list[str]


@dataclass
class EvalResult:
    case_id: str
    success: bool
    return_code: int
    latency_ms: float
    matched_substrings: list[str]
    missing_substrings: list[str]
    content_score: float
    latency_score: float
    total_score: float
    max_latency_ms: float
    output_excerpt: str
    error: str
    tags: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation dataset and generate scoring dashboard")
    parser.add_argument("--dataset", default="benchmarks/evaluation_dataset.json", help="Evaluation dataset JSON path")
    parser.add_argument("--output", default="benchmarks/results/evaluation-latest.json", help="Output JSON path")
    parser.add_argument(
        "--dashboard-md",
        default="benchmarks/results/evaluation-dashboard.md",
        help="Dashboard markdown output path",
    )
    parser.add_argument(
        "--dashboard-json",
        default="benchmarks/results/evaluation-dashboard.json",
        help="Dashboard chart-ready JSON output path",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Run only first N cases (0 means all)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per case in seconds")
    parser.add_argument(
        "--pass-score-threshold",
        type=float,
        default=0.6,
        help="Minimum total_score required for a live case to pass",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run prompts through main.py. If omitted, only dataset validation is performed.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dataset(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON array")

    cases: list[EvalCase] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case index {idx} must be an object")

        case_id = str(item.get("id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        expected = item.get("expected_substrings", [])
        tags = item.get("tags", [])
        weights = item.get("weights", {})

        if not case_id:
            raise ValueError(f"Case index {idx} has missing id")
        if not prompt:
            raise ValueError(f"Case '{case_id}' has missing prompt")
        if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
            raise ValueError(f"Case '{case_id}' has invalid expected_substrings")
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            raise ValueError(f"Case '{case_id}' has invalid tags")
        if not isinstance(weights, dict):
            raise ValueError(f"Case '{case_id}' has invalid weights")

        weight_content = _safe_float(weights.get("content", 0.8), 0.8)
        weight_latency = _safe_float(weights.get("latency", 0.2), 0.2)
        weight_sum = weight_content + weight_latency
        if weight_sum <= 0:
            raise ValueError(f"Case '{case_id}' weight sum must be > 0")

        # Normalize weights to avoid accidental misconfiguration.
        weight_content = weight_content / weight_sum
        weight_latency = weight_latency / weight_sum

        max_latency_ms = _safe_float(item.get("max_latency_ms", 10000), 10000)
        if max_latency_ms <= 0:
            raise ValueError(f"Case '{case_id}' max_latency_ms must be > 0")

        cases.append(
            EvalCase(
                case_id=case_id,
                prompt=prompt,
                expected_substrings=[x.strip() for x in expected if x.strip()],
                max_latency_ms=max_latency_ms,
                weight_content=weight_content,
                weight_latency=weight_latency,
                tags=[x.strip() for x in tags if x.strip()],
            )
        )

    return cases


def evaluate_output(case: EvalCase, output: str) -> tuple[list[str], list[str], float]:
    normalized_output = (output or "").lower()
    if not case.expected_substrings:
        return [], [], 1.0

    matched: list[str] = []
    missing: list[str] = []
    for token in case.expected_substrings:
        if token.lower() in normalized_output:
            matched.append(token)
        else:
            missing.append(token)

    content_score = len(matched) / len(case.expected_substrings)
    return matched, missing, round(content_score, 4)


def latency_score(latency_ms: float, max_latency_ms: float) -> float:
    if latency_ms <= max_latency_ms:
        return 1.0
    return round(max(0.0, max_latency_ms / latency_ms), 4)


def run_case_live(case: EvalCase, timeout_seconds: int, pass_score_threshold: float) -> EvalResult:
    command = [sys.executable, "main.py", "--prompt", case.prompt]
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        matched, missing, content = evaluate_output(case, output)
        lat_score = latency_score(elapsed_ms, case.max_latency_ms)
        total = round((content * case.weight_content) + (lat_score * case.weight_latency), 4)

        success = completed.returncode == 0 and total >= pass_score_threshold
        error = "" if completed.returncode == 0 else f"Non-zero exit code: {completed.returncode}"

        return EvalResult(
            case_id=case.case_id,
            success=success,
            return_code=completed.returncode,
            latency_ms=elapsed_ms,
            matched_substrings=matched,
            missing_substrings=missing,
            content_score=content,
            latency_score=lat_score,
            total_score=total,
            max_latency_ms=case.max_latency_ms,
            output_excerpt=output[-1200:],
            error=error,
            tags=case.tags,
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return EvalResult(
            case_id=case.case_id,
            success=False,
            return_code=124,
            latency_ms=elapsed_ms,
            matched_substrings=[],
            missing_substrings=case.expected_substrings,
            content_score=0.0,
            latency_score=0.0,
            total_score=0.0,
            max_latency_ms=case.max_latency_ms,
            output_excerpt="",
            error=f"Timed out after {timeout_seconds} seconds",
            tags=case.tags,
        )


def run_case_validate_only(case: EvalCase) -> EvalResult:
    return EvalResult(
        case_id=case.case_id,
        success=True,
        return_code=0,
        latency_ms=0.0,
        matched_substrings=[],
        missing_substrings=[],
        content_score=1.0,
        latency_score=1.0,
        total_score=1.0,
        max_latency_ms=case.max_latency_ms,
        output_excerpt="Validation-only run: case schema accepted.",
        error="",
        tags=case.tags,
    )


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    avg_latency_ms = round(sum(r.latency_ms for r in results) / total, 2) if total else 0.0
    avg_score = round(sum(r.total_score for r in results) / total, 4) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "avg_total_score": avg_score,
    }


def build_dashboard_markdown(summary: dict[str, Any], results: list[EvalResult], mode: str) -> str:
    lines = [
        "# Evaluation Dashboard",
        "",
        f"- mode: {mode}",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- pass_rate: {summary['pass_rate']:.2f}%",
        f"- avg_total_score: {summary['avg_total_score']:.4f}",
        f"- avg_latency_ms: {summary['avg_latency_ms']:.2f}",
        "",
        "## Per-case Scores",
        "",
        "| case_id | success | total_score | content_score | latency_score | latency_ms | max_latency_ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in results:
        lines.append(
            f"| {item.case_id} | {str(item.success).lower()} | {item.total_score:.4f} | "
            f"{item.content_score:.4f} | {item.latency_score:.4f} | {item.latency_ms:.2f} | {item.max_latency_ms:.2f} |"
        )

    return "\n".join(lines) + "\n"


def build_dashboard_json(summary: dict[str, Any], results: list[EvalResult], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "summary": summary,
        "chart": {
            "labels": [r.case_id for r in results],
            "total_score": [r.total_score for r in results],
            "content_score": [r.content_score for r in results],
            "latency_score": [r.latency_score for r in results],
            "latency_ms": [r.latency_ms for r in results],
        },
        "results": [asdict(r) for r in results],
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()
    dashboard_md_path = Path(args.dashboard_md).resolve()
    dashboard_json_path = Path(args.dashboard_json).resolve()

    cases = load_dataset(dataset_path)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("No evaluation cases found")

    mode = "live" if args.live else "validate_only"
    results: list[EvalResult] = []
    for case in cases:
        result = (
            run_case_live(case, timeout_seconds=args.timeout, pass_score_threshold=args.pass_score_threshold)
            if args.live
            else run_case_validate_only(case)
        )
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"[{status}] {case.case_id} score={result.total_score:.4f} latency={result.latency_ms:.2f}ms")
        if result.error:
            print(f"  error: {result.error}")

    summary = summarize(results)
    generated_at_utc = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_payload = {
        "generated_at_utc": generated_at_utc,
        "git_sha": os.getenv("GITHUB_SHA", ""),
        "mode": mode,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    dashboard_md = build_dashboard_markdown(summary, results, mode=mode)
    dashboard_md_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_md_path.write_text(dashboard_md, encoding="utf-8")

    dashboard_json = build_dashboard_json(summary, results, mode=mode)
    dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_json_path.write_text(json.dumps(dashboard_json, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Evaluation summary: "
        f"total={summary['total']} passed={summary['passed']} "
        f"failed={summary['failed']} avg_score={summary['avg_total_score']:.4f}"
    )
    print(f"Raw results: {output_path}")
    print(f"Dashboard markdown: {dashboard_md_path}")
    print(f"Dashboard json: {dashboard_json_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
