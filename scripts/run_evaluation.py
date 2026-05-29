import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings


@dataclass
class EvalCase:
    case_id: str
    prompt: str
    route: str
    semantic_reference: str
    checks_all: list[str]
    checks_any: list[str]
    regex_all: list[str]
    regex_any: list[str]
    forbidden_substrings: list[str]
    min_output_chars: int
    expected_substrings: list[str]
    max_latency_ms: float
    weight_content: float
    weight_latency: float
    dimension_weights: dict[str, float]
    tags: list[str]


@dataclass
class EvalResult:
    case_id: str
    route: str
    success: bool
    return_code: int
    latency_ms: float
    matched_substrings: list[str]
    missing_substrings: list[str]
    matched_all_checks: list[str]
    missing_all_checks: list[str]
    matched_any_checks: list[str]
    missing_any_checks: list[str]
    matched_regex_all: list[str]
    missing_regex_all: list[str]
    matched_regex_any: list[str]
    missing_regex_any: list[str]
    matched_forbidden_substrings: list[str]
    output_char_count: int
    dimension_scores: dict[str, float]
    semantic_score: float
    semantic_method: str
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
    parser.add_argument(
        "--semantic-scoring",
        action="store_true",
        help="Enable semantic similarity scoring for cases that define semantic_reference.",
    )
    parser.add_argument(
        "--semantic-model",
        default="nomic-embed-text",
        help="Embedding model name for semantic scoring.",
    )
    parser.add_argument(
        "--policy",
        default="",
        help="Optional policy JSON file to enforce global/route score thresholds.",
    )
    parser.add_argument(
        "--enforce-policy",
        action="store_true",
        help="Fail command when policy thresholds are not met.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_weight_map(raw: Any, default: dict[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        raw = {}

    normalized: dict[str, float] = {}
    for key, fallback in default.items():
        candidate = _safe_float(raw.get(key, fallback), fallback)
        normalized[key] = max(0.0, candidate)

    total = sum(normalized.values())
    if total <= 0:
        return default

    return {key: value / total for key, value in normalized.items()}


def _as_clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                cleaned.append(text)
    return cleaned


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _token_set_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9_]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9_]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union == 0:
        return 0.0
    return intersection / union


def semantic_similarity_score(
    output: str,
    reference: str,
    semantic_scoring: bool,
    semantic_model: str,
    embedding_cache: dict[str, list[float]],
    embeddings_client: OllamaEmbeddings | None,
) -> tuple[float, str, OllamaEmbeddings | None]:
    output_text = (output or "").strip()
    reference_text = (reference or "").strip()
    if not reference_text:
        return 1.0, "not_configured", embeddings_client
    if not output_text:
        return 0.0, "empty_output", embeddings_client

    if semantic_scoring:
        try:
            client = embeddings_client
            if client is None:
                client = OllamaEmbeddings(model=semantic_model, validate_model_on_init=False)

            if reference_text not in embedding_cache:
                embedding_cache[reference_text] = list(client.embed_query(reference_text))
            if output_text not in embedding_cache:
                embedding_cache[output_text] = list(client.embed_query(output_text))

            score = _cosine_similarity(embedding_cache[output_text], embedding_cache[reference_text])
            return round(score, 4), "embedding", client
        except Exception:
            pass

    lexical_score = _token_set_similarity(output_text, reference_text)
    return round(lexical_score, 4), "lexical_fallback", embeddings_client


def load_policy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Evaluation policy must be a JSON object")
    return raw


def evaluate_policy(summary: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    global_policy = policy.get("global", {}) if isinstance(policy.get("global"), dict) else {}
    route_policy = policy.get("routes", {}) if isinstance(policy.get("routes"), dict) else {}

    min_pass_rate = _safe_float(global_policy.get("min_pass_rate", 0.0), 0.0)
    min_avg_total = _safe_float(global_policy.get("min_avg_total_score", 0.0), 0.0)
    min_cases = int(_safe_float(global_policy.get("min_cases", 0), 0))

    if float(summary.get("pass_rate", 0.0)) < min_pass_rate:
        errors.append(
            f"Global pass_rate {float(summary.get('pass_rate', 0.0)):.2f}% is below policy minimum {min_pass_rate:.2f}%"
        )
    if float(summary.get("avg_total_score", 0.0)) < min_avg_total:
        errors.append(
            f"Global avg_total_score {float(summary.get('avg_total_score', 0.0)):.4f} is below policy minimum {min_avg_total:.4f}"
        )
    if int(summary.get("total", 0)) < min_cases:
        errors.append(f"Total cases {int(summary.get('total', 0))} is below policy minimum {min_cases}")

    route_summary = summary.get("routes", {}) if isinstance(summary.get("routes"), dict) else {}
    for route_name, route_cfg in route_policy.items():
        if not isinstance(route_cfg, dict):
            continue
        if route_name not in route_summary:
            errors.append(f"Route '{route_name}' missing from evaluation summary")
            continue
        route_row = route_summary.get(route_name, {})
        route_min_pass = _safe_float(route_cfg.get("min_pass_rate", 0.0), 0.0)
        route_min_avg = _safe_float(route_cfg.get("min_avg_total_score", 0.0), 0.0)
        route_min_cases = int(_safe_float(route_cfg.get("min_cases", 0), 0))

        if float(route_row.get("pass_rate", 0.0)) < route_min_pass:
            errors.append(
                f"Route '{route_name}' pass_rate {float(route_row.get('pass_rate', 0.0)):.2f}% is below {route_min_pass:.2f}%"
            )
        if float(route_row.get("avg_total_score", 0.0)) < route_min_avg:
            errors.append(
                f"Route '{route_name}' avg_total_score {float(route_row.get('avg_total_score', 0.0)):.4f} is below {route_min_avg:.4f}"
            )
        if int(route_row.get("total", 0)) < route_min_cases:
            errors.append(
                f"Route '{route_name}' total cases {int(route_row.get('total', 0))} is below minimum {route_min_cases}"
            )

    return len(errors) == 0, errors


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
        route = str(item.get("route", "unclassified")).strip().lower() or "unclassified"
        semantic_reference = str(item.get("semantic_reference", "")).strip()
        checks = item.get("checks", {})
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
        if checks and not isinstance(checks, dict):
            raise ValueError(f"Case '{case_id}' has invalid checks")

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

        checks_all = _as_clean_string_list(checks.get("all", [])) if isinstance(checks, dict) else []
        checks_any = _as_clean_string_list(checks.get("any", [])) if isinstance(checks, dict) else []
        regex_all = _as_clean_string_list(checks.get("regex_all", [])) if isinstance(checks, dict) else []
        regex_any = _as_clean_string_list(checks.get("regex_any", [])) if isinstance(checks, dict) else []
        forbidden_substrings = _as_clean_string_list(checks.get("forbidden_substrings", [])) if isinstance(checks, dict) else []
        min_output_chars = int(_safe_float(checks.get("min_output_chars", 0), 0)) if isinstance(checks, dict) else 0
        if min_output_chars < 0:
            min_output_chars = 0

        dimension_weights = _normalize_weight_map(
            item.get("dimension_weights", {}),
            {
                "substring": 0.45,
                "checks": 0.35,
                "semantic": 0.10,
                "safety": 0.10,
                "format": 0.00,
            },
        )

        cases.append(
            EvalCase(
                case_id=case_id,
                prompt=prompt,
                route=route,
                semantic_reference=semantic_reference,
                checks_all=checks_all,
                checks_any=checks_any,
                regex_all=regex_all,
                regex_any=regex_any,
                forbidden_substrings=forbidden_substrings,
                min_output_chars=min_output_chars,
                expected_substrings=[x.strip() for x in expected if x.strip()],
                max_latency_ms=max_latency_ms,
                weight_content=weight_content,
                weight_latency=weight_latency,
                dimension_weights=dimension_weights,
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


def _evaluate_extended_checks(case: EvalCase, output: str) -> dict[str, Any]:
    normalized_output = (output or "").lower()
    raw_output = output or ""

    matched_all_checks = [token for token in case.checks_all if token.lower() in normalized_output]
    missing_all_checks = [token for token in case.checks_all if token.lower() not in normalized_output]

    matched_any_checks = [token for token in case.checks_any if token.lower() in normalized_output]
    missing_any_checks = [token for token in case.checks_any if token.lower() not in normalized_output]

    matched_regex_all = [pattern for pattern in case.regex_all if re.search(pattern, raw_output, re.IGNORECASE)]
    missing_regex_all = [pattern for pattern in case.regex_all if pattern not in matched_regex_all]

    matched_regex_any = [pattern for pattern in case.regex_any if re.search(pattern, raw_output, re.IGNORECASE)]
    missing_regex_any = [pattern for pattern in case.regex_any if pattern not in matched_regex_any]

    matched_forbidden_substrings = [token for token in case.forbidden_substrings if token.lower() in normalized_output]
    output_char_count = len(raw_output.strip())

    checks_components: list[float] = []
    if case.checks_all:
        checks_components.append(len(matched_all_checks) / len(case.checks_all))
    if case.checks_any:
        checks_components.append(1.0 if matched_any_checks else 0.0)
    if case.regex_all:
        checks_components.append(len(matched_regex_all) / len(case.regex_all))
    if case.regex_any:
        checks_components.append(1.0 if matched_regex_any else 0.0)
    checks_score = sum(checks_components) / len(checks_components) if checks_components else 1.0

    safety_score = 1.0 if not matched_forbidden_substrings else 0.0
    format_score = 1.0 if output_char_count >= case.min_output_chars else 0.0

    return {
        "matched_all_checks": matched_all_checks,
        "missing_all_checks": missing_all_checks,
        "matched_any_checks": matched_any_checks,
        "missing_any_checks": missing_any_checks,
        "matched_regex_all": matched_regex_all,
        "missing_regex_all": missing_regex_all,
        "matched_regex_any": matched_regex_any,
        "missing_regex_any": missing_regex_any,
        "matched_forbidden_substrings": matched_forbidden_substrings,
        "output_char_count": output_char_count,
        "checks_score": round(checks_score, 4),
        "safety_score": round(safety_score, 4),
        "format_score": round(format_score, 4),
    }


def _content_score_with_dimensions(
    case: EvalCase,
    substring_score: float,
    checks_score: float,
    semantic_score: float,
    safety_score: float,
    format_score: float,
) -> tuple[float, dict[str, float]]:
    dimension_scores = {
        "substring": round(substring_score, 4),
        "checks": round(checks_score, 4),
        "semantic": round(semantic_score, 4),
        "safety": round(safety_score, 4),
        "format": round(format_score, 4),
    }

    weighted = sum(dimension_scores[key] * case.dimension_weights.get(key, 0.0) for key in dimension_scores)
    return round(weighted, 4), dimension_scores


def latency_score(latency_ms: float, max_latency_ms: float) -> float:
    if latency_ms <= max_latency_ms:
        return 1.0
    return round(max(0.0, max_latency_ms / latency_ms), 4)


def run_case_live(
    case: EvalCase,
    timeout_seconds: int,
    pass_score_threshold: float,
    semantic_scoring: bool,
    semantic_model: str,
    embedding_cache: dict[str, list[float]],
    embeddings_client: OllamaEmbeddings | None,
) -> tuple[EvalResult, OllamaEmbeddings | None]:
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
        matched, missing, substring_score = evaluate_output(case, output)
        extended = _evaluate_extended_checks(case, output)
        semantic_score, semantic_method, embeddings_client = semantic_similarity_score(
            output=output,
            reference=case.semantic_reference,
            semantic_scoring=semantic_scoring,
            semantic_model=semantic_model,
            embedding_cache=embedding_cache,
            embeddings_client=embeddings_client,
        )
        content, dimension_scores = _content_score_with_dimensions(
            case,
            substring_score=substring_score,
            checks_score=float(extended["checks_score"]),
            semantic_score=semantic_score,
            safety_score=float(extended["safety_score"]),
            format_score=float(extended["format_score"]),
        )
        lat_score = latency_score(elapsed_ms, case.max_latency_ms)
        total = round((content * case.weight_content) + (lat_score * case.weight_latency), 4)

        success = completed.returncode == 0 and total >= pass_score_threshold
        error = "" if completed.returncode == 0 else f"Non-zero exit code: {completed.returncode}"

        return EvalResult(
            case_id=case.case_id,
            route=case.route,
            success=success,
            return_code=completed.returncode,
            latency_ms=elapsed_ms,
            matched_substrings=matched,
            missing_substrings=missing,
            matched_all_checks=list(extended["matched_all_checks"]),
            missing_all_checks=list(extended["missing_all_checks"]),
            matched_any_checks=list(extended["matched_any_checks"]),
            missing_any_checks=list(extended["missing_any_checks"]),
            matched_regex_all=list(extended["matched_regex_all"]),
            missing_regex_all=list(extended["missing_regex_all"]),
            matched_regex_any=list(extended["matched_regex_any"]),
            missing_regex_any=list(extended["missing_regex_any"]),
            matched_forbidden_substrings=list(extended["matched_forbidden_substrings"]),
            output_char_count=int(extended["output_char_count"]),
            dimension_scores=dimension_scores,
            semantic_score=semantic_score,
            semantic_method=semantic_method,
            content_score=content,
            latency_score=lat_score,
            total_score=total,
            max_latency_ms=case.max_latency_ms,
            output_excerpt=output[-1200:],
            error=error,
            tags=case.tags,
        ), embeddings_client
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return EvalResult(
            case_id=case.case_id,
            route=case.route,
            success=False,
            return_code=124,
            latency_ms=elapsed_ms,
            matched_substrings=[],
            missing_substrings=case.expected_substrings,
            matched_all_checks=[],
            missing_all_checks=case.checks_all,
            matched_any_checks=[],
            missing_any_checks=case.checks_any,
            matched_regex_all=[],
            missing_regex_all=case.regex_all,
            matched_regex_any=[],
            missing_regex_any=case.regex_any,
            matched_forbidden_substrings=[],
            output_char_count=0,
            dimension_scores={
                "substring": 0.0,
                "checks": 0.0,
                "semantic": 0.0,
                "safety": 1.0,
                "format": 0.0,
            },
            semantic_score=0.0,
            semantic_method="timeout",
            content_score=0.0,
            latency_score=0.0,
            total_score=0.0,
            max_latency_ms=case.max_latency_ms,
            output_excerpt="",
            error=f"Timed out after {timeout_seconds} seconds",
            tags=case.tags,
        ), embeddings_client


def run_case_validate_only(case: EvalCase) -> EvalResult:
    return EvalResult(
        case_id=case.case_id,
        route=case.route,
        success=True,
        return_code=0,
        latency_ms=0.0,
        matched_substrings=[],
        missing_substrings=[],
        matched_all_checks=[],
        missing_all_checks=[],
        matched_any_checks=[],
        missing_any_checks=[],
        matched_regex_all=[],
        missing_regex_all=[],
        matched_regex_any=[],
        missing_regex_any=[],
        matched_forbidden_substrings=[],
        output_char_count=0,
        dimension_scores={
            "substring": 1.0,
            "checks": 1.0,
            "semantic": 1.0,
            "safety": 1.0,
            "format": 1.0,
        },
        semantic_score=1.0,
        semantic_method="validate_only",
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
    by_route: dict[str, dict[str, Any]] = {}
    for item in results:
        route = item.route or "unclassified"
        current = by_route.setdefault(
            route,
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "avg_total_score": 0.0,
                "avg_latency_ms": 0.0,
            },
        )
        current["total"] += 1
        if item.success:
            current["passed"] += 1
        else:
            current["failed"] += 1
        current["avg_total_score"] += item.total_score
        current["avg_latency_ms"] += item.latency_ms

    for route_name, current in by_route.items():
        route_total = current["total"] or 1
        current["avg_total_score"] = round(current["avg_total_score"] / route_total, 4)
        current["avg_latency_ms"] = round(current["avg_latency_ms"] / route_total, 2)
        current["pass_rate"] = round((current["passed"] / route_total) * 100.0, 2)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100.0, 2) if total else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "avg_total_score": avg_score,
        "routes": by_route,
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

    route_summary = summary.get("routes", {}) if isinstance(summary.get("routes", {}), dict) else {}
    if route_summary:
        lines.extend(
            [
                "",
                "## Route Breakdown",
                "",
                "| route | total | passed | failed | pass_rate | avg_total_score | avg_latency_ms |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for route_name in sorted(route_summary.keys()):
            row = route_summary.get(route_name, {})
            lines.append(
                f"| {route_name} | {int(row.get('total', 0))} | {int(row.get('passed', 0))} | {int(row.get('failed', 0))} | "
                f"{float(row.get('pass_rate', 0.0)):.2f}% | {float(row.get('avg_total_score', 0.0)):.4f} | {float(row.get('avg_latency_ms', 0.0)):.2f} |"
            )

    return "\n".join(lines) + "\n"


def build_dashboard_json(summary: dict[str, Any], results: list[EvalResult], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "summary": summary,
        "route_summary": summary.get("routes", {}),
        "chart": {
            "labels": [r.case_id for r in results],
            "routes": [r.route for r in results],
            "total_score": [r.total_score for r in results],
            "content_score": [r.content_score for r in results],
            "latency_score": [r.latency_score for r in results],
            "latency_ms": [r.latency_ms for r in results],
            "substring_score": [r.dimension_scores.get("substring", 0.0) for r in results],
            "checks_score": [r.dimension_scores.get("checks", 0.0) for r in results],
            "semantic_score": [r.dimension_scores.get("semantic", 0.0) for r in results],
            "safety_score": [r.dimension_scores.get("safety", 0.0) for r in results],
            "format_score": [r.dimension_scores.get("format", 0.0) for r in results],
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
    embedding_cache: dict[str, list[float]] = {}
    embeddings_client: OllamaEmbeddings | None = None
    for case in cases:
        if args.live:
            result, embeddings_client = run_case_live(
                case,
                timeout_seconds=args.timeout,
                pass_score_threshold=args.pass_score_threshold,
                semantic_scoring=bool(args.semantic_scoring),
                semantic_model=str(args.semantic_model),
                embedding_cache=embedding_cache,
                embeddings_client=embeddings_client,
            )
        else:
            result = run_case_validate_only(case)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(
            f"[{status}] {case.case_id} score={result.total_score:.4f} latency={result.latency_ms:.2f}ms "
            f"semantic={result.semantic_score:.4f} ({result.semantic_method})"
        )
        if result.error:
            print(f"  error: {result.error}")

    summary = summarize(results)
    generated_at_utc = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_payload = {
        "generated_at_utc": generated_at_utc,
        "git_sha": os.getenv("GITHUB_SHA", ""),
        "mode": mode,
        "semantic_scoring": bool(args.semantic_scoring),
        "semantic_model": str(args.semantic_model),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    policy_status = {
        "enabled": False,
        "enforced": bool(args.enforce_policy),
        "passed": True,
        "errors": [],
    }
    if args.policy:
        policy_path = Path(args.policy).resolve()
        policy = load_policy(policy_path)
        policy_ok, policy_errors = evaluate_policy(summary, policy)
        policy_status = {
            "enabled": True,
            "policy_path": str(policy_path),
            "enforced": bool(args.enforce_policy),
            "passed": policy_ok,
            "errors": policy_errors,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_payload["policy"] = policy_status
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
    if policy_status.get("enabled"):
        if policy_status.get("passed"):
            print("Policy check: PASS")
        else:
            print("Policy check: FAIL")
            for error in policy_status.get("errors", []):
                print(f"  - {error}")
    print(f"Raw results: {output_path}")
    print(f"Dashboard markdown: {dashboard_md_path}")
    print(f"Dashboard json: {dashboard_json_path}")

    failed_eval = summary["failed"] != 0
    failed_policy = bool(policy_status.get("enabled") and policy_status.get("enforced") and not policy_status.get("passed"))
    return 0 if not failed_eval and not failed_policy else 1


if __name__ == "__main__":
    raise SystemExit(main())
