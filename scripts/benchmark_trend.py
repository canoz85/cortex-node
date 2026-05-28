import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark trend markdown from result JSON files")
    parser.add_argument("--results-dir", default="benchmarks/results", help="Directory with benchmark JSON files")
    parser.add_argument("--output", default="benchmarks/results/trend.md", help="Output markdown path")
    parser.add_argument("--limit", type=int, default=20, help="Max runs to include in trend table")
    return parser.parse_args()


def load_run(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None

    generated_at = str(payload.get("generated_at_utc") or "")
    mode = str(payload.get("mode") or "")
    pass_rate = float(summary.get("pass_rate") or 0.0)
    total = int(summary.get("total") or 0)
    passed = int(summary.get("passed") or 0)
    failed = int(summary.get("failed") or 0)
    avg_latency_ms = float(summary.get("avg_latency_ms") or 0.0)

    return {
        "file": path.name,
        "generated_at_utc": generated_at,
        "mode": mode,
        "pass_rate": pass_rate,
        "total": total,
        "passed": passed,
        "failed": failed,
        "avg_latency_ms": avg_latency_ms,
    }


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# Benchmark Trend",
        "",
        "| timestamp_utc | file | mode | pass_rate | passed/total | failed | avg_latency_ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ts = row["generated_at_utc"] or "(missing)"
        lines.append(
            f"| {ts} | {row['file']} | {row['mode']} | {row['pass_rate']:.2f}% | "
            f"{row['passed']}/{row['total']} | {row['failed']} | {row['avg_latency_ms']:.2f} |"
        )

    if len(rows) >= 2:
        latest = rows[0]
        prev = rows[1]
        delta = latest["pass_rate"] - prev["pass_rate"]
        lines.extend(
            [
                "",
                "## Delta vs previous run",
                "",
                f"- pass_rate_delta: {delta:+.2f}%",
                f"- avg_latency_delta_ms: {latest['avg_latency_ms'] - prev['avg_latency_ms']:+.2f}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_path = Path(args.output).resolve()

    json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for file_path in json_files:
        row = load_run(file_path)
        if row is not None:
            rows.append(row)

    if args.limit > 0:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(rows), encoding="utf-8")
    print(f"Trend report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
