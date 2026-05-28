import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage ratchet policy checker")
    parser.add_argument("--coverage-json", default="coverage.json", help="Path to pytest-cov JSON report")
    parser.add_argument("--policy", default=".github/coverage-policy.json", help="Path to coverage policy JSON")
    parser.add_argument(
        "--summary-out",
        default="benchmarks/results/coverage-ratchet.md",
        help="Markdown summary output path",
    )
    parser.add_argument(
        "--enforce-ready",
        action="store_true",
        help="Fail if coverage is high enough to ratchet but current gate has not been raised",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_summary(
    out_path: Path,
    measured: float,
    current_gate: int,
    step: int,
    target_gate: int,
    next_gate: int,
    ready_to_ratchet: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = "ready-to-ratchet" if ready_to_ratchet else "stable"
    content = "\n".join(
        [
            "# Coverage Ratchet Summary",
            "",
            f"- measured_coverage: {measured:.2f}%",
            f"- current_gate: {current_gate}%",
            f"- policy_step: +{step}%",
            f"- target_gate: {target_gate}%",
            f"- next_recommended_gate: {next_gate}%",
            f"- status: {status}",
            "",
            "## Policy",
            "",
            "Raise `--cov-fail-under` by one step whenever measured coverage reaches or exceeds the next step threshold.",
        ]
    )
    out_path.write_text(content + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    coverage_json = Path(args.coverage_json).resolve()
    policy_json = Path(args.policy).resolve()
    summary_out = Path(args.summary_out).resolve()

    coverage_data = load_json(coverage_json)
    policy_data = load_json(policy_json)

    measured = float(coverage_data["totals"]["percent_covered"])
    current_gate = int(policy_data["current_gate"])
    step = int(policy_data["step"])
    target_gate = int(policy_data["target_gate"])

    if step <= 0:
        raise ValueError("Policy 'step' must be positive")
    if target_gate < current_gate:
        raise ValueError("Policy 'target_gate' must be >= 'current_gate'")

    next_gate = min(current_gate + step, target_gate)
    ready_to_ratchet = current_gate < target_gate and measured >= next_gate

    write_summary(
        summary_out,
        measured=measured,
        current_gate=current_gate,
        step=step,
        target_gate=target_gate,
        next_gate=next_gate,
        ready_to_ratchet=ready_to_ratchet,
    )

    print(f"Measured coverage: {measured:.2f}%")
    print(f"Current gate: {current_gate}% | Next recommended gate: {next_gate}%")
    print(f"Summary written to: {summary_out}")

    if measured < current_gate:
        print("Coverage is below current gate.")
        return 1

    if ready_to_ratchet and args.enforce_ready:
        print("Coverage reached ratchet threshold; increase --cov-fail-under and policy current_gate.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
