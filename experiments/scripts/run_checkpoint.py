from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:  
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.audit_comparison import ( 
    DEFAULT_REQUIRED_ENGINES,
    audit_comparison,
)
from experiments.scripts.run_baseline import (  
    DEFAULT_SCENARIO_TIMEOUT_SECONDS,
    DEFAULT_STABILITY_CONFIG,
    DEFAULT_STABILITY_TIMEOUT_SECONDS,
)
from experiments.scripts.run_benchmark import DEFAULT_OUTPUT_ROOT  
from experiments.scripts.run_comparison import (  
    DEFAULT_ENGINE_CONFIGS,
    run_comparison,
)
from experiments.scripts.run_scenarios import DEFAULT_SCENARIO_CONFIGS  


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def build_checkpoint_report(audit_payload: dict[str, Any]) -> str:
    status = "PASS" if audit_payload["ok"] else "FAIL"
    lines = [
        "# Experiment Checkpoint",
        "",
        f"- Status: `{status}`",
        f"- Comparison root: `{audit_payload['comparison_root']}`",
        f"- Observed engines: {', '.join(audit_payload['observed_engines'])}",
        "- Rows: "
        f"scenario={audit_payload['row_counts']['scenario']}, "
        f"stability={audit_payload['row_counts']['stability']}, "
        f"candidates={audit_payload['row_counts']['candidates']}",
        "",
        "## Scenario Summary",
        "",
        "| Engine | Scenarios | Completed | Timeout | Failed | Mean feasible | Max violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for engine, item in audit_payload["scenario_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    engine,
                    str(item["scenario_count"]),
                    str(item["completed_count"]),
                    str(item["timeout_count"]),
                    str(item["failed_count"]),
                    _percent(float(item["mean_feasible_rate"])),
                    _number(float(item["max_violation_rate"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Stability Summary",
            "",
            "| Engine | Cases | Mean feasible | Evaluable | Feasible-only Jaccard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for engine, item in audit_payload["stability_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    engine,
                    str(item["case_count"]),
                    _percent(float(item["mean_feasible_rate"])),
                    _percent(float(item["stability_evaluable_case_rate"])),
                    _number(float(item["mean_feasible_only_jaccard_changed_features"])),
                ]
            )
            + " |"
        )

    if audit_payload["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in audit_payload["errors"])

    if audit_payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in audit_payload["warnings"])

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison_report.md`",
            "- `comparison_manifest.json`",
            "- `audit_report.json`",
            "- `checkpoint_report.md`",
            "- `combined/scenario_summary.csv`",
            "- `combined/stability_summary.csv`",
            "- `combined/inputs.csv`",
            "- `combined/candidates.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def write_checkpoint_report(comparison_root: Path, audit_payload: dict[str, Any]) -> Path:
    report_path = comparison_root / "checkpoint_report.md"
    report_path.write_text(build_checkpoint_report(audit_payload), encoding="utf-8")
    return report_path


def write_checkpoint_manifest(
    comparison_root: Path,
    *,
    audit_payload: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> Path:
    manifest_path = comparison_root / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "comparison_root": str(comparison_root),
                "ok": bool(audit_payload["ok"]),
                "checkpoint_config": checkpoint_config,
                "audit_report": str(comparison_root / "audit_report.json"),
                "checkpoint_report": str(comparison_root / "checkpoint_report.md"),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def run_checkpoint(
    *,
    output_root: Path,
    engine_config_paths: list[Path],
    scenario_config_paths: list[Path],
    stability_config_path: Path,
    scenario_limit: int | None,
    stability_limit: int | None,
    repeat_count: int,
    skip_stability: bool,
    scenario_timeout_seconds: int | None,
    stability_timeout_seconds: int | None,
    required_engines: list[str],
    max_allowed_violation_rate: float,
    fail_on_timeout: bool,
) -> dict[str, Any]:
    comparison_root = run_comparison(
        output_root=output_root,
        engine_config_paths=engine_config_paths,
        scenario_config_paths=scenario_config_paths,
        stability_config_path=stability_config_path,
        scenario_limit=scenario_limit,
        stability_limit=stability_limit,
        repeat_count=repeat_count,
        skip_stability=skip_stability,
        scenario_timeout_seconds=scenario_timeout_seconds,
        stability_timeout_seconds=stability_timeout_seconds,
    )
    audit_payload = audit_comparison(
        comparison_root,
        required_engines=required_engines,
        max_allowed_violation_rate=max_allowed_violation_rate,
        fail_on_timeout=fail_on_timeout,
    )
    report_path = write_checkpoint_report(comparison_root, audit_payload)
    manifest_path = write_checkpoint_manifest(
        comparison_root,
        audit_payload=audit_payload,
        checkpoint_config={
            "engine_configs": [str(path) for path in engine_config_paths],
            "scenario_configs": [str(path) for path in scenario_config_paths],
            "stability_config": str(stability_config_path),
            "scenario_limit": scenario_limit,
            "stability_limit": stability_limit,
            "repeat_count": repeat_count,
            "skip_stability": skip_stability,
            "scenario_timeout_seconds": scenario_timeout_seconds,
            "stability_timeout_seconds": stability_timeout_seconds,
            "required_engines": required_engines,
            "max_allowed_violation_rate": max_allowed_violation_rate,
            "fail_on_timeout": fail_on_timeout,
        },
    )
    return {
        "comparison_root": comparison_root,
        "audit": audit_payload,
        "checkpoint_report": report_path,
        "checkpoint_manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comparison, audit it, and write a checkpoint report."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--engine-configs",
        nargs="*",
        type=Path,
        default=DEFAULT_ENGINE_CONFIGS,
    )
    parser.add_argument(
        "--scenario-configs",
        nargs="*",
        type=Path,
        default=DEFAULT_SCENARIO_CONFIGS,
    )
    parser.add_argument("--stability-config", type=Path, default=DEFAULT_STABILITY_CONFIG)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--stability-limit", type=int, default=None)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument(
        "--scenario-timeout-seconds",
        type=int,
        default=DEFAULT_SCENARIO_TIMEOUT_SECONDS,
        help="Timeout per engine/scenario subprocess. Use 0 to disable.",
    )
    parser.add_argument(
        "--stability-timeout-seconds",
        type=int,
        default=DEFAULT_STABILITY_TIMEOUT_SECONDS,
        help="Timeout per engine stability subprocess. Use 0 to disable.",
    )
    parser.add_argument(
        "--required-engines",
        nargs="*",
        default=DEFAULT_REQUIRED_ENGINES,
        help="Engines that must be present in scenario summary rows.",
    )
    parser.add_argument(
        "--max-allowed-violation-rate",
        type=float,
        default=0.0,
        help="Maximum accepted constraint violation rate before failing the checkpoint.",
    )
    parser.add_argument(
        "--fail-on-timeout",
        action="store_true",
        help="Treat scenario timeouts as checkpoint failures.",
    )
    args = parser.parse_args()

    result = run_checkpoint(
        output_root=args.output_root,
        engine_config_paths=args.engine_configs,
        scenario_config_paths=args.scenario_configs,
        stability_config_path=args.stability_config,
        scenario_limit=args.scenario_limit,
        stability_limit=args.stability_limit,
        repeat_count=args.repeat_count,
        skip_stability=args.skip_stability,
        scenario_timeout_seconds=args.scenario_timeout_seconds,
        stability_timeout_seconds=args.stability_timeout_seconds,
        required_engines=list(args.required_engines),
        max_allowed_violation_rate=args.max_allowed_violation_rate,
        fail_on_timeout=args.fail_on_timeout,
    )

    audit_payload = result["audit"]
    status = "PASS" if audit_payload["ok"] else "FAIL"
    print(f"Checkpoint: {status}")
    print(f"Comparison root: {result['comparison_root']}")
    print(f"Checkpoint report: {result['checkpoint_report']}")
    print(f"Checkpoint manifest: {result['checkpoint_manifest']}")
    if not audit_payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
