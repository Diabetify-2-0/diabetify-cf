from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.collect_results import collect_results
from experiments.scripts.run_benchmark import DEFAULT_OUTPUT_ROOT
from experiments.scripts.run_comparison import engine_from_source_file

DEFAULT_REQUIRED_ENGINES = ["dice_plain", "dice_constrained_native", "nn_production"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _latest_comparison_root(output_root: Path) -> Path:
    pointer_path = output_root / "latest" / "comparison.txt"
    if not pointer_path.exists():
        raise FileNotFoundError(f"Latest comparison pointer not found: {pointer_path}")
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Latest comparison pointer is empty: {pointer_path}")
    return Path(raw)


def _engines_from_rows(rows: list[dict[str, str]]) -> set[str]:
    return {engine_from_source_file(row.get("source_file", "")) for row in rows}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_steps_by_engine(comparison_root: Path) -> dict[str, list[dict[str, Any]]]:
    steps_by_engine: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(comparison_root.rglob("baseline_manifest.json")):
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        engine = str(manifest.get("engine") or "")
        if engine:
            steps_by_engine[engine] = list(manifest.get("scenario_steps", []))
    return steps_by_engine


def _scenario_engine_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        engine = engine_from_source_file(row.get("source_file", ""))
        item = summary.setdefault(
            engine,
            {
                "scenario_count": 0,
                "completed_count": 0,
                "timeout_count": 0,
                "failed_count": 0,
                "mean_success_rate": 0.0,
            },
        )
        item["scenario_count"] += 1
        item["mean_success_rate"] += _as_float(row, "target_success_rate_all_candidates")

    for item in summary.values():
        count = max(int(item["scenario_count"]), 1)
        item["mean_success_rate"] = item["mean_success_rate"] / count
    return dict(sorted(summary.items()))


def _stability_engine_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        engine = engine_from_source_file(row.get("source_file", ""))
        summary[engine] = {
            "mean_feasible_only_jaccard_changed_features": _as_float(
                row,
                "mean_feasible_only_jaccard_changed_features",
            ),
            "mean_feasible_only_stability_std_norm": _as_float(
                row,
                "mean_feasible_only_stability_std_norm",
            ),
        }
    return dict(sorted(summary.items()))


def audit_comparison(
    comparison_root: Path,
    *,
    required_engines: list[str],
    fail_on_timeout: bool = False,
) -> dict[str, Any]:
    collect_results(comparison_root)
    scenario_path = comparison_root / "combined" / "scenario_summary.csv"
    stability_path = comparison_root / "combined" / "stability_summary.csv"
    candidate_path = comparison_root / "combined" / "candidates.csv"
    report_path = comparison_root / "comparison_report.md"
    manifest_path = comparison_root / "comparison_manifest.json"

    scenario_rows = _read_csv(scenario_path)
    stability_rows = _read_csv(stability_path)
    candidate_rows = _read_csv(candidate_path)
    scenario_steps = _scenario_steps_by_engine(comparison_root)
    engines = _engines_from_rows(scenario_rows) | set(scenario_steps)
    required = set(required_engines)

    errors: list[str] = []
    warnings: list[str] = []

    for path in [manifest_path, report_path, scenario_path, stability_path, candidate_path]:
        if not path.exists():
            errors.append(f"Missing required output file: {path}")

    if not scenario_rows:
        errors.append("No scenario summary rows found.")
    if not stability_rows:
        errors.append("No stability summary rows found.")

    missing_engines = sorted(required - engines)
    if missing_engines:
        errors.append(f"Missing required engine rows: {', '.join(missing_engines)}")

    scenario_summary = _scenario_engine_summary(scenario_rows)
    for engine, steps in scenario_steps.items():
        summary_item = scenario_summary.setdefault(
            engine,
            {
                "scenario_count": 0,
                "completed_count": 0,
                "timeout_count": 0,
                "failed_count": 0,
                "mean_success_rate": 0.0,
            },
        )
        summary_item["scenario_count"] = len(steps)
        for step in steps:
            status = str(step.get("status") or "completed")
            scenario = str(step.get("scenario") or "unknown")
            if status == "completed":
                summary_item["completed_count"] += 1
            elif status == "timeout":
                summary_item["timeout_count"] += 1
                message = f"{engine}/{scenario} timed out."
                if fail_on_timeout:
                    errors.append(message)
                else:
                    warnings.append(message)
            elif status == "failed":
                summary_item["failed_count"] += 1
                errors.append(f"{engine}/{scenario} failed.")

    stability_summary = _stability_engine_summary(stability_rows)
    payload = {
        "comparison_root": str(comparison_root),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "required_engines": required_engines,
        "fail_on_timeout": fail_on_timeout,
        "observed_engines": sorted(engines),
        "row_counts": {
            "scenario": len(scenario_rows),
            "stability": len(stability_rows),
            "candidates": len(candidate_rows),
        },
        "scenario_summary": scenario_summary,
        "stability_summary": stability_summary,
    }
    _write_json(comparison_root / "audit_report.json", payload)
    return payload


def print_audit(payload: dict[str, Any]) -> None:
    status = "PASS" if payload["ok"] else "FAIL"
    print(f"Comparison audit: {status}")
    print(f"Root: {payload['comparison_root']}")
    print(f"Observed engines: {', '.join(payload['observed_engines'])}")
    print(
        "Rows: "
        f"scenario={payload['row_counts']['scenario']} "
        f"stability={payload['row_counts']['stability']} "
        f"candidates={payload['row_counts']['candidates']}"
    )

    if payload["errors"]:
        print("\nErrors")
        for item in payload["errors"]:
            print(f"- {item}")

    if payload["warnings"]:
        print("\nWarnings")
        for item in payload["warnings"]:
            print(f"- {item}")

    print("\nScenario Summary")
    for engine, item in payload["scenario_summary"].items():
        print(
            f"- {engine}: scenarios={item['scenario_count']} "
            f"completed={item['completed_count']} "
            f"timeout={item['timeout_count']} "
            f"failed={item['failed_count']} "
            f"mean_success={item['mean_success_rate']:.3f}"
        )

    print("\nStability Summary")
    for engine, item in payload["stability_summary"].items():
        print(
            f"- {engine}: "
            f"feasible_only_jaccard="
            f"{item['mean_feasible_only_jaccard_changed_features']:.3f} "
            f"feasible_only_std_norm="
            f"{item['mean_feasible_only_stability_std_norm']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit comparison experiment outputs.")
    parser.add_argument(
        "comparison_root",
        nargs="?",
        type=Path,
        help="Comparison root. Defaults to experiments/results/latest/comparison.txt.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--required-engines",
        nargs="*",
        default=DEFAULT_REQUIRED_ENGINES,
        help="Engines that must be present in scenario summary rows.",
    )
    parser.add_argument(
        "--fail-on-timeout",
        action="store_true",
        help="Treat scenario timeouts as audit failures.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    comparison_root = args.comparison_root or _latest_comparison_root(args.output_root)
    payload = audit_comparison(
        comparison_root,
        required_engines=list(args.required_engines),
        fail_on_timeout=args.fail_on_timeout,
    )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print_audit(payload)

    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
