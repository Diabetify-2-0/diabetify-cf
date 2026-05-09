from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from experiments.scripts._bootstrap import bootstrap_path
except ModuleNotFoundError:
    from _bootstrap import bootstrap_path


bootstrap_path(__file__)

from experiments.scripts.collect_results import collect_results
from experiments.scripts.run_baseline import (
    DEFAULT_SCENARIO_TIMEOUT_SECONDS,
    DEFAULT_STABILITY_CONFIG,
    DEFAULT_STABILITY_TIMEOUT_SECONDS,
    run_engine_baseline,
)
from experiments.scripts.run_benchmark import (
    DEFAULT_OUTPUT_ROOT,
    engine_output_label,
    load_config,
)
from experiments.scripts.run_scenarios import DEFAULT_SCENARIO_CONFIGS

DEFAULT_ENGINE_CONFIGS = [
    Path("experiments/configs/engines/dice.json"),
    Path("experiments/configs/engines/ocean.json"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _engine_name_from_config(engine_config_path: Path) -> str:
    config = load_config(engine_config_path)
    return engine_output_label(config) or "unknown"


def _make_comparison_root(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_root = output_root / "comparisons" / timestamp
    comparison_root.mkdir(parents=True, exist_ok=False)
    return comparison_root


def _write_latest_pointer(output_root: Path, comparison_root: Path) -> Path:
    pointer_path = output_root / "latest" / "comparison.txt"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(comparison_root), encoding="utf-8")
    return pointer_path


def engine_from_source_file(source_file: str) -> str:
    parts = Path(source_file).parts
    if "baselines" in parts:
        index = parts.index("baselines")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _status_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.get("step_status") or "completed"] += 1
    return counts


def _candidate_counts_by_engine(candidate_rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in candidate_rows:
        engine = row.get("engine_name") or engine_from_source_file(row.get("source_file", ""))
        counts[engine] += 1
    return counts


def _top_changed_features_by_engine(
    candidate_rows: list[dict[str, str]], top_n: int
) -> dict[str, list[tuple[str, int]]]:
    counts_by_engine: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidate_rows:
        engine = row.get("engine_name") or engine_from_source_file(row.get("source_file", ""))
        try:
            delta = json.loads(row.get("delta", "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(delta, dict):
            continue
        for feature_name in delta:
            counts_by_engine[engine][str(feature_name)] += 1

    return {
        engine: counts.most_common(top_n) for engine, counts in sorted(counts_by_engine.items())
    }


def _scenario_rows_by_engine(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[engine_from_source_file(row.get("source_file", ""))].append(row)
    return dict(sorted(grouped.items()))


def _stability_rows_by_engine(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[engine_from_source_file(row.get("source_file", ""))].append(row)
    return dict(sorted(grouped.items()))


def build_comparison_report(comparison_root: Path, top_n: int = 10) -> str:
    collect_results(comparison_root)
    scenario_rows = _read_csv(comparison_root / "combined" / "scenario_summary.csv")
    stability_rows = _read_csv(comparison_root / "combined" / "stability_summary.csv")
    candidate_rows = _read_csv(comparison_root / "combined" / "candidates.csv")
    scenarios_by_engine = _scenario_rows_by_engine(scenario_rows)
    stability_by_engine = _stability_rows_by_engine(stability_rows)
    candidate_counts = _candidate_counts_by_engine(candidate_rows)

    lines = [
        "# Comparison Report",
        "",
        f"- Comparison root: `{comparison_root}`",
        f"- Scenario rows: {len(scenario_rows)}",
        f"- Stability rows: {len(stability_rows)}",
        f"- Candidate rows: {len(candidate_rows)}",
        "",
        "## Engine Summary",
        "",
    ]

    if scenarios_by_engine:
        lines.append(
            "| Engine | Scenarios | Completed | Timeout | Failed | Mean feasible | "
            "Candidate rows |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for engine, rows in scenarios_by_engine.items():
            statuses = _status_counts(rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        engine,
                        str(len(rows)),
                        str(statuses.get("completed", 0)),
                        str(statuses.get("timeout", 0)),
                        str(statuses.get("failed", 0)),
                        _percent(_mean([_as_float(row, "feasible_rate") for row in rows])),
                        str(candidate_counts.get(engine, 0)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No scenario summary rows found.")

    lines.extend(["", "## Scenario Matrix", ""])
    if scenario_rows:
        lines.append(
            "| Engine | Scenario | Status | Feasible | Target | Runtime ms | Step runtime s | "
            "Reason counts |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in scenario_rows:
            engine = engine_from_source_file(row.get("source_file", ""))
            lines.append(
                "| "
                + " | ".join(
                    [
                        engine,
                        row.get("scenario", "-"),
                        row.get("step_status") or "completed",
                        _percent(_as_float(row, "feasible_rate")),
                        _percent(_as_float(row, "target_success_rate")),
                        _number(_as_float(row, "mean_runtime_ms")),
                        _number(_as_float(row, "step_runtime_seconds")),
                        row.get("reason_counts", "{}").replace("|", "\\|"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No scenario matrix available.")

    lines.extend(["", "## Stability", ""])
    if stability_by_engine:
        lines.append(
            "| Engine | Case count | Mean feasible | Fully feasible cases | "
            "Stability evaluable cases | Feasible-only Jaccard | "
            "Feasible-only std norm | All-repeat Jaccard |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for engine, rows in stability_by_engine.items():
            case_count = sum(int(_as_float(row, "case_count")) for row in rows)
            lines.append(
                "| "
                + " | ".join(
                    [
                        engine,
                        str(case_count),
                        _percent(_mean([_as_float(row, "mean_feasible_rate") for row in rows])),
                        _percent(
                            _mean([_as_float(row, "fully_feasible_case_rate") for row in rows])
                        ),
                        _percent(
                            _mean([_as_float(row, "stability_evaluable_case_rate") for row in rows])
                        ),
                        _number(
                            _mean(
                                [
                                    _as_float(
                                        row,
                                        "mean_feasible_only_jaccard_changed_features",
                                    )
                                    for row in rows
                                ]
                            )
                        ),
                        _number(
                            _mean(
                                [
                                    _as_float(row, "mean_feasible_only_stability_std_norm")
                                    for row in rows
                                ]
                            )
                        ),
                        _number(
                            _mean([_as_float(row, "mean_jaccard_changed_features") for row in rows])
                        ),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No stability summary rows found.")

    lines.extend(["", "## Top Changed Features", ""])
    top_features_by_engine = _top_changed_features_by_engine(candidate_rows, top_n)
    if top_features_by_engine:
        for engine, top_features in top_features_by_engine.items():
            lines.extend([f"### {engine}", ""])
            lines.append("| Feature | Count |")
            lines.append("| --- | ---: |")
            for feature_name, count in top_features:
                lines.append(f"| {feature_name} | {count} |")
            lines.append("")
    else:
        lines.append("No candidate deltas found.")

    lines.extend(
        [
            "",
            "## Design Notes",
            "",
            "- Engine comparison uses the same scenario configs, limits, repeat count, "
            "and timeouts.",
            "- `completed` means the subprocess finished; success is measured by feasible "
            "rate and violations.",
            "- Scenario summary rates are computed from the top-ranked candidate per request; "
            "all-candidate statistics remain available in raw CSV outputs.",
            "- Feasible-only stability is computed only from repeats that produced "
            "FEASIBLE counterfactuals.",
            "- Small limits are smoke-level evidence. Increase limits before drawing final "
            "research claims.",
            "",
            "## Important Files",
            "",
            "- `comparison_manifest.json`",
            "- `comparison_report.md`",
            "- `combined/scenario_summary.csv`",
            "- `combined/stability_summary.csv`",
            "- `combined/inputs.csv`",
            "- `combined/candidates.csv`",
            "- `baselines/<engine>/<timestamp>/report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_report(comparison_root: Path, top_n: int = 10) -> Path:
    report_path = comparison_root / "comparison_report.md"
    report_path.write_text(
        build_comparison_report(comparison_root, top_n=top_n),
        encoding="utf-8",
    )
    return report_path


def run_comparison(
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
) -> Path:
    comparison_root = _make_comparison_root(output_root)
    engine_names = [_engine_name_from_config(path) for path in engine_config_paths]
    print(f"Comparison root: {comparison_root}", flush=True)
    print(
        "Design="
        f"engines={engine_names} | scenarios={len(scenario_config_paths)} | "
        f"scenario_limit={scenario_limit} | stability_limit={stability_limit} | "
        f"repeat_count={repeat_count}",
        flush=True,
    )

    baseline_roots: list[Path] = []
    for engine_config_path in engine_config_paths:
        engine_name = _engine_name_from_config(engine_config_path)
        print(f"Running comparison baseline for engine '{engine_name}'...", flush=True)
        baseline_root = run_engine_baseline(
            output_root=comparison_root,
            engine_config_path=engine_config_path,
            scenario_config_paths=scenario_config_paths,
            stability_config_path=stability_config_path,
            scenario_limit=scenario_limit,
            stability_limit=stability_limit,
            repeat_count=repeat_count,
            skip_stability=skip_stability,
            scenario_timeout_seconds=scenario_timeout_seconds,
            stability_timeout_seconds=stability_timeout_seconds,
        )
        baseline_roots.append(baseline_root)

    print("Collecting comparison outputs...", flush=True)
    collected = collect_results(comparison_root)
    manifest = {
        "comparison_root": str(comparison_root),
        "engines": engine_names,
        "engine_configs": [str(path) for path in engine_config_paths],
        "scenario_configs": [str(path) for path in scenario_config_paths],
        "stability_config": str(stability_config_path),
        "scenario_limit": scenario_limit,
        "stability_limit": stability_limit,
        "repeat_count": repeat_count,
        "skip_stability": skip_stability,
        "scenario_timeout_seconds": scenario_timeout_seconds,
        "stability_timeout_seconds": stability_timeout_seconds,
        "baseline_roots": [str(path) for path in baseline_roots],
        "collected_outputs": {key: str(value) for key, value in collected.items()},
    }
    _write_json(comparison_root / "comparison_manifest.json", manifest)
    report_path = write_comparison_report(comparison_root)
    latest_path = _write_latest_pointer(output_root, comparison_root)
    print(f"Comparison manifest written to {comparison_root / 'comparison_manifest.json'}.")
    print(f"Comparison report written to {report_path}.")
    print(f"Latest comparison pointer written to {latest_path}.")
    return comparison_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fair comparison across experiment engines.")
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
    args = parser.parse_args()

    output_dir = run_comparison(
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
    )
    print(f"Comparison output written to {output_dir}")


if __name__ == "__main__":
    main()
