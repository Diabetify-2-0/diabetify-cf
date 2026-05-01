import csv
from pathlib import Path

from experiments.scripts.run_comparison import (
    build_comparison_report,
    engine_from_source_file,
    write_comparison_report,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_engine_from_source_file_reads_baseline_layout() -> None:
    source_file = str(
        Path("experiments/results/comparisons/20260501_000000")
        / "baselines"
        / "dice"
        / "20260501_000001"
        / "combined"
        / "scenario_summary.csv"
    )

    assert engine_from_source_file(source_file) == "dice"


def test_build_comparison_report_contains_engine_summary(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "completed",
                "feasible_rate": "1.0",
                "target_success_rate": "1.0",
                "mean_runtime_ms": "12.0",
                "step_runtime_seconds": "2.0",
                "reason_counts": '{"OK": 1}',
            }
        ],
    )
    _write_csv(
        tmp_path / "baselines" / "ocean" / "run-1" / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "completed",
                "feasible_rate": "0.5",
                "target_success_rate": "1.0",
                "mean_runtime_ms": "50.0",
                "step_runtime_seconds": "4.0",
                "reason_counts": '{"OK": 1, "TARGET_UNREACHABLE_UNDER_CONSTRAINTS": 1}',
            }
        ],
    )
    _write_csv(
        tmp_path
        / "baselines"
        / "dice"
        / "run-1"
        / "stability"
        / "run-1"
        / "stability_aggregate.csv",
        [
            {
                "case_count": "1",
                "mean_feasible_rate": "1.0",
                "fully_feasible_case_rate": "1.0",
                "stability_evaluable_case_rate": "1.0",
                "mean_jaccard_changed_features": "1.0",
                "mean_stability_std_norm": "0.0",
                "mean_feasible_only_jaccard_changed_features": "1.0",
                "mean_feasible_only_stability_std_norm": "0.0",
            }
        ],
    )
    _write_csv(
        tmp_path
        / "baselines"
        / "dice"
        / "run-1"
        / "scenarios"
        / "all_mutable"
        / "run-1"
        / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": '{"BMI": -1.0}'}],
    )

    report = build_comparison_report(tmp_path)

    assert "# Comparison Report" in report
    assert "| dice | 1 | 1 | 0 | 0 | 100.0% | 1 |" in report
    assert "| ocean | 1 | 1 | 0 | 0 | 50.0% | 0 |" in report
    assert "Scenario Matrix" in report
    assert "Feasible-only Jaccard" in report
    assert "BMI" in report


def test_write_comparison_report_creates_report_file(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "baselines" / "dice" / "run-1" / "scenarios" / "scenario_summary.csv",
        [{"scenario": "all_mutable", "feasible_rate": "1.0"}],
    )

    report_path = write_comparison_report(tmp_path)

    assert report_path == tmp_path / "comparison_report.md"
    assert report_path.exists()
