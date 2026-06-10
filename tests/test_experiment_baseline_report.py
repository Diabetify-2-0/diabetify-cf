import csv
from pathlib import Path

from experiments.scripts.print_baseline_report import (
    _scenario_rows,
    _top_changed_features,
    build_markdown_report,
    write_markdown_report,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_scenario_rows_can_read_partial_baseline_output(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "scenarios" / "all_mutable" / "run-1" / "summary.csv",
        [{"feasible_rate": "1.0", "target_success_rate": "1.0"}],
    )

    rows = _scenario_rows(tmp_path)

    assert rows[0]["scenario"] == "all_mutable"
    assert rows[0]["feasible_rate"] == "1.0"


def test_top_changed_features_counts_delta_keys() -> None:
    rows = [
        {"delta": '{"BMI": -1.0, "is_hypertension": -1.0}'},
        {"delta": '{"BMI": -2.0}'},
    ]

    top_features = _top_changed_features(rows, top_n=2)

    assert top_features == [("BMI", 2), ("is_hypertension", 1)]


def test_build_markdown_report_contains_scenario_and_stability_sections(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "scenarios" / "scenario_summary.csv",
        [
            {
                "scenario": "all_mutable",
                "step_status": "completed",
                "feasible_rate": "1.0",
                "target_success_rate": "1.0",
                "immutable_violation_rate": "0.0",
                "mutable_violation_rate": "0.0",
                "directional_violation_rate": "0.0",
                "mean_runtime_ms": "12.0",
                "mean_lof_score": "1.0",
            }
        ],
    )
    _write_csv(
        tmp_path / "stability" / "run-1" / "stability_aggregate.csv",
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
        tmp_path / "scenarios" / "all_mutable" / "run-1" / "candidates.csv",
        [{"delta": '{"BMI": -1.0}'}],
    )

    report = build_markdown_report(tmp_path)

    assert "# Baseline Report" in report
    assert "all_mutable" in report
    assert "Feasible-only Jaccard" in report
    assert "BMI" in report


def test_write_markdown_report_creates_report_file(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "combined_scenario_summary.csv",
        [{"scenario": "all_mutable", "feasible_rate": "1.0"}],
    )

    report_path = write_markdown_report(tmp_path)

    assert report_path == tmp_path / "report.md"
    assert report_path.exists()
