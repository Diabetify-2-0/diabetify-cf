import csv
from pathlib import Path

from experiments.scripts.print_baseline_report import _scenario_rows, _top_changed_features


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
