import csv
from pathlib import Path

from experiments.scripts.collect_results import collect_results


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_collect_results_combines_scenario_and_stability_summaries(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "run_a" / "scenario_summary.csv",
        [{"scenario": "all_mutable", "target_success_rate_all_candidates": "1.0"}],
    )
    _write_csv(
        tmp_path / "run_b" / "stability_aggregate.csv",
        [
            {
                "mean_feasible_only_jaccard_changed_features": "0.8",
                "mean_feasible_only_stability_std_norm": "0.1",
            }
        ],
    )
    _write_csv(
        tmp_path / "run_c" / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": "{}"}],
    )
    _write_csv(
        tmp_path / "run_c" / "inputs.csv",
        [{"engine_name": "dice", "request_id": "req-1", "features": "{}"}],
    )

    outputs = collect_results(tmp_path)

    assert outputs["scenario_summary"] == tmp_path / "combined" / "scenario_summary.csv"
    assert outputs["stability_summary"] == tmp_path / "combined" / "stability_summary.csv"
    assert outputs["inputs"] == tmp_path / "combined" / "inputs.csv"
    assert outputs["candidates"] == tmp_path / "combined" / "candidates.csv"

    scenario_rows = _read_csv(outputs["scenario_summary"])
    stability_rows = _read_csv(outputs["stability_summary"])
    input_rows = _read_csv(outputs["inputs"])
    candidate_rows = _read_csv(outputs["candidates"])

    assert scenario_rows[0]["scenario"] == "all_mutable"
    assert scenario_rows[0]["target_success_rate_all_candidates"] == "1.0"
    assert stability_rows[0]["mean_feasible_only_jaccard_changed_features"] == "0.8"
    assert stability_rows[0]["mean_feasible_only_stability_std_norm"] == "0.1"
    assert input_rows[0]["engine_name"] == "dice"
    assert input_rows[0]["request_id"] == "req-1"
    assert "source_file" not in input_rows[0]
    assert candidate_rows[0]["engine_name"] == "dice"
    assert candidate_rows[0]["request_id"] == "req-1"
