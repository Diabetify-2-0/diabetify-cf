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
        [{"scenario": "dice_all_mutable", "feasible_rate": "1.0"}],
    )
    _write_csv(
        tmp_path / "run_b" / "stability_aggregate.csv",
        [{"case_count": "2", "mean_stability_std_norm": "0.1"}],
    )
    _write_csv(
        tmp_path / "run_c" / "candidates.csv",
        [{"engine_name": "dice", "request_id": "req-1", "delta": "{}"}],
    )

    outputs = collect_results(tmp_path)

    scenario_rows = _read_csv(outputs["scenario_summary"])
    stability_rows = _read_csv(outputs["stability_summary"])
    candidate_rows = _read_csv(outputs["candidates"])

    assert scenario_rows[0]["scenario"] == "dice_all_mutable"
    assert scenario_rows[0]["feasible_rate"] == "1.0"
    assert stability_rows[0]["case_count"] == "2"
    assert stability_rows[0]["mean_stability_std_norm"] == "0.1"
    assert candidate_rows[0]["engine_name"] == "dice"
    assert candidate_rows[0]["request_id"] == "req-1"
