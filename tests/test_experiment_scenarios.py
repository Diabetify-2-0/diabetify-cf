from experiments.scripts.evaluate_stability import _aggregate_summary
from experiments.scripts.run_scenarios import _flatten_summary


def test_aggregate_stability_summary_computes_means() -> None:
    summary = _aggregate_summary(
        [
            {
                "feasible_rate": 1.0,
                "mean_jaccard_changed_features": 0.5,
                "stability_std_norm": 0.1,
            },
            {
                "feasible_rate": 0.5,
                "mean_jaccard_changed_features": 1.0,
                "stability_std_norm": 0.3,
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["mean_feasible_rate"] == 0.75
    assert summary["mean_jaccard_changed_features"] == 0.75
    assert summary["mean_stability_std_norm"] == 0.2


def test_flatten_summary_serializes_nested_counts() -> None:
    row = _flatten_summary(
        {
            "total_cases": 2,
            "status_counts": {"FEASIBLE": 1, "INFEASIBLE": 1},
        },
        scenario_name="test_scenario",
    )

    assert row["scenario"] == "test_scenario"
    assert row["total_cases"] == 2
    assert row["status_counts"] == '{"FEASIBLE": 1, "INFEASIBLE": 1}'
