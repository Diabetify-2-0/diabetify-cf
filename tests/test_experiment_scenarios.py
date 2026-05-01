from experiments.scripts.evaluate_stability import _aggregate_summary
from experiments.scripts.run_scenarios import _flatten_summary


def test_aggregate_stability_summary_computes_means() -> None:
    summary = _aggregate_summary(
        [
            {
                "repeat_count": 2,
                "feasible_count": 2,
                "feasible_rate": 1.0,
                "stability_evaluable": True,
                "fully_feasible": True,
                "mean_jaccard_changed_features": 0.5,
                "stability_std_norm": 0.1,
                "feasible_only_mean_jaccard_changed_features": 0.5,
                "feasible_only_stability_std_norm": 0.1,
            },
            {
                "repeat_count": 2,
                "feasible_count": 1,
                "feasible_rate": 0.5,
                "stability_evaluable": False,
                "fully_feasible": False,
                "mean_jaccard_changed_features": 1.0,
                "stability_std_norm": 0.3,
                "feasible_only_mean_jaccard_changed_features": None,
                "feasible_only_stability_std_norm": None,
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["mean_feasible_rate"] == 0.75
    assert summary["fully_feasible_case_count"] == 1
    assert summary["fully_feasible_case_rate"] == 0.5
    assert summary["stability_evaluable_case_count"] == 1
    assert summary["stability_evaluable_case_rate"] == 0.5
    assert summary["mean_jaccard_changed_features"] == 0.75
    assert summary["mean_stability_std_norm"] == 0.2
    assert summary["mean_feasible_only_jaccard_changed_features"] == 0.5
    assert summary["mean_feasible_only_stability_std_norm"] == 0.1


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
