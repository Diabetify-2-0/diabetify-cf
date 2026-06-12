import json
from pathlib import Path

from diabetify_cf.engine.feature_registry import FeatureRegistry
from experiments.scripts.evaluate_stability import _aggregate_summary
from experiments.scripts.run_scenarios import _flatten_summary

SCENARIO_DIR = Path("experiments/configs/scenarios")
GENERATION_FIELDS = {"generation_method", "total_cfs", "timeout_ms"}


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


def test_scenario_configs_do_not_define_engine_generation_fields() -> None:
    for path in SCENARIO_DIR.glob("*.json"):
        config = json.loads(path.read_text(encoding="utf-8"))

        assert GENERATION_FIELDS.isdisjoint(config), path


def test_bmi_activity_smoking_hypertension_uses_actionable_registry_features() -> None:
    registry = FeatureRegistry.from_file("configs/feature_registry.json")
    config = json.loads(
        (SCENARIO_DIR / "bmi_activity_smoking_hypertension.json").read_text(encoding="utf-8")
    )

    assert config["name"] == "bmi_activity_smoking_hypertension"
    assert config["mutable_allowed"] == [
        "BMI",
        "moderate_physical_activity_frequency",
        "smoking_status",
        "is_hypertension",
    ]
    for feature_name in config["mutable_allowed"]:
        feature = registry.get(feature_name)
        assert feature is not None
        assert feature.actionable
        assert not feature.immutable


def test_bmi_activity_cholesterol_uses_actionable_registry_features() -> None:
    registry = FeatureRegistry.from_file("configs/feature_registry.json")
    config = json.loads(
        (SCENARIO_DIR / "bmi_activity_cholesterol.json").read_text(encoding="utf-8")
    )

    assert config["name"] == "bmi_activity_cholesterol"
    assert config["mutable_allowed"] == [
        "BMI",
        "moderate_physical_activity_frequency",
        "is_cholesterol",
    ]
    for feature_name in config["mutable_allowed"]:
        feature = registry.get(feature_name)
        assert feature is not None
        assert feature.actionable
        assert not feature.immutable


def test_default_mutable_uses_default_registry_flags() -> None:
    config = json.loads((SCENARIO_DIR / "default_mutable.json").read_text(encoding="utf-8"))

    assert config["name"] == "default_mutable"
    assert config["mutable_allowed"] == []
    assert config["use_default_mutable"] is True
    assert config["use_default_immutable"] is True


def test_bmi_activity_declares_scenario_role() -> None:
    activity = json.loads((SCENARIO_DIR / "bmi_activity.json").read_text(encoding="utf-8"))

    assert activity["scenario_role"] == "operational_bridge"


def test_bmi_activity_cholesterol_declares_scenario_role() -> None:
    config = json.loads((SCENARIO_DIR / "bmi_activity_cholesterol.json").read_text(encoding="utf-8"))

    assert config["scenario_role"] == "operational_expansion"
