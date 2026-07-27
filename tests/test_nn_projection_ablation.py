from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from diabetify_cf.engine.feature_registry import FeatureDefinition
from diabetify_cf.engine.nn_engine import NearestNeighborCounterfactualEngine
from diabetify_cf.experiments.nn_projection_ablation import (
    NNProjectionAblation,
    NNProjectionAblationConfig,
    ProfileEvaluation,
    ProjectionResult,
    SelectedProfile,
)


def _feature(name: str, *, span: float, cost_weight: float) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type="continuous",
        immutable=False,
        actionable=True,
        default_mutable=True,
        global_min=0.0,
        global_max=span,
        cost_weight=cost_weight,
        preferred_direction="any",
        aliases=[],
    )


def _config() -> NNProjectionAblationConfig:
    return NNProjectionAblationConfig(
        profile_count=1,
        target_class="low_risk",
        min_target_probability=0.5,
        mutable_allowed=("cheap", "expensive"),
        candidate_pool_size=10,
        max_neighbors=5,
        profile_selection_seed=42,
    )


def test_projection_specs_use_cheapest_prefixes_and_keep_full_variant() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    engine.artifacts = SimpleNamespace(
        feature_columns=["fixed", "cheap", "expensive"],
        feature_registry=SimpleNamespace(
            get=lambda name: {
                "cheap": _feature("cheap", span=100.0, cost_weight=1.0),
                "expensive": _feature("expensive", span=10.0, cost_weight=2.0),
            }.get(name),
            coerce_value=lambda _name, value: float(value),
        ),
    )
    experiment = NNProjectionAblation.__new__(NNProjectionAblation)
    experiment.engine = engine
    experiment.config = _config()
    experiment.artifacts = engine.artifacts
    prepared = SimpleNamespace(
        instance_features={"fixed": 7.0, "cheap": 10.0, "expensive": 1.0},
        mutable_allowed=["cheap", "expensive"],
        registry=engine.artifacts.feature_registry,
    )
    neighbor = pd.Series({"fixed": 999.0, "cheap": 20.0, "expensive": 4.0})

    specs = experiment._build_projection_specs(
        ranked_neighbors=[neighbor],
        prepared=prepared,
    )

    full = next(item for item in specs if item.method == "full")
    sparse = [item for item in specs if item.method == "sparse"]
    assert full.features == {"fixed": 7.0, "cheap": 20.0, "expensive": 4.0}
    assert [item.prefix_length for item in sparse] == [1, 2]
    assert sparse[0].features == {"fixed": 7.0, "cheap": 20.0, "expensive": 1.0}
    assert sparse[1].features == full.features
    assert full.ordered_changed_features == ("cheap", "expensive")


def test_sparse_selection_uses_smallest_valid_prefix_from_full_neighbor() -> None:
    results = [
        _result(method="sparse", changed=2, rank=1, proximity=0.01),
        _result(method="sparse", changed=1, rank=4, proximity=0.20),
        _result(method="full", changed=3, rank=1, proximity=0.30),
    ]
    full = NNProjectionAblation._select_full(results)

    selected = NNProjectionAblation._select_sparse_from_same_neighbor(
        results,
        full=full,
    )

    assert selected is not None
    assert selected.changed_feature_count == 2
    assert selected.neighbor_rank == 1


def test_config_rejects_non_low_risk_target() -> None:
    config = NNProjectionAblationConfig(
        profile_count=20,
        target_class="high_risk",
        min_target_probability=0.5,
        mutable_allowed=("BMI",),
        candidate_pool_size=256,
        max_neighbors=64,
        profile_selection_seed=42,
    )

    with pytest.raises(ValueError, match="low_risk"):
        config.validated()


def test_report_summary_only_contains_mean_metrics_and_reduction_percentages() -> None:
    experiment = NNProjectionAblation.__new__(NNProjectionAblation)
    experiment.config = _config()
    experiment.profile_input_path = None
    selected = [
        SelectedProfile(
            profile_id="profile-1",
            risk_stratum=1,
            evaluation=ProfileEvaluation(
                reference_index=10,
                baseline_features={"x": 0.0},
                baseline_probability_low_risk=0.2,
                mutable_allowed=("x",),
                full=_result(method="full", changed=3, rank=1, proximity=0.3),
                sparse=_result(method="sparse", changed=1, rank=1, proximity=0.1),
            ),
        ),
        SelectedProfile(
            profile_id="profile-2",
            risk_stratum=1,
            evaluation=ProfileEvaluation(
                reference_index=11,
                baseline_features={"x": 0.0},
                baseline_probability_low_risk=0.2,
                mutable_allowed=("x",),
                full=_result(method="full", changed=2, rank=1, proximity=0.2),
                sparse=_result(method="sparse", changed=2, rank=1, proximity=0.2),
            ),
        ),
    ]

    report = experiment._build_report_payload(
        selected,
        selection_summary={"selected_profile_count": 2},
    )

    assert report["summary"] == {
        "valid_pair_count": 2,
        "full_mean_proximity": 0.25,
        "sparse_mean_proximity": 0.15000000000000002,
        "full_mean_changed_feature_count": 2.5,
        "sparse_mean_changed_feature_count": 1.5,
        "proximity_reduction_percent": pytest.approx(40.0),
        "changed_feature_count_reduction_percent": pytest.approx(40.0),
    }


def test_report_omits_null_reference_index_and_risk_stratum() -> None:
    experiment = NNProjectionAblation.__new__(NNProjectionAblation)
    experiment.config = _config()
    experiment.profile_input_path = "evaluation/fixtures/profile_input.json"
    selected = [
        SelectedProfile(
            profile_id="profile-1",
            risk_stratum=None,
            evaluation=ProfileEvaluation(
                reference_index=None,
                baseline_features={"x": 0.0},
                baseline_probability_low_risk=0.2,
                mutable_allowed=("x",),
                full=_result(method="full", changed=1, rank=1, proximity=0.1),
                sparse=_result(method="sparse", changed=1, rank=1, proximity=0.1),
            ),
        )
    ]

    report = experiment._build_report_payload(
        selected,
        selection_summary={"selected_profile_count": 1},
    )

    profile_payload = report["profiles"][0]
    assert "reference_index" not in profile_payload
    assert "risk_stratum" not in profile_payload


def test_fixed_profile_report_config_omits_sampling_fields() -> None:
    experiment = NNProjectionAblation.__new__(NNProjectionAblation)
    experiment.config = _config()
    experiment.profile_input_path = "evaluation/fixtures/profile_input.json"
    selected = [
        SelectedProfile(
            profile_id="profile-1",
            risk_stratum=None,
            evaluation=ProfileEvaluation(
                reference_index=None,
                baseline_features={"x": 0.0},
                baseline_probability_low_risk=0.2,
                mutable_allowed=("x",),
                full=_result(method="full", changed=1, rank=1, proximity=0.1),
                sparse=_result(method="sparse", changed=1, rank=1, proximity=0.1),
            ),
        )
    ]

    report = experiment._build_report_payload(
        selected,
        selection_summary={"selected_profile_count": 1},
    )

    assert report["config"] == {
        "target_class": "low_risk",
        "min_target_probability": 0.5,
        "candidate_pool_size": 10,
        "max_neighbors": 5,
    }


def _result(
    *,
    method: str,
    changed: int,
    rank: int,
    proximity: float,
) -> ProjectionResult:
    return ProjectionResult(
        method=method,
        features={"x": float(changed)},
        delta={"x": float(changed)},
        probability_low_risk=0.75,
        proximity_normalized_l1=proximity,
        changed_feature_count=changed,
        weighted_action_cost=proximity,
        neighbor_rank=rank,
        prefix_length=changed,
        ordered_changed_features=("x",),
    )
