from __future__ import annotations

from types import SimpleNamespace

import pytest

from diabetify_cf.experiments.nn_objective_selection_sensitivity import (
    OBJECTIVE_CONFIGS,
    OBJECTIVE_WEIGHT_SWEEP_CONFIGS,
    NNObjectiveSelectionSensitivity,
    ValidProjectionCandidate,
    _proximity_plausibility_overlap,
    _summary_for_selected,
)
from diabetify_cf.schemas import CandidateMetrics, CounterfactualCandidate, PredictionInfo


def _valid_candidate(
    *,
    candidate_id: str,
    proximity: float,
    sparsity: int,
    lof: float,
) -> ValidProjectionCandidate:
    return ValidProjectionCandidate(
        candidate=CounterfactualCandidate(
            candidate_id=candidate_id,
            features={"x": proximity},
            delta={"x": proximity},
            prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.8),
            metrics=CandidateMetrics(
                distance_l1=proximity,
                changed_feature_count=sparsity,
                lof_score=lof,
            ),
        ),
        neighbor_rank=1,
        projection_method="sparse",
        prefix_length=sparsity,
    )


def test_objective_selection_configs_select_different_best_candidates() -> None:
    experiment = NNObjectiveSelectionSensitivity.__new__(NNObjectiveSelectionSensitivity)
    experiment.config = SimpleNamespace(mutable_allowed=("x",))
    experiment.engine = SimpleNamespace(
        artifacts=SimpleNamespace(feature_registry=object()),
        _objective_score=lambda candidate, preferences, **_kwargs: (
            preferences["proximity"] * candidate.metrics.distance_l1
            + preferences["plausibility"] * candidate.metrics.lof_score
        ),
    )
    candidates = [
        _valid_candidate(
            candidate_id="near",
            proximity=0.1,
            sparsity=2,
            lof=2.0,
        ),
        _valid_candidate(
            candidate_id="plausible",
            proximity=0.9,
            sparsity=1,
            lof=0.5,
        ),
    ]

    selected = experiment._select_by_objective_configs(candidates)

    assert set(selected) == set(OBJECTIVE_CONFIGS)
    assert selected["proximity_only"]["candidate_id"] == "near"
    assert selected["plausibility_only"]["candidate_id"] == "plausible"


def test_objective_weight_sweep_configs_use_expected_tradeoff_weights() -> None:
    assert list(OBJECTIVE_WEIGHT_SWEEP_CONFIGS) == [
        "proximity_only",
        "plausibility_only",
        "proximity_75_plausibility_25",
        "proximity_50_plausibility_50",
        "proximity_25_plausibility_75",
    ]
    assert OBJECTIVE_WEIGHT_SWEEP_CONFIGS["proximity_only"] == {
        "proximity": 1.0,
        "plausibility": 0.0,
    }
    assert OBJECTIVE_WEIGHT_SWEEP_CONFIGS["plausibility_only"] == {
        "proximity": 0.0,
        "plausibility": 1.0,
    }
    assert OBJECTIVE_WEIGHT_SWEEP_CONFIGS["proximity_75_plausibility_25"] == {
        "proximity": 0.75,
        "plausibility": 0.25,
    }
    assert OBJECTIVE_WEIGHT_SWEEP_CONFIGS["proximity_50_plausibility_50"] == {
        "proximity": 0.50,
        "plausibility": 0.50,
    }
    assert OBJECTIVE_WEIGHT_SWEEP_CONFIGS["proximity_25_plausibility_75"] == {
        "proximity": 0.25,
        "plausibility": 0.75,
    }
    for weights in OBJECTIVE_WEIGHT_SWEEP_CONFIGS.values():
        assert sum(weights.values()) == pytest.approx(1.0)


def test_objective_selection_can_omit_diagnostics() -> None:
    experiment = NNObjectiveSelectionSensitivity.__new__(NNObjectiveSelectionSensitivity)
    experiment.include_diagnostics = False
    experiment.objective_configs = {"proximity_only": OBJECTIVE_CONFIGS["proximity_only"]}
    experiment.experiment_name = "test_experiment"
    experiment.description = "test description"
    profile = SimpleNamespace(
        profile_id="profile-1",
        reference_index=1,
        request=SimpleNamespace(constraints=SimpleNamespace(mutable_allowed=("x",))),
    )
    selected = {
        "candidate_id": "candidate-1",
        "proximity": 0.1,
        "plausibility_lof": 1.0,
        "target_probability": 0.8,
    }
    experiment._profiles = lambda: [profile]  
    experiment._input_source_payload = lambda: {"type": "test"}  
    valid_candidate = _valid_candidate(
        candidate_id="candidate-1",
        proximity=0.1,
        sparsity=1,
        lof=1.0,
    )
    experiment._valid_candidates_for_profile = lambda _profile: [valid_candidate]  
    experiment._select_by_objective_configs = lambda _candidates, **_kwargs: {  
        "proximity_only": selected
    }

    payload = experiment.run()

    assert "diagnostics" not in payload


def test_objective_selection_summary_reports_candidate_quality_means() -> None:
    summary = _summary_for_selected(
        [
            {
                "proximity": 0.1,
                "plausibility_lof": 1.2,
                "target_probability": 0.7,
            },
            {
                "proximity": 0.3,
                "plausibility_lof": 1.0,
                "target_probability": 0.9,
            },
        ]
    )

    assert summary["profile_count"] == 2
    assert summary["mean_proximity"] == 0.2
    assert summary["mean_plausibility_lof"] == 1.1
    assert "mean_sparsity" not in summary
    assert "mean_action_cost" not in summary
    assert "median_proximity" not in summary
    assert "median_sparsity" not in summary
    assert "median_plausibility_lof" not in summary
    assert "median_action_cost" not in summary
    assert "mean_target_probability" not in summary
    assert "median_target_probability" not in summary


def test_proximity_plausibility_overlap_counts_same_selected_candidate() -> None:
    profiles = [
        {
            "selected_candidates": {
                "proximity_only": {"candidate_id": "a"},
                "plausibility_only": {"candidate_id": "b"},
            }
        },
        {
            "selected_candidates": {
                "proximity_only": {"candidate_id": "c"},
                "plausibility_only": {"candidate_id": "d"},
            }
        },
    ]

    overlap = _proximity_plausibility_overlap(profiles)

    assert overlap == {
        "left_objective": "proximity_only",
        "right_objective": "plausibility_only",
        "same_candidate_count": 0,
        "comparable_profile_count": 2,
        "same_candidate_rate": 0.0,
    }
