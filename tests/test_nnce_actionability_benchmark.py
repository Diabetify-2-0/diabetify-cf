from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diabetify_cf.engine.artifacts import ModelArtifacts
from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.experiments.nnce_actionability_benchmark import (
    NNCEActionabilityBenchmark,
    heom_distance,
)
from diabetify_cf.reason_codes import ReasonCode, Status
from diabetify_cf.schemas import CounterfactualRequest
from diabetify_cf.verification.runner import ScenarioExpectation, VerificationScenario


@dataclass
class FakeModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = []
        for _, row in frame.iterrows():
            low_risk_probability = 0.8 if float(row["BMI"]) <= 28.0 else 0.2
            probabilities.append([low_risk_probability, 1.0 - low_risk_probability])
        return np.asarray(probabilities, dtype=float)


def _feature(
    name: str,
    *,
    feature_type: str,
    immutable: bool = False,
    actionable: bool = True,
    global_min: float = 0.0,
    global_max: float = 1.0,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type=feature_type,
        immutable=immutable,
        actionable=actionable,
        default_mutable=actionable,
        global_min=global_min,
        global_max=global_max,
        cost_weight=1.0,
        preferred_direction="any",
        aliases=[],
    )


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        version="test_v1",
        features=[
            _feature(
                "age",
                feature_type="continuous",
                immutable=True,
                actionable=False,
                global_min=18.0,
                global_max=100.0,
            ),
            _feature(
                "BMI",
                feature_type="continuous",
                global_min=10.0,
                global_max=60.0,
            ),
            _feature(
                "is_hypertension",
                feature_type="binary",
                global_min=0.0,
                global_max=1.0,
            ),
        ],
    )


def _artifacts() -> ModelArtifacts:
    return ModelArtifacts(
        model=FakeModel(),
        feature_columns=["age", "BMI", "is_hypertension"],
        reference_data=pd.DataFrame(
            [
                {"age": 60, "BMI": 35.0, "is_hypertension": 1},
                {"age": 30, "BMI": 27.0, "is_hypertension": 0},
            ]
        ),
        feature_registry=_registry(),
        lof_model=None,
    )


def _scenario() -> VerificationScenario:
    request = CounterfactualRequest.model_validate(
        {
            "request_id": "nnce-actionability-test",
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "instance": {
                "features": {
                    "age": 60,
                    "BMI": 35.0,
                    "is_hypertension": 1,
                }
            },
            "constraints": {"mutable_allowed": ["BMI"]},
        }
    )
    return VerificationScenario(
        name="nnce_actionability_case",
        request=request,
        expectation=ScenarioExpectation(
            expected_status=Status.FEASIBLE,
            expected_reason_codes=(ReasonCode.OK,),
        ),
    )


def test_heom_distance_uses_normalized_numeric_and_binary_overlap() -> None:
    registry = _registry()

    distance = heom_distance(
        baseline={"age": 60, "BMI": 35.0, "is_hypertension": 1},
        candidate={"age": 30, "BMI": 27.0, "is_hypertension": 0},
        feature_names=("age", "BMI", "is_hypertension"),
        registry=registry,
    )

    expected = np.sqrt(((30.0 / 82.0) ** 2) + ((8.0 / 50.0) ** 2) + 1.0)
    assert distance == expected


def test_benchmark_compares_pure_nnce_against_adapted_projection() -> None:
    benchmark = NNCEActionabilityBenchmark(
        artifacts=_artifacts(),
        scenarios=[_scenario()],
    )

    payload = benchmark.run()

    summary = payload["summary"]
    assert summary["pure_nnce"]["immutable_violation_rate"] == 1.0
    assert summary["pure_nnce"]["outside_selected_mutable_violation_rate"] == 1.0
    assert summary["pure_nnce"]["average_changed_immutable_feature_count_all_scenarios"] == 1.0
    assert summary["pure_nnce"]["changed_immutable_feature_frequency"] == [
        {"feature_name": "age", "changed_count": 1, "changed_rate": 1.0}
    ]
    assert summary["adapted_nnce"]["immutable_violation_rate"] == 0.0
    assert summary["adapted_nnce"]["outside_selected_mutable_violation_rate"] == 0.0

    scenario = payload["scenarios"][0]
    assert scenario["pure_nnce"]["candidate_features"] == {
        "age": 30.0,
        "BMI": 27.0,
        "is_hypertension": 0,
    }
    assert scenario["adapted_nnce"]["candidate_features"] == {
        "age": 60,
        "BMI": 27.0,
        "is_hypertension": 1,
    }
