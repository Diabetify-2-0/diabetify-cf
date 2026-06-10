from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.schemas import CounterfactualRequest
from experiments.evaluation import evaluate_candidate


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        version="test_v1",
        features=[
            FeatureDefinition(
                name="age",
                feature_type="continuous",
                immutable=True,
                actionable=False,
                default_mutable=False,
                global_min=18,
                global_max=100,
                cost_weight=10.0,
                preferred_direction="any",
                aliases=[],
            ),
            FeatureDefinition(
                name="BMI",
                feature_type="continuous",
                immutable=False,
                actionable=True,
                default_mutable=True,
                global_min=10,
                global_max=60,
                cost_weight=1.0,
                preferred_direction="decrease",
                aliases=["bmi"],
            ),
            FeatureDefinition(
                name="is_bloodline",
                feature_type="binary",
                immutable=True,
                actionable=False,
                default_mutable=False,
                global_min=0,
                global_max=1,
                cost_weight=10.0,
                preferred_direction="any",
                aliases=[],
            ),
        ],
    )


def _request() -> CounterfactualRequest:
    return CounterfactualRequest.model_validate(
        {
            "request_id": "req-eval",
            "instance": {"features": {"age": 45, "BMI": 31.2, "is_bloodline": 1}},
            "target": {"target_class": "low_risk", "min_target_probability": 0.5},
            "constraints": {
                "immutable_features": ["age", "is_bloodline"],
                "mutable_allowed": ["BMI"],
            },
        }
    )


def test_evaluate_candidate_marks_valid_candidate() -> None:
    report = evaluate_candidate(
        candidate={
            "features": {"age": 45, "BMI": 29.0, "is_bloodline": 1},
            "prediction": {"class": "low_risk", "probability_low_risk": 0.75},
            "metrics": {"lof_score": 1.1},
        },
        baseline={"age": 45, "BMI": 31.2, "is_bloodline": 1},
        request=_request(),
        registry=_registry(),
        max_lof_score=2.5,
    )

    assert report["target_success"] is True
    assert report["immutable_violation_count"] == 0
    assert report["mutable_violation_count"] == 0
    assert report["directional_violation_count"] == 0
    assert report["plausibility_pass"] is True


def test_evaluate_candidate_counts_constraint_violations() -> None:
    report = evaluate_candidate(
        candidate={
            "features": {"age": 46, "BMI": 37.0, "is_bloodline": 0},
            "prediction": {"class": "high_risk", "probability_low_risk": 0.2},
            "metrics": {"lof_score": 3.0},
        },
        baseline={"age": 45, "BMI": 31.2, "is_bloodline": 1},
        request=_request(),
        registry=_registry(),
        max_lof_score=2.5,
    )

    assert report["target_success"] is False
    assert report["immutable_violation_count"] == 2
    assert report["mutable_violation_count"] == 2
    assert report["directional_violation_count"] == 1
    assert report["plausibility_pass"] is False
