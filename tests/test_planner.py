from datetime import datetime, timezone

from diabetify_cf.config import Settings
from diabetify_cf.planner.factory import build_planner
from diabetify_cf.planner.template_planner import TemplatePrescriptivePlanner
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    PredictionInfo,
)


def _request_payload() -> dict:
    return {
        "request_id": "req-plan-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "xgb_v3",
        "target": {"target_class": "low_risk", "min_target_probability": 0.5},
        "instance": {"features": {"age": 45, "BMI": 31.2}},
        "constraints": {
            "immutable_features": ["age"],
            "mutable_allowed": ["BMI"],
            "feature_bounds": {"BMI": {"min": 18.5, "max": 35}},
            "must_not_change": [],
            "medical_rule_set_version": "med_rule_v1",
        },
    }


def _candidate() -> CounterfactualCandidate:
    return CounterfactualCandidate(
        candidate_id="cf_1",
        features={"age": 45, "BMI": 27.8},
        delta={"BMI": -3.4},
        prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.74),
        metrics=CandidateMetrics(
            distance_l1=0.18,
            changed_feature_count=1,
            lof_score=1.05,
            constraint_violations=0,
        ),
    )


def test_template_planner_generates_non_empty_plan() -> None:
    request = CounterfactualRequest.model_validate(_request_payload())
    planner = TemplatePrescriptivePlanner()

    plan = planner.build_plan(request=request, candidate=_candidate())

    assert plan.generation_mode == "template"
    assert plan.intended_user == "clinician"
    assert plan.clinical_scope == "clinician_support"
    assert len(plan.summary) > 0
    assert len(plan.action_steps) > 0
    assert len(plan.monitoring_plan) > 0
    assert plan.human_review_required is True
    assert len(plan.missing_context) > 0


def test_template_planner_can_render_patient_safe_output() -> None:
    request = CounterfactualRequest.model_validate(_request_payload())
    planner = TemplatePrescriptivePlanner(intended_user="patient")

    plan = planner.build_plan(request=request, candidate=_candidate())

    assert plan.intended_user == "patient"
    assert plan.clinical_scope == "patient_education"
    assert plan.human_review_required is False
    assert "edukatif" in plan.summary.lower()


def test_factory_falls_back_to_template_when_openai_key_missing() -> None:
    settings = Settings(
        planner_enabled=True,
        planner_provider="openai",
        openai_api_key="",
    )

    planner = build_planner(settings)

    assert isinstance(planner, TemplatePrescriptivePlanner)
