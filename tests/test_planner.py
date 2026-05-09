from datetime import datetime, timezone

from diabetify_cf.config import Settings
from diabetify_cf.planner.factory import build_planner
from diabetify_cf.planner.openai_planner import OpenAIPrescriptivePlanner
from diabetify_cf.planner.template_planner import TemplatePrescriptivePlanner
from diabetify_cf.schemas import (
    CandidateMetrics,
    CounterfactualCandidate,
    CounterfactualRequest,
    PlannerFeatureChange,
    PlannerInput,
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


def _planner_input() -> PlannerInput:
    return PlannerInput(
        recommended_candidate_id="cf_1",
        target_deltas={"BMI": -3.4},
        input_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.31),
        candidate_prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.74),
        candidate_metrics=CandidateMetrics(
            distance_l1=0.18,
            changed_feature_count=1,
            lof_score=1.05,
            constraint_violations=0,
        ),
        changed_features=[
            PlannerFeatureChange(
                feature_name="BMI",
                baseline_value=31.2,
                candidate_value=27.8,
                delta=-3.4,
                direction="decrease",
            )
        ],
        mutable_allowed=["BMI"],
        immutable_features=["age"],
        must_not_change=[],
    )


def test_template_planner_generates_non_empty_plan() -> None:
    request = CounterfactualRequest.model_validate(_request_payload())
    planner = TemplatePrescriptivePlanner()

    plan = planner.build_plan(
        request=request,
        candidate=_candidate(),
        planner_input=_planner_input(),
    )

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

    plan = planner.build_plan(
        request=request,
        candidate=_candidate(),
        planner_input=_planner_input(),
    )

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


def test_factory_passes_intended_user_to_template_planner() -> None:
    settings = Settings(
        planner_enabled=True,
        planner_provider="template",
        planner_intended_user="patient",
    )

    planner = build_planner(settings)

    assert isinstance(planner, TemplatePrescriptivePlanner)
    assert planner.intended_user == "patient"


def test_template_planner_uses_feature_change_context() -> None:
    request = CounterfactualRequest.model_validate(_request_payload())
    planner = TemplatePrescriptivePlanner()

    plan = planner.build_plan(
        request=request,
        candidate=_candidate(),
        planner_input=_planner_input(),
    )

    assert any("31.20" in goal and "27.80" in goal for goal in plan.goals)


def test_openai_prompt_includes_rich_counterfactual_context() -> None:
    request = CounterfactualRequest.model_validate(_request_payload())
    planner = OpenAIPrescriptivePlanner(api_key="test-key", model="gpt-4o-mini")
    policy = type(
        "Policy",
        (),
        {
            "intended_user": "clinician",
            "clinical_scope": "clinician_support",
            "summary": "summary",
            "goals": ["goal"],
            "action_steps": ["step"],
            "safety_notes": ["note"],
            "monitoring_plan": ["monitor"],
            "missing_context": ["context"],
            "contraindication_flags": [],
            "human_review_required": True,
        },
    )()

    prompt = planner._build_prompt(
        request=request,
        candidate=_candidate(),
        planner_input=_planner_input(),
        policy=policy,
    )

    assert '"changed_features"' in prompt
    assert '"baseline_value": 31.2' in prompt
    assert '"candidate_metrics"' in prompt
    assert '"input_prediction"' in prompt
