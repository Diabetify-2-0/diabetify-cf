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
    assert plan.clinical_scope == "decision_support"
    assert len(plan.summary) > 0
    assert len(plan.action_steps) > 0
    assert len(plan.monitoring_plan) > 0
    assert plan.human_review_required is True
    assert len(plan.missing_context) > 0


def test_factory_falls_back_to_template_when_openai_key_missing() -> None:
    settings = Settings(
        planner_enabled=True,
        planner_provider="openai",
        openai_api_key="",
    )

    planner = build_planner(settings)

    assert isinstance(planner, TemplatePrescriptivePlanner)


def test_factory_builds_template_planner_with_default_config() -> None:
    settings = Settings(
        planner_enabled=True,
        planner_provider="template",
    )

    planner = build_planner(settings)

    assert isinstance(planner, TemplatePrescriptivePlanner)
    assert planner.max_steps == settings.planner_max_steps


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
            "clinical_scope": "decision_support",
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


def test_template_planner_translates_stop_smoking_without_brinkman_wording() -> None:
    request = CounterfactualRequest.model_validate(
        {
            **_request_payload(),
            "instance": {"features": {"age": 45, "smoking_status": 2, "brinkman_index": 3}},
            "constraints": {
                **_request_payload()["constraints"],
                "mutable_allowed": ["smoking_status", "brinkman_index"],
            },
        }
    )
    planner = TemplatePrescriptivePlanner()
    candidate = CounterfactualCandidate(
        candidate_id="cf_smoke_stop",
        features={"age": 45, "smoking_status": 1, "brinkman_index": 3},
        delta={"smoking_status": -1},
        prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.72),
        metrics=CandidateMetrics(
            distance_l1=0.10,
            changed_feature_count=1,
            lof_score=1.01,
            constraint_violations=0,
        ),
    )
    planner_input = PlannerInput(
        recommended_candidate_id="cf_smoke_stop",
        target_deltas={"smoking_status": -1},
        input_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.30),
        candidate_prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.72),
        changed_features=[
            PlannerFeatureChange(
                feature_name="smoking_status",
                baseline_value=2,
                candidate_value=1,
                delta=-1,
                direction="decrease",
            )
        ],
        mutable_allowed=["smoking_status", "brinkman_index"],
        immutable_features=["age"],
        must_not_change=[],
    )

    plan = planner.build_plan(
        request=request,
        candidate=candidate,
        planner_input=planner_input,
    )

    assert any("berhenti merokok" in goal.lower() for goal in plan.goals)
    assert not any("brinkman" in goal.lower() for goal in plan.goals)
    assert any("berhenti merokok" in step.lower() for step in plan.action_steps)


def test_template_planner_translates_brinkman_to_daily_smoking_reduction() -> None:
    request = CounterfactualRequest.model_validate(
        {
            **_request_payload(),
            "instance": {"features": {"age": 45, "smoking_status": 2, "brinkman_index": 3}},
            "constraints": {
                **_request_payload()["constraints"],
                "mutable_allowed": ["smoking_status", "brinkman_index"],
            },
        }
    )
    planner = TemplatePrescriptivePlanner()
    candidate = CounterfactualCandidate(
        candidate_id="cf_smoke_reduce",
        features={"age": 45, "smoking_status": 2, "brinkman_index": 2},
        delta={"brinkman_index": -1},
        prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.64),
        metrics=CandidateMetrics(
            distance_l1=0.12,
            changed_feature_count=1,
            lof_score=1.02,
            constraint_violations=0,
        ),
    )
    planner_input = PlannerInput(
        recommended_candidate_id="cf_smoke_reduce",
        target_deltas={"brinkman_index": -1},
        input_prediction=PredictionInfo(class_name="high_risk", probability_low_risk=0.28),
        candidate_prediction=PredictionInfo(class_name="low_risk", probability_low_risk=0.64),
        changed_features=[
            PlannerFeatureChange(
                feature_name="brinkman_index",
                baseline_value=3,
                candidate_value=2,
                delta=-1,
                direction="decrease",
            )
        ],
        mutable_allowed=["smoking_status", "brinkman_index"],
        immutable_features=["age"],
        must_not_change=[],
    )

    plan = planner.build_plan(
        request=request,
        candidate=candidate,
        planner_input=planner_input,
    )

    assert any("konsumsi rokok harian" in goal.lower() for goal in plan.goals)
    assert not any("brinkman" in goal.lower() for goal in plan.goals)
    assert any("konsumsi rokok harian" in step.lower() for step in plan.action_steps)
