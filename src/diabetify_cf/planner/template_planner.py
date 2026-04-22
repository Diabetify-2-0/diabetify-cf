from __future__ import annotations

from diabetify_cf.planner.policy import build_policy_result, normalize_intended_user
from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.schemas import CounterfactualCandidate, CounterfactualRequest, PrescriptivePlan


class TemplatePrescriptivePlanner(PrescriptivePlanner):
    def __init__(self, max_steps: int = 6, intended_user: str = "clinician") -> None:
        self.max_steps = max_steps
        self.intended_user = normalize_intended_user(intended_user)

    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
    ) -> PrescriptivePlan:
        policy = build_policy_result(
            request=request,
            candidate=candidate,
            intended_user=self.intended_user,
        )

        return PrescriptivePlan(
            generation_mode="template",
            provider="template_v1",
            intended_user=policy.intended_user,
            clinical_scope=policy.clinical_scope,
            policy_version=policy.policy_version,
            summary=policy.summary,
            goals=policy.goals,
            action_steps=policy.action_steps[: self.max_steps],
            safety_notes=policy.safety_notes,
            monitoring_plan=policy.monitoring_plan,
            missing_context=policy.missing_context,
            contraindication_flags=policy.contraindication_flags,
            human_review_required=policy.human_review_required,
        )
