from __future__ import annotations

from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.planner.policy import build_policy_result
from diabetify_cf.schemas import (
    CounterfactualCandidate,
    CounterfactualRequest,
    PlannerInput,
    PrescriptivePlan,
)


class TemplatePrescriptivePlanner(PrescriptivePlanner):
    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps

    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
        planner_input: PlannerInput,
    ) -> PrescriptivePlan:
        policy = build_policy_result(
            request=request,
            candidate=candidate,
            planner_input=planner_input,
        )

        return PrescriptivePlan(
            generation_mode="template",
            provider="template_v1",
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
