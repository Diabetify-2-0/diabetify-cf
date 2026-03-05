from __future__ import annotations

from diabetify_cf.config import Settings
from diabetify_cf.planner.base import FallbackPrescriptivePlanner, PrescriptivePlanner
from diabetify_cf.planner.openai_planner import OpenAIPrescriptivePlanner
from diabetify_cf.planner.template_planner import TemplatePrescriptivePlanner


def build_planner(settings: Settings) -> PrescriptivePlanner | None:
    if not settings.planner_enabled:
        return None

    template_planner = TemplatePrescriptivePlanner(max_steps=settings.planner_max_steps)
    provider = settings.planner_provider.strip().lower()

    if provider == "openai":
        if not settings.openai_api_key:
            return template_planner

        openai_planner = OpenAIPrescriptivePlanner(
            api_key=settings.openai_api_key,
            model=settings.planner_model,
            timeout_ms=settings.planner_timeout_ms,
            temperature=settings.planner_temperature,
            endpoint=settings.openai_endpoint,
        )
        return FallbackPrescriptivePlanner(primary=openai_planner, fallback=template_planner)

    return template_planner
