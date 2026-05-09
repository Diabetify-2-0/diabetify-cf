from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import cast

from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.planner.policy import build_policy_result, normalize_intended_user
from diabetify_cf.schemas import (
    CounterfactualCandidate,
    CounterfactualRequest,
    PlannerInput,
    PrescriptivePlan,
)


class OpenAIPrescriptivePlanner(PrescriptivePlanner):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_ms: int = 4000,
        temperature: float = 0.2,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        intended_user: str = "clinician",
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI prescriptive planner.")

        self.api_key = api_key
        self.model = model
        self.timeout_sec = max(1.0, float(timeout_ms) / 1000.0)
        self.temperature = max(0.0, min(float(temperature), 1.0))
        self.endpoint = endpoint
        self.intended_user = normalize_intended_user(intended_user)

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
            intended_user=self.intended_user,
        )
        prompt = self._build_prompt(
            request=request,
            candidate=candidate,
            planner_input=planner_input,
            policy=policy,
        )
        raw_text = self._call_openai(prompt)
        payload = self._parse_json(raw_text)

        return PrescriptivePlan(
            generation_mode="llm",
            provider=f"openai:{self.model}",
            intended_user=policy.intended_user,
            clinical_scope=policy.clinical_scope,
            policy_version=policy.policy_version,
            summary=str(payload.get("summary", policy.summary)).strip(),
            goals=self._string_list(payload, "goals"),
            action_steps=self._string_list(payload, "action_steps"),
            safety_notes=self._string_list(payload, "safety_notes"),
            monitoring_plan=self._string_list(payload, "monitoring_plan"),
            missing_context=policy.missing_context,
            contraindication_flags=policy.contraindication_flags,
            human_review_required=policy.human_review_required,
            disclaimer=str(
                payload.get(
                    "disclaimer",
                    "Panduan ini bersifat edukatif dan tidak menggantikan konsultasi dokter.",
                )
            ).strip(),
        )

    def _string_list(self, payload: dict[str, object], key: str) -> list[str]:
        raw_items = payload.get(key, [])
        if not isinstance(raw_items, list):
            return []
        return [str(item).strip() for item in raw_items if str(item).strip()]

    def _build_prompt(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
        planner_input: PlannerInput,
        policy: object,
    ) -> str:
        compact = {
            "intended_user": getattr(policy, "intended_user"),
            "clinical_scope": getattr(policy, "clinical_scope"),
            "target_class": request.target.target_class,
            "min_target_probability": request.target.min_target_probability,
            "input_prediction": (
                planner_input.input_prediction.to_wire()
                if planner_input.input_prediction is not None
                else None
            ),
            "candidate_prediction": candidate.prediction.to_wire(),
            "candidate_metrics": candidate.metrics.model_dump(),
            "target_deltas": planner_input.target_deltas,
            "changed_features": [item.model_dump() for item in planner_input.changed_features],
            "mutable_allowed": planner_input.mutable_allowed,
            "immutable_features": planner_input.immutable_features,
            "must_not_change": planner_input.must_not_change,
            "policy_summary": getattr(policy, "summary"),
            "policy_goals": getattr(policy, "goals"),
            "policy_action_steps": getattr(policy, "action_steps"),
            "policy_safety_notes": getattr(policy, "safety_notes"),
            "policy_monitoring_plan": getattr(policy, "monitoring_plan"),
            "missing_context": getattr(policy, "missing_context"),
            "contraindication_flags": getattr(policy, "contraindication_flags"),
            "human_review_required": getattr(policy, "human_review_required"),
        }
        return (
            "Anda adalah asisten untuk merapikan output decision-support kesehatan. "
            "Jangan membuat instruksi terapi baru. Jangan menambah target numerik baru. "
            "Jangan menyebut fitur sebagai berubah jika fitur itu tidak muncul pada changed_features. "
            "Gunakan before/after dari changed_features jika menjelaskan arah perubahan. "
            "Gunakan policy_summary dan policy_action_steps sebagai batas keras. "
            "Jika intended_user=patient, gunakan bahasa edukatif dan non-direktif. "
            "Jika intended_user=clinician, gunakan bahasa decision-support yang menekankan perlunya penilaian profesional. "
            "Jawab ketat dalam JSON object dengan kunci: "
            "summary, goals (array), action_steps (array), safety_notes (array), "
            "monitoring_plan (array), disclaimer. "
            "Gunakan Bahasa Indonesia yang jelas dan praktis.\n\n"
            f"Data kasus: {json.dumps(compact, ensure_ascii=True)}"
        )

    def _call_openai(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                response_payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI HTTP error: {err.code} {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"OpenAI request failed: {err}") from err

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise RuntimeError("OpenAI response format unexpected for planner output.") from err

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI returned empty planner content.")
        return content

    def _parse_json(self, text: str) -> dict[str, object]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            return cast(dict[str, object], json.loads(cleaned))
        except json.JSONDecodeError as err:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                return cast(dict[str, object], json.loads(cleaned[start : end + 1]))
            raise RuntimeError("Planner output is not valid JSON.") from err
