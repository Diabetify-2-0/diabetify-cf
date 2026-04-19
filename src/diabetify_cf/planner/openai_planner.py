from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import cast

from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.schemas import CounterfactualCandidate, CounterfactualRequest, PrescriptivePlan


class OpenAIPrescriptivePlanner(PrescriptivePlanner):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_ms: int = 4000,
        temperature: float = 0.2,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI prescriptive planner.")

        self.api_key = api_key
        self.model = model
        self.timeout_sec = max(1.0, float(timeout_ms) / 1000.0)
        self.temperature = max(0.0, min(float(temperature), 1.0))
        self.endpoint = endpoint

    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
    ) -> PrescriptivePlan:
        prompt = self._build_prompt(request=request, candidate=candidate)
        raw_text = self._call_openai(prompt)
        payload = self._parse_json(raw_text)

        return PrescriptivePlan(
            generation_mode="llm",
            provider=f"openai:{self.model}",
            summary=str(payload.get("summary", "")).strip(),
            goals=self._string_list(payload, "goals"),
            action_steps=self._string_list(payload, "action_steps"),
            safety_notes=self._string_list(payload, "safety_notes"),
            monitoring_plan=self._string_list(payload, "monitoring_plan"),
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
    ) -> str:
        compact = {
            "target_class": request.target.target_class,
            "min_target_probability": request.target.min_target_probability,
            "candidate_prediction": candidate.prediction.to_wire(),
            "target_deltas": candidate.delta,
            "mutable_allowed": request.constraints.mutable_allowed,
            "immutable_features": request.constraints.immutable_features,
        }
        return (
            "Anda adalah asisten kesehatan untuk perencanaan perubahan gaya hidup. "
            "Buat rencana preskriptif yang realistis, bertahap, dan aman, "
            "dengan mematuhi immutable_features dan hanya menindaklanjuti target_deltas. "
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
        try:
            return cast(dict[str, object], json.loads(text))
        except json.JSONDecodeError as err:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return cast(dict[str, object], json.loads(text[start : end + 1]))
            raise RuntimeError("Planner output is not valid JSON.") from err
