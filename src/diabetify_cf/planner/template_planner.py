from __future__ import annotations

from diabetify_cf.planner.base import PrescriptivePlanner
from diabetify_cf.schemas import CounterfactualCandidate, CounterfactualRequest, PrescriptivePlan


class TemplatePrescriptivePlanner(PrescriptivePlanner):
    def __init__(self, max_steps: int = 6) -> None:
        self.max_steps = max_steps

    def build_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
    ) -> PrescriptivePlan:
        deltas = sorted(
            candidate.delta.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        goals = [self._goal_line(name, delta) for name, delta in deltas]
        action_steps = self._build_actions(deltas)
        safety_notes = self._build_safety_notes(deltas)
        monitoring_plan = self._build_monitoring_plan(request=request, candidate=candidate)
        summary = self._build_summary(request=request, candidate=candidate, deltas=deltas)

        return PrescriptivePlan(
            generation_mode="template",
            provider="template_v1",
            summary=summary,
            goals=goals,
            action_steps=action_steps[: self.max_steps],
            safety_notes=safety_notes,
            monitoring_plan=monitoring_plan,
        )

    def _build_summary(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
        deltas: list[tuple[str, float]],
    ) -> str:
        target = request.target.target_class
        if not deltas:
            return (
                f"Target {target} tercapai tanpa perubahan fitur besar. "
                "Fokus pada konsistensi kebiasaan sehat."
            )

        top_feature, top_delta = deltas[0]
        direction = "diturunkan" if top_delta < 0 else "dinaikkan"
        return (
            f"Rencana utama: capai target {target} dengan fokus {top_feature} "
            f"yang perlu {direction} sekitar {abs(top_delta):.2f} unit."
        )

    def _goal_line(self, feature: str, delta: float) -> str:
        if delta < 0:
            return f"Turunkan {feature} sebesar {abs(delta):.2f}."
        return f"Tingkatkan {feature} sebesar {abs(delta):.2f}."

    def _build_actions(self, deltas: list[tuple[str, float]]) -> list[str]:
        actions: list[str] = []
        for feature, delta in deltas:
            actions.extend(self._feature_actions(feature=feature, delta=delta))

        if not actions:
            actions.append("Pertahankan pola makan seimbang dan aktivitas fisik rutin.")
        return actions

    def _feature_actions(self, feature: str, delta: float) -> list[str]:
        name = feature.lower()
        if name == "bmi":
            return [
                "Terapkan defisit kalori moderat 300-500 kkal per hari dengan menu terukur.",
                "Lakukan aktivitas aerobik minimal 150 menit per minggu ditambah latihan kekuatan 2 kali per minggu.",
            ]
        if "physical_activity" in name:
            return [
                "Tingkatkan frekuensi aktivitas fisik bertahap 1-2 sesi per minggu.",
                "Gunakan target langkah harian dan evaluasi mingguan agar progres konsisten.",
            ]
        if "smoking" in name or "brinkman" in name:
            return [
                "Turunkan paparan rokok secara bertahap dengan jadwal pengurangan yang jelas.",
                "Pertimbangkan dukungan klinik berhenti merokok atau konseling perilaku.",
            ]
        if "cholesterol" in name:
            return [
                "Kurangi lemak jenuh dan trans, tingkatkan serat larut dari sayur, buah, dan kacang-kacangan.",
                "Jadwalkan cek profil lipid ulang sesuai arahan tenaga kesehatan.",
            ]
        if "hypertension" in name:
            return [
                "Batasi konsumsi garam, perbanyak makanan segar, dan pantau tekanan darah di rumah.",
                "Konsultasikan tata laksana tekanan darah dengan dokter bila nilai tetap tinggi.",
            ]

        direction = "turunkan" if delta < 0 else "naikkan"
        return [f"Lakukan intervensi bertahap untuk {direction} nilai {feature} sesuai target."]

    def _build_safety_notes(self, deltas: list[tuple[str, float]]) -> list[str]:
        notes = [
            "Hindari perubahan ekstrem dalam waktu singkat; gunakan progres bertahap dan konsisten.",
            "Jika ada gejala memburuk, hentikan intervensi dan konsultasikan ke dokter.",
        ]
        if any(feature.lower() == "bmi" for feature, _ in deltas):
            notes.append("Penurunan berat badan aman umumnya sekitar 0.25-0.75 kg per minggu.")
        return notes

    def _build_monitoring_plan(
        self,
        request: CounterfactualRequest,
        candidate: CounterfactualCandidate,
    ) -> list[str]:
        current = candidate.prediction.probability_low_risk
        target = request.target.min_target_probability
        return [
            "Evaluasi indikator utama setiap 1-2 minggu (berat badan, aktivitas, tekanan darah, kebiasaan merokok).",
            f"Review prediksi ulang setelah perubahan utama; target probabilitas low_risk >= {target:.2f}.",
            f"Probabilitas low_risk kandidat saat ini: {current:.2f}.",
        ]
