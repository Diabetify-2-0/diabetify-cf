from __future__ import annotations

from dataclasses import dataclass

from diabetify_cf.schemas import (
    CounterfactualCandidate,
    CounterfactualRequest,
    PlannerFeatureChange,
    PlannerInput,
)


@dataclass(frozen=True)
class PlanningPolicyResult:
    clinical_scope: str
    policy_version: str
    summary: str
    goals: list[str]
    action_steps: list[str]
    safety_notes: list[str]
    monitoring_plan: list[str]
    missing_context: list[str]
    contraindication_flags: list[str]
    human_review_required: bool


def build_policy_result(
    request: CounterfactualRequest,
    candidate: CounterfactualCandidate,
    planner_input: PlannerInput | None = None,
) -> PlanningPolicyResult:
    target = request.target.target_class
    changed_features = _ordered_feature_changes(candidate=candidate, planner_input=planner_input)
    goal_lines = [_goal_line(change) for change in changed_features]
    action_steps = _base_steps()
    safety_notes = _base_safety_notes()
    missing_context = _base_missing_context()
    contraindication_flags: list[str] = []

    for change in changed_features:
        action_steps.extend(
            _feature_actions(
                feature=change.feature_name,
                delta=float(change.delta),
                baseline_value=change.baseline_value,
                candidate_value=change.candidate_value,
            )
        )
        safety_notes.extend(_feature_safety_notes(change.feature_name))
        missing_context.extend(_feature_missing_context(change.feature_name))
        contraindication_flags.extend(_feature_contraindications(change.feature_name))

    clinical_scope = "decision_support"
    summary = (
        f"Kandidat {candidate.candidate_id} dapat dipakai sebagai dasar diskusi terarah "
        f"untuk menuju target {target}. Output planner ini bersifat decision-support "
        "dan tetap memerlukan penilaian profesional sebelum diturunkan menjadi aksi spesifik."
    )
    human_review_required = True

    monitoring_plan = [
        "Konfirmasi kembali baseline klinis dan konteks pasien sebelum menurunkan target numerik ke rencana tindakan.",
        f"Target probabilitas low_risk pada request: >= {request.target.min_target_probability:.2f}.",
        f"Probabilitas low_risk kandidat counterfactual: {candidate.prediction.probability_low_risk:.2f}.",
    ]
    if planner_input is not None and planner_input.input_prediction is not None:
        monitoring_plan.insert(
            1,
            "Probabilitas low_risk input awal: "
            f"{planner_input.input_prediction.probability_low_risk:.2f}.",
        )

    if not goal_lines:
        goal_lines.append(
            "Tidak ada delta besar; pertahankan strategi kontrol risiko yang sudah berjalan."
        )

    return PlanningPolicyResult(
        clinical_scope=clinical_scope,
        policy_version="planner_policy_v1",
        summary=summary,
        goals=_dedupe(goal_lines),
        action_steps=_dedupe(action_steps),
        safety_notes=_dedupe(safety_notes),
        monitoring_plan=_dedupe(monitoring_plan),
        missing_context=_dedupe(missing_context),
        contraindication_flags=_dedupe(contraindication_flags),
        human_review_required=human_review_required,
    )


def _goal_line(change: PlannerFeatureChange) -> str:
    action = "diturunkan" if float(change.delta) < 0 else "ditingkatkan"
    baseline = _format_value(change.baseline_value)
    candidate = _format_value(change.candidate_value)
    magnitude = abs(float(change.delta))
    return (
        f"Pertimbangkan perubahan pada {change.feature_name} yang menurut kandidat perlu "
        f"{action} dari {baseline} ke {candidate} sekitar {magnitude:.2f} unit."
    )


def _base_steps() -> list[str]:
    return [
        "Verifikasi bahwa target perubahan numerik relevan dengan kondisi klinis, terapi berjalan, dan preferensi pasien.",
        "Saring lebih dahulu kontraindikasi dan konteks yang belum tersedia sebelum menurunkan target ke rekomendasi spesifik.",
    ]


def _base_safety_notes() -> list[str]:
    return [
        "Hindari menerjemahkan delta model menjadi instruksi terapi otomatis tanpa menilai konteks klinis lengkap.",
        "Bila konteks klinis tidak cukup, sistem sebaiknya berhenti pada level edukasi atau decision-support.",
    ]


def _base_missing_context() -> list[str]:
    return [
        "Riwayat obat dan terapi yang sedang berjalan.",
        "Komorbiditas, kondisi akut, atau pembatasan medis yang relevan.",
        "Preferensi pasien, kemampuan menjalankan perubahan, dan hambatan sosial/lingkungan.",
    ]


def _feature_actions(
    feature: str,
    delta: float,
    baseline_value: object,
    candidate_value: object,
) -> list[str]:
    name = feature.lower()
    value_context = (
        f"Pertahankan fokus pada transisi dari {_format_value(baseline_value)} "
        f"menuju {_format_value(candidate_value)} secara bertahap dan realistis."
    )
    if "bmi" in name:
        return [
            "Pertimbangkan strategi pengelolaan berat badan bertahap yang sesuai dengan status gizi, pola makan, dan kapasitas aktivitas pasien. "
            + value_context
        ]
    if "physical_activity" in name:
        return [
            "Nilai kapasitas fungsional dan gejala sebelum menganjurkan peningkatan aktivitas; bila aman, gunakan peningkatan bertahap. "
            + value_context
        ]
    if "smoking" in name or "brinkman" in name:
        return [
            "Pertimbangkan dukungan berhenti merokok berbasis konseling, follow-up, dan terapi yang sesuai kebutuhan pasien. "
            + value_context
        ]
    if "cholesterol" in name:
        return [
            "Review pola makan, hasil lipid, dan terapi yang sedang berjalan sebelum menetapkan target spesifik. "
            + value_context
        ]
    if "hypertension" in name:
        return [
            "Evaluasi kontrol tekanan darah, obat yang sedang dipakai, dan keamanan target aktivitas sebelum membuat rencana rinci. "
            + value_context
        ]
    direction = "penurunan" if delta < 0 else "peningkatan"
    return [
        f"Tafsirkan target {direction} pada {feature} sebagai area fokus, bukan instruksi otomatis. {value_context}"
    ]


def _feature_safety_notes(feature: str) -> list[str]:
    name = feature.lower()
    if "bmi" in name:
        return [
            "Target terkait berat badan perlu hati-hati pada kehamilan, gangguan makan, atau kondisi yang memengaruhi status gizi.",
        ]
    if "physical_activity" in name:
        return [
            "Peningkatan aktivitas fisik memerlukan kehati-hatian bila ada nyeri dada, sesak, pusing saat aktivitas, atau keterbatasan mobilitas.",
        ]
    if "hypertension" in name:
        return [
            "Perubahan aktivitas dan gaya hidup pada pasien hipertensi perlu mempertimbangkan kontrol tekanan darah dan terapi yang sedang berjalan.",
        ]
    return []


def _feature_missing_context(feature: str) -> list[str]:
    name = feature.lower()
    if "bmi" in name:
        return [
            "Status gizi, pola makan, dan faktor yang memengaruhi berat badan belum diketahui penuh."
        ]
    if "physical_activity" in name:
        return [
            "Kapasitas fungsional, gejala saat aktivitas, dan riwayat pembatasan fisik belum tersedia."
        ]
    if "hypertension" in name:
        return ["Data tekanan darah serial dan terapi antihipertensi belum tersedia."]
    if "cholesterol" in name:
        return ["Profil lipid terkini dan terapi penurun lipid belum tersedia."]
    return []


def _feature_contraindications(feature: str) -> list[str]:
    name = feature.lower()
    if "physical_activity" in name:
        return [
            "Tunda target aktivitas spesifik bila ada gejala kardiorespirasi aktif atau keterbatasan fisik yang belum dievaluasi."
        ]
    if "bmi" in name:
        return [
            "Hindari target penurunan berat badan agresif tanpa evaluasi bila ada kondisi yang memengaruhi status gizi."
        ]
    return []
def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        stripped = item.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        out.append(stripped)
    return out


def _ordered_feature_changes(
    *,
    candidate: CounterfactualCandidate,
    planner_input: PlannerInput | None,
) -> list[PlannerFeatureChange]:
    if planner_input is not None and planner_input.changed_features:
        return list(planner_input.changed_features)

    changes = [
        PlannerFeatureChange(
            feature_name=feature_name,
            baseline_value=0.0,
            candidate_value=0.0,
            delta=delta,
            direction="decrease" if float(delta) < 0 else "increase",
        )
        for feature_name, delta in sorted(
            candidate.delta.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    ]
    return changes


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
