from __future__ import annotations

from dataclasses import dataclass

from diabetify_cf.schemas import CounterfactualCandidate, CounterfactualRequest


@dataclass(frozen=True)
class PlanningPolicyResult:
    intended_user: str
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


def normalize_intended_user(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"clinician", "patient"}:
        return normalized
    return "clinician"


def build_policy_result(
    request: CounterfactualRequest,
    candidate: CounterfactualCandidate,
    intended_user: str = "clinician",
) -> PlanningPolicyResult:
    target = request.target.target_class
    normalized_user = normalize_intended_user(intended_user)
    deltas = sorted(
        candidate.delta.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    goal_lines = [_goal_line(feature, delta, normalized_user) for feature, delta in deltas]
    action_steps = _base_steps(normalized_user)
    safety_notes = _base_safety_notes(normalized_user)
    missing_context = _base_missing_context()
    contraindication_flags: list[str] = []

    for feature, delta in deltas:
        action_steps.extend(_feature_actions(feature, delta, normalized_user))
        safety_notes.extend(_feature_safety_notes(feature))
        missing_context.extend(_feature_missing_context(feature))
        contraindication_flags.extend(_feature_contraindications(feature))

    if normalized_user == "patient":
        clinical_scope = "patient_education"
        summary = (
            f"Model menunjukkan beberapa area yang dapat didiskusikan untuk membantu "
            f"menuju target {target}. Gunakan ringkasan ini sebagai panduan edukatif, "
            "bukan instruksi medis final."
        )
        human_review_required = False
    else:
        clinical_scope = "clinician_support"
        summary = (
            f"Kandidat {candidate.candidate_id} dapat dipakai sebagai dasar diskusi klinis "
            f"untuk menuju target {target}. Output planner ini bersifat decision-support "
            "dan tetap memerlukan penilaian profesional sebelum diturunkan menjadi aksi spesifik."
        )
        human_review_required = True

    monitoring_plan = [
        "Konfirmasi kembali baseline klinis dan konteks pasien sebelum menurunkan target numerik ke rencana tindakan.",
        f"Target probabilitas low_risk pada request: >= {request.target.min_target_probability:.2f}.",
        f"Probabilitas low_risk kandidat counterfactual: {candidate.prediction.probability_low_risk:.2f}.",
    ]
    if normalized_user == "patient":
        monitoring_plan[0] = (
            "Pantau perubahan secara bertahap dan konsultasikan dengan tenaga kesehatan "
            "bila ada gejala baru, hambatan besar, atau kondisi khusus."
        )

    if not goal_lines:
        if normalized_user == "patient":
            goal_lines.append("Pertahankan kebiasaan sehat yang sudah berjalan dan lakukan pemantauan rutin.")
        else:
            goal_lines.append("Tidak ada delta besar; pertahankan strategi kontrol risiko yang sudah berjalan.")

    return PlanningPolicyResult(
        intended_user=normalized_user,
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


def _goal_line(feature: str, delta: float, intended_user: str) -> str:
    action = "diturunkan" if delta < 0 else "ditingkatkan"
    if intended_user == "patient":
        return (
            f"Fokuskan perhatian pada {feature} yang menurut model perlu {action} "
            f"sekitar {abs(delta):.2f} unit."
        )
    return (
        f"Pertimbangkan perubahan pada {feature} yang menurut kandidat perlu {action} "
        f"sekitar {abs(delta):.2f} unit."
    )


def _base_steps(intended_user: str) -> list[str]:
    if intended_user == "patient":
        return [
            "Gunakan hasil ini sebagai bahan diskusi dengan tenaga kesehatan, terutama bila Anda memiliki kondisi medis lain atau sedang menjalani terapi.",
            "Fokus pada perubahan yang bertahap, realistis, dan dapat dipantau, bukan perubahan ekstrem dalam waktu singkat.",
        ]
    return [
        "Verifikasi bahwa target perubahan numerik relevan dengan kondisi klinis, terapi berjalan, dan preferensi pasien.",
        "Saring lebih dahulu kontraindikasi dan konteks yang belum tersedia sebelum menurunkan target ke rekomendasi spesifik.",
    ]


def _base_safety_notes(intended_user: str) -> list[str]:
    notes = [
        "Hindari menerjemahkan delta model menjadi instruksi terapi otomatis tanpa menilai konteks klinis lengkap.",
        "Bila konteks klinis tidak cukup, sistem sebaiknya berhenti pada level edukasi atau decision-support.",
    ]
    if intended_user == "patient":
        notes[0] = (
            "Jangan menganggap hasil ini sebagai pengganti konsultasi dokter; gunakan hanya sebagai panduan edukatif."
        )
    return notes


def _base_missing_context() -> list[str]:
    return [
        "Riwayat obat dan terapi yang sedang berjalan.",
        "Komorbiditas, kondisi akut, atau pembatasan medis yang relevan.",
        "Preferensi pasien, kemampuan menjalankan perubahan, dan hambatan sosial/lingkungan.",
    ]


def _feature_actions(feature: str, delta: float, intended_user: str) -> list[str]:
    name = feature.lower()
    if "bmi" in name:
        return [
            _line(
                intended_user,
                clinician="Pertimbangkan strategi pengelolaan berat badan bertahap yang sesuai dengan status gizi, pola makan, dan kapasitas aktivitas pasien.",
                patient="Diskusikan cara pengelolaan berat badan yang bertahap dan aman sesuai kondisi Anda.",
            )
        ]
    if "physical_activity" in name:
        return [
            _line(
                intended_user,
                clinician="Nilai kapasitas fungsional dan gejala sebelum menganjurkan peningkatan aktivitas; bila aman, gunakan peningkatan bertahap.",
                patient="Jika aman menurut tenaga kesehatan, tingkatkan aktivitas fisik secara bertahap dan hentikan bila muncul gejala yang mengkhawatirkan.",
            )
        ]
    if "smoking" in name or "brinkman" in name:
        return [
            _line(
                intended_user,
                clinician="Pertimbangkan dukungan berhenti merokok berbasis konseling, follow-up, dan terapi yang sesuai kebutuhan pasien.",
                patient="Bila Anda merokok, pertimbangkan dukungan berhenti merokok secara bertahap bersama tenaga kesehatan.",
            )
        ]
    if "cholesterol" in name:
        return [
            _line(
                intended_user,
                clinician="Review pola makan, hasil lipid, dan terapi yang sedang berjalan sebelum menetapkan target spesifik.",
                patient="Gunakan hasil ini untuk membahas pola makan, pemantauan lipid, dan terapi yang sedang berjalan dengan tenaga kesehatan.",
            )
        ]
    if "hypertension" in name:
        return [
            _line(
                intended_user,
                clinician="Evaluasi kontrol tekanan darah, obat yang sedang dipakai, dan keamanan target aktivitas sebelum membuat rencana rinci.",
                patient="Bila Anda memiliki tekanan darah tinggi, pastikan perubahan gaya hidup dibahas bersama tenaga kesehatan terlebih dahulu.",
            )
        ]
    direction = "penurunan" if delta < 0 else "peningkatan"
    return [
        _line(
            intended_user,
            clinician=f"Tafsirkan target {direction} pada {feature} sebagai area fokus, bukan instruksi otomatis.",
            patient=f"Gunakan target {direction} pada {feature} sebagai area yang perlu dibicarakan lebih lanjut, bukan tugas medis final.",
        )
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
        return ["Status gizi, pola makan, dan faktor yang memengaruhi berat badan belum diketahui penuh."]
    if "physical_activity" in name:
        return ["Kapasitas fungsional, gejala saat aktivitas, dan riwayat pembatasan fisik belum tersedia."]
    if "hypertension" in name:
        return ["Data tekanan darah serial dan terapi antihipertensi belum tersedia."]
    if "cholesterol" in name:
        return ["Profil lipid terkini dan terapi penurun lipid belum tersedia."]
    return []


def _feature_contraindications(feature: str) -> list[str]:
    name = feature.lower()
    if "physical_activity" in name:
        return ["Tunda target aktivitas spesifik bila ada gejala kardiorespirasi aktif atau keterbatasan fisik yang belum dievaluasi."]
    if "bmi" in name:
        return ["Hindari target penurunan berat badan agresif tanpa evaluasi bila ada kondisi yang memengaruhi status gizi."]
    return []


def _line(intended_user: str, clinician: str, patient: str) -> str:
    if intended_user == "patient":
        return patient
    return clinician


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
