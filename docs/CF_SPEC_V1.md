# CF Spec V1 - Diabetify Counterfactual Prescriptive Engine

## 1) Scope
Dokumen ini mendefinisikan spesifikasi formal untuk modul Counterfactual (CF) berbasis DiCE yang:
- Menghasilkan counterfactual valid untuk menurunkan risiko diabetes.
- Patuh 100% pada aturan immutable dan mutable pilihan pengguna.
- Mendeteksi kondisi infeasible secara eksplisit (bukan memberi rekomendasi palsu).
- Menyediakan output siap dipakai layer planner preskriptif (LLM) tanpa mengubah validitas CF inti.

Versi: `cf_spec_version=1.0.0`

## 2) Terminology
- `x`: vektor fitur input pasien saat ini.
- `x_cf`: kandidat counterfactual.
- `f(x)`: model prediksi risiko diabetes (XGBoost) yang mengembalikan probabilitas dan kelas.
- `target_class`: kelas risiko tujuan (default: `low_risk`).
- `immutable_features`: fitur yang tidak boleh berubah.
- `mutable_allowed`: subset fitur yang diizinkan user untuk diubah.
- `medical_constraints`: aturan kausalitas/medis yang wajib dipatuhi.
- `feasible`: ada minimal satu solusi yang memenuhi seluruh hard constraints.

## 3) Objective Formal
Cari himpunan solusi `CF = {x_cf_1..x_cf_k}` sehingga:
1. `class_flip`: `f_class(x_cf_i) == target_class`.
2. `immutable_safe`: `x_cf_i[j] == x[j]` untuk semua `j` di `immutable_features`.
3. `mutable_compliant`: perubahan hanya terjadi di `mutable_allowed`.
4. `medical_valid`: seluruh `medical_constraints` terpenuhi.
5. `plausible`: berada pada distribusi data wajar.
6. `optimized`: minim perubahan, realistis, stabil, dan cepat dihitung.

Jika tidak ada solusi dalam ruang pencarian yang dibatasi user, status harus `INFEASIBLE`.

## 4) Required Inputs
### 4.1 Model Artifacts
- Model XGBoost final (dengan interface `predict_proba`).
- Pipeline preprocessing identik training (encoder, scaler, column order).
- `model_version` wajib disertakan di response.

### 4.2 Reference Data
- Dataset referensi (train/background) untuk:
  - konstrain domain,
  - plausibility (LOF),
  - kalibrasi bounds.

### 4.3 Feature Registry
Setiap fitur wajib punya metadata:
- `name`
- `type`: `continuous` | `categorical` | `ordinal` | `binary`
- `immutable`: `true|false`
- `global_min`, `global_max` (untuk continuous/ordinal)
- `allowed_values` (untuk categorical/binary)
- `actionable`: `true|false`
- `default_mutable`: `true|false`
- `cost_weight`: bobot kesulitan perubahan (untuk objective)

## 5) Hard Constraints
Hard constraints selalu wajib lolos:
1. Immutable constraint:
   - fitur immutable tidak boleh berubah.
2. User mutable constraint:
   - fitur di luar `mutable_allowed` tidak boleh berubah.
3. User range constraint:
   - jika user memberi batas fitur, `x_cf` wajib di dalam batas tersebut.
4. Medical causal constraints:
   - aturan domain medis (contoh format rule):
     - `bmi = weight / (height_m^2)` harus konsisten jika fitur turunan dipakai.
     - nilai klinis tidak boleh di luar domain biologis yang ditetapkan.
5. Valid data type constraints:
   - tipe data, kategori, dan skala harus valid.

## 6) Soft Objectives (Optimization)
Untuk kandidat feasible, optimasi skor:

`score = w_prox * proximity + w_sparse * sparsity + w_plaus * plausibility + w_cost * action_cost + w_stab * stability_proxy`

- `proximity`: jarak terhadap `x` (misal weighted L1/L2).
- `sparsity`: jumlah fitur yang berubah.
- `plausibility`: kedekatan terhadap distribusi normal (LOF).
- `action_cost`: biaya aksi berdasarkan `cost_weight`.
- `stability_proxy`: penalti untuk solusi yang sensitif terhadap noise kecil.

Bobot default dikonfigurasikan per eksperimen dan dicatat di metadata response.

## 7) API / Message Contract
Arsitektur async via RabbitMQ.
- Request queue: `ml.cf.request`
- Response queue: `ml.cf.response`

### 7.1 Request Schema (JSON)
```json
{
  "request_id": "uuid",
  "timestamp": "2026-03-02T10:00:00Z",
  "patient_id": "string-optional",
  "model_version": "xgb_v3",
  "target": {
    "target_class": "low_risk",
    "min_target_probability": 0.5
  },
  "instance": {
    "features": {
      "age": 45,
      "bmi": 31.2,
      "glucose": 165
    }
  },
  "constraints": {
    "immutable_features": ["age", "sex", "family_history"],
    "mutable_allowed": ["bmi", "glucose", "physical_activity", "diet_score"],
    "feature_bounds": {
      "bmi": { "min": 20.0, "max": 29.0 },
      "physical_activity": { "min": 2, "max": 7 }
    },
    "must_not_change": [],
    "medical_rule_set_version": "med_rule_v1"
  },
  "generation": {
    "total_cfs": 3,
    "method": "dice_genetic",
    "random_seed": 42,
    "timeout_ms": 5000
  },
  "preferences": {
    "cost_weights": {
      "bmi": 1.0,
      "glucose": 1.5,
      "physical_activity": 0.8
    },
    "objective_weights": {
      "proximity": 0.30,
      "sparsity": 0.20,
      "plausibility": 0.20,
      "action_cost": 0.20,
      "stability_proxy": 0.10
    }
  }
}
```

### 7.2 Response Schema (JSON)
```json
{
  "request_id": "uuid",
  "status": "FEASIBLE",
  "reason_code": "OK",
  "message": "3 feasible counterfactuals generated",
  "model_version": "xgb_v3",
  "cf_engine_version": "dice_engine_v1",
  "constraint_version": "cf_spec_1.0.0",
  "runtime_ms": 1840,
  "input_prediction": {
    "class": "high_risk",
    "probability_low_risk": 0.18
  },
  "candidates": [
    {
      "candidate_id": "cf_1",
      "features": {
        "age": 45,
        "bmi": 27.4,
        "glucose": 118
      },
      "delta": {
        "bmi": -3.8,
        "glucose": -47
      },
      "prediction": {
        "class": "low_risk",
        "probability_low_risk": 0.74
      },
      "metrics": {
        "distance_l1": 0.21,
        "changed_feature_count": 2,
        "lof_score": 1.03,
        "constraint_violations": 0
      }
    }
  ],
  "validation": {
    "immutable_violation": false,
    "mutable_compliance": true,
    "medical_rules_passed": true
  },
  "planner_input": {
    "recommended_candidate_id": "cf_1",
    "target_deltas": {
      "bmi": -3.8,
      "glucose": -47
    }
  }
}
```

### 7.3 Status Values
- `FEASIBLE`: minimal 1 kandidat valid ditemukan.
- `INFEASIBLE`: tidak ada solusi valid dalam constraint user/medis.
- `ERROR`: kegagalan teknis (timeout, model load, dsb).

### 7.4 Infeasible Reason Codes
- `NO_MUTABLE_FEATURE`
- `TARGET_UNREACHABLE_UNDER_CONSTRAINTS`
- `CONFLICTING_BOUNDS`
- `MEDICAL_RULE_VIOLATION_ONLY`
- `TIMEOUT_NO_FEASIBLE_SOLUTION`
- `INVALID_INPUT_SCHEMA`

## 8) Success Metrics (Formal)
Seluruh metrik dihitung per batch evaluasi.

1. Feasible Class-Flip Success
   - `success_feasible = (# kasus feasible yang berhasil flip) / (# kasus feasible)`
   - Target: `100%`.

2. Infeasible Detection Correctness
   - `infeasible_precision = (# prediksi infeasible yang benar) / (# prediksi infeasible)`
   - Sistem wajib return `INFEASIBLE` jika memang tidak ada solusi.

3. Immutable Violation Rate
   - `immutable_violation_rate = (# kandidat melanggar immutable) / (# seluruh kandidat)`
   - Target: `0%`.

4. Mutable Compliance Rate
   - `mutable_compliance_rate = (# kandidat patuh mutable_allowed) / (# seluruh kandidat)`
   - Target: `100%`.

5. Stability
   - Untuk input sama, jalankan N kali (default N=10) dengan seed terkontrol.
   - Ukur std dev delta fitur per kandidat terpilih.
   - `stability_std_norm = mean(std(delta_feature_normalized))`
   - Target default V1: `stability_std_norm <= 0.01` (dapat dituning sesuai hasil awal).

6. Plausibility (LOF)
   - Hitung `LOF(x_cf)` terhadap reference data.
   - `lof_deviation = mean(abs(LOF - 1.0))`
   - Target default V1: `lof_deviation <= 0.10`.

7. Latency
   - `runtime_ms` per request.
   - Target: `P95 runtime < 5000 ms`.

## 9) Evaluation Protocol
1. Gunakan dataset evaluasi tetap (holdout) + seed tetap.
2. Pisahkan laporan:
   - subset feasible,
   - subset infeasible.
3. Simpan artifact eksperimen:
   - parameter DiCE,
   - objective weights,
   - model/constraint versions,
   - log kandidat dan alasan reject.
4. Laporkan tabel utama:
   - success_feasible,
   - immutable_violation_rate,
   - mutable_compliance_rate,
   - stability_std_norm,
   - lof_deviation,
   - P95 runtime.

## 10) Integration with Prescriptive LLM Planner
Prinsip:
1. CF engine menghasilkan `target_deltas` yang sudah valid.
2. LLM planner hanya menerjemahkan delta menjadi action plan bertahap.
3. LLM tidak boleh mengubah target numerik CF tanpa validasi ulang engine.

Kontrak minimum ke LLM:
- input: baseline patient profile, `target_deltas`, constraints user, preferensi hidup.
- output: rencana aksi harian/mingguan + estimasi dampak terhadap fitur mutable.

## 11) Non-Goals for V1
- Belum mencakup multi-objective global optimizer custom di luar DiCE.
- Belum mencakup fairness metrics lintas subgroup.
- Belum mencakup adaptive online learning.

## 12) Implementation Guidance
Untuk project ini, modul CF direkomendasikan sebagai service terpisah:
- Nama folder service: `diabetify-cf/`
- Alasan:
  - dependency DICE/eksperimen tidak mengganggu `diabetify-ml` existing worker,
  - scaling dan lifecycle deployment bisa dipisah,
  - memudahkan eksperimen TA (versioning engine, objective, dan evaluasi).