# Experiments

Folder `experiments/` adalah pipeline benchmark untuk membandingkan engine counterfactual di `diabetify-cf`.

Fungsi utamanya:
- menjalankan benchmark per-engine atau multi-engine,
- menggabungkan hasil run,
- mengaudit hasil secara otomatis,
- menyiapkan artefak analisis untuk notebook.

Benchmark inti final yang dipakai di repo ini:
- `dice`
- `nn`

## Struktur Singkat

```text
experiments/
  configs/
    benchmark_scope/   # scope benchmark inti
    engines/           # config per engine
    scenarios/         # config scenario / constraint
  engines/             # adapter engine eksperimen
  evaluation/          # metrik evaluasi kandidat
  notebooks/           # analisis hasil
  scripts/             # runner benchmark, comparison, audit, diagnosis
  results/             # output run lokal (di-ignore Git)
```

## Script yang Paling Penting

### 1. Jalankan benchmark inti final

Gunakan ini untuk run resmi yang membandingkan `dice` dan `nn` dalam desain yang sama.

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_core_checkpoint.py `
  --scenario-limit 20 `
  --stability-limit 10 `
  --repeat-count 5 `
  --scenario-timeout-seconds 180 `
  --stability-timeout-seconds 180
```

Output utama:
- `comparison_report.md`
- `checkpoint_report.md`
- `comparison_manifest.json`
- `checkpoint_manifest.json`
- `audit_report.json`

### 2. Cek kesiapan environment

```powershell
.\.venv\Scripts\python.exe experiments\scripts\check_engine_availability.py
```

Ini mengecek:
- dependency engine,
- model artifact,
- feature columns,
- reference data,
- feature registry.

### 3. Jalankan satu benchmark kecil

Untuk debug satu engine dan satu scenario:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_benchmark.py --engine-config experiments\configs\engines\dice.json --scenario-config experiments\configs\scenarios\all_mutable.json
```

## Folder `results/`

Semua hasil eksperimen ditulis ke:

```text
experiments/results/
```

Layout utamanya:

```text
experiments/results/
  benchmarks/<engine>/<timestamp>/
  baselines/<engine>/<timestamp>/
  comparisons/<timestamp>/
  scenarios/<engine>/<timestamp>/
  stability/<engine>/<timestamp>/
  combined/
  latest/
```

## Posisi Benchmark Saat Ini

Desain benchmark pada folder ini saat ini diperlakukan sebagai **beku secara metodologis**. Artinya, pipeline, pengelompokan skenario, metrik utama, dan konfigurasi run inti sudah dianggap cukup matang untuk dijadikan dasar evaluasi dan penarikan kesimpulan.

Konfigurasi benchmark inti final:
- `scenario-limit 20`
- `stability-limit 10`
- `repeat-count 5`
- `scenario-timeout-seconds 180`
- `stability-timeout-seconds 180`

Kelompok skenario yang dipakai:

- **Operasional**
  - `all_mutable`
  - `lifestyle_combo`
  - `minimal_realistic_combo`
  - `bmi_only`
- **Stress test**
  - `activity_only`
- **Kontrol infeasible**
  - `no_mutable`
- **Stabilitas**
  - `stability`

Metrik dibaca dalam tiga lapisan:

- **Metrik utama pembanding**
  - `feasible_rate` pada skenario operasional
  - `plausibility_pass_rate`
  - `fully_feasible_case_rate`
  - `mean_feasible_only_jaccard_changed_features`
  - `mean_feasible_only_stability_std_norm`
  - `mean_runtime_ms`
- **Metrik validasi constraint**
  - `immutable_violation_rate`
  - `mutable_violation_rate`
  - `directional_violation_rate`
- **Metrik diagnostik pendukung**
  - `reason_counts`
  - `status_counts`
  - `mean_lof_score`
  - `mean_distance_l1`
  - `mean_changed_feature_count`
