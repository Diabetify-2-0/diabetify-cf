# Experiments

Folder `experiments/` adalah pipeline benchmark untuk membandingkan engine counterfactual di `diabetify-cf`.

Fungsi utamanya:
- menjalankan benchmark per-engine atau multi-engine,
- menggabungkan hasil run,
- mengaudit hasil secara otomatis,
- menyiapkan artefak analisis untuk notebook.

Benchmark inti final yang dipakai di repo ini:
- `dice`
- `ocean`
- `ft`
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

Gunakan ini untuk run resmi yang membandingkan `dice`, `ocean`, `ft`, dan `nn` dalam desain yang sama.

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

### 4. Diagnosa hasil comparison

Untuk membaca gap satu engine terhadap baseline:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\diagnose_engine.py --target-engine ocean --baseline-engine dice
```

Output:
- `<target_engine>_diagnostics.md`
- `<target_engine>_diagnostics.json`

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
  - `tight_bounds`
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
  - `bounds_violation_rate`
  - `directional_violation_rate`
- **Metrik diagnostik pendukung**
  - `reason_counts`
  - `status_counts`
  - `mean_lof_score`
  - `mean_distance_l1`
  - `mean_changed_feature_count`

## Hasil Benchmark Saat Ini

Ringkasan berikut mengacu pada run perbandingan final dengan konfigurasi `20/10/5` di:

```text
experiments/results/comparisons/20260509_050703/
```

Matriks utama:

| Engine | Feasible Rate Operasional | Plausibility Pass Rate | Mean LOF Score | Fully Feasible Case Rate | Feasible-Only Jaccard | Feasible-Only Std Norm | Mean Runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiCE | 85.00% | 100.00% | 1.185 | 100.00% | 72.08% | 0.162 | 539.55 |
| OCEAN | 40.00% | 100.00% | 1.011 | 60.00% | 100.00% | 0.000 | 3122.25 |
| FT | 37.50% | 75.00% | 0.757 | 40.00% | 100.00% | 0.000 | 195.34 |
| NN | 81.25% | 100.00% | 1.019 | 100.00% | 100.00% | 0.000 | 369.53 |

Validasi constraint pada kelompok operasional tetap bersih:

| Engine | Immutable Viol. | Mutable Viol. | Bounds Viol. | Directional Viol. |
|---|---:|---:|---:|---:|
| DiCE | 0.00% | 0.00% | 0.00% | 0.00% |
| OCEAN | 0.00% | 0.00% | 0.00% | 0.00% |
| FT | 0.00% | 0.00% | 0.00% | 0.00% |
| NN | 0.00% | 0.00% | 0.00% | 0.00% |

Pembacaan praktis hasil:

- `DiCE` memberikan coverage operasional tertinggi.
- `NN` tetap sangat dekat pada coverage operasional, tetapi lebih kuat pada stabilitas.
- `NN` juga menjaga plausibility tetap sangat baik dan runtime lebih efisien daripada `DiCE`.
- `OCEAN` tetap plausible, tetapi coverage rendah dan runtime sangat berat.
- `FT` cepat, tetapi terlalu lemah pada coverage untuk dijadikan engine utama sistem.

## Kesimpulan yang Diambil

Keputusan kerja saat ini dari folder `experiments/` adalah:

- `DiCE` dipertahankan sebagai **engine referensi terbaik untuk coverage operasional**.
- `NN` dipilih sebagai **engine utama yang paling layak untuk integrasi ke sistem**.

Alasannya bukan karena `NN` menang di semua metrik, tetapi karena `NN` memberi **trade-off keseluruhan terbaik**. Dalam benchmark ini, sistem tidak hanya membutuhkan coverage solusi, tetapi juga membutuhkan rekomendasi yang:

- stabil saat request diulang,
- tetap plausible terhadap distribusi data referensi,
- patuh terhadap constraint,
- dan cukup efisien secara operasional.

Dengan pembacaan tersebut, `NN` menjadi pilihan paling andal untuk integrasi sistem, sedangkan `DiCE` tetap penting sebagai pembanding utama pada aspek feasibility coverage.

## Keterangan Tambahan Folder `results/`

- `latest/` berisi pointer `.txt` ke run terbaru,
- `combined/` berisi CSV gabungan,
- folder timestamp menyimpan artefak run aktual.

## Notebook

Notebook di `experiments/notebooks/` dipakai untuk:
- membaca report hasil benchmark,
- merangkum scenario, stability, dan candidate metrics,
- menyiapkan analisis untuk penulisan TA.

Notebook bukan source of truth eksekusi eksperimen. Eksekusi utama tetap lewat script di `experiments/scripts/`.

## Catatan

- `run_core_checkpoint.py` adalah jalur utama benchmark final.
- `run_benchmark.py`, `run_baseline.py`, dan `run_comparison.py` lebih cocok untuk debug atau eksperimen kecil.
- `collect_results.py` sebaiknya dipakai hanya pada root hasil yang memang satu scope, misalnya satu `comparison_root` atau satu `baseline_root`.
