# Experiments

Panduan ini diasumsikan dijalankan dari root repository:

```powershell
cd diabetify-cf
```

Gunakan Python dari virtual environment project:

```powershell
.\.venv\Scripts\python.exe
```

Untuk OCEAN, install extra eksperimen:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,experiments]"
```

## 1. Cek Kesiapan Environment

```powershell
.\.venv\Scripts\python.exe experiments\scripts\check_engine_availability.py
```

Command ini mengecek:

- package counterfactual engine yang tersedia,
- kesiapan artifact model,
- kesiapan `x_columns.pkl`,
- kesiapan reference data,
- kesiapan feature registry.

Untuk saat ini, engine yang aktif tanpa extra adalah DiCE. CARLA, OCEAN, dan FOCUS akan muncul sebagai missing sampai dependency-nya dipasang.
Dengan extra `experiments`, OCEAN akan muncul sebagai available.

Catatan: CARLA belum diaktifkan karena `carla-recourse==0.0.5` mengunci `numpy==1.19.4`, sedangkan project ini memakai `numpy==2.2.6`. Memaksa downgrade NumPy akan berisiko merusak stack `pandas`, `scikit-learn`, dan `xgboost`.

Catatan OCEAN: adapter OCEAN memakai constraint scenario saat membangun search space. Fitur yang tidak mutable dikunci pada nilai baseline, sedangkan fitur mutable dibatasi oleh permitted range dari feature registry dan scenario config. Postprocessor tetap menjadi evaluator akhir untuk target, bounds, directional rules, dan plausibility.

OCEAN juga mendukung opsi tuning khusus eksperimen lewat `engine_options` di `experiments/configs/engines/ocean.json`:

- `norm`: norm objektif yang dikirim ke `ConstraintProgrammingExplainer`.
- `attempt_count`: jumlah attempt deterministic dengan seed berbeda untuk satu request.
- `seed_step`: jarak seed antar attempt.
- `max_time_per_attempt_seconds`: batas waktu per attempt; jika `null`, budget `timeout_ms` dibagi ke jumlah attempt.
- `num_workers`: jumlah worker OCEAN; `null` memakai default library.

## 2. Jalankan Satu Benchmark DiCE

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_benchmark.py --engine-config experiments\configs\engines\dice.json --scenario-config experiments\configs\scenarios\all_mutable.json
```

Output akan dibuat di:

```text
experiments/results/benchmarks/dice/<timestamp>/
```

Isi output utama:

- `cases.jsonl`: status per request/case.
- `candidates.csv`: kandidat counterfactual dan metrik evaluasinya.
- `run_config.json`: konfigurasi, path artifact, dan metadata run.

## 3. Ringkas Satu Benchmark

Ganti `<run-folder>` dengan folder hasil benchmark.

```powershell
.\.venv\Scripts\python.exe experiments\scripts\summarize_results.py experiments\results\<run-folder>
```

Command ini membuat:

```text
<run-folder>/summary.csv
```

Summary berisi metrik seperti feasible rate, target success rate, violation rate, mean runtime, mean LOF, dan mean changed feature count.

## 4. Jalankan Semua Scenario DiCE

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_scenarios.py
```

Untuk smoke test cepat, batasi jumlah case dan pasang timeout per scenario:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_scenarios.py --configs experiments\configs\scenarios\all_mutable.json experiments\configs\scenarios\bmi_only.json --engine-config experiments\configs\engines\dice.json --limit 1 --timeout-seconds 60
```

Default scenario yang dijalankan:

- `all_mutable.json`
- `lifestyle_combo.json`
- `bmi_only.json`
- `activity_only.json`
- `tight_bounds.json`
- `no_mutable.json`

Output akan dibuat di:

```text
experiments/results/scenarios/dice/<timestamp>/
```

File penting:

- `scenario_summary.csv`
- `scenario_step_results.json`
- folder per scenario
- `run_metadata.json`

## 5. Jalankan Stability Evaluation

```powershell
.\.venv\Scripts\python.exe experiments\scripts\evaluate_stability.py --engine-config experiments\configs\engines\dice.json --scenario-config experiments\configs\scenarios\stability.json --limit 1 --repeat-count 2
```

Output akan dibuat di:

```text
experiments/results/stability/dice/<timestamp>/
```

File penting:

- `stability_runs.csv`: detail setiap repeated run.
- `stability_summary.csv`: stability per case.
- `stability_aggregate.csv`: ringkasan stability global.
- `run_config.json`: config dan metadata run.

Stability report memisahkan:

- all-repeat stability: kestabilan semua repeat, termasuk repeat yang gagal/infeasible,
- feasible-only stability: kestabilan hanya dari repeat yang menghasilkan counterfactual `FEASIBLE`,
- fully feasible case rate: proporsi case yang feasible pada semua repeat,
- stability evaluable case rate: proporsi case yang punya minimal dua repeat feasible.

Gunakan `--show-engine-output` hanya jika perlu debug output mentah dari engine. Secara default, progress bar internal engine disembunyikan agar terminal tetap bersih.

## 6. Jalankan Baseline Engine Lengkap

Untuk percobaan kecil:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_baseline.py --engine-config experiments\configs\engines\dice.json --scenario-limit 1 --stability-limit 1 --repeat-count 2 --scenario-timeout-seconds 60 --stability-timeout-seconds 60
```

Untuk baseline kecil yang lebih informatif:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_baseline.py --engine-config experiments\configs\engines\dice.json --scenario-limit 5 --stability-limit 5 --repeat-count 5
```

Baseline runner menjalankan setiap scenario dan stability evaluation sebagai subprocess terpisah. Default timeout:

- `--scenario-timeout-seconds 300`
- `--stability-timeout-seconds 300`

Jika satu scenario timeout atau gagal, baseline tetap lanjut ke step berikutnya dan mencatat statusnya di summary/manifest. Gunakan `0` untuk mematikan timeout.

Output akan dibuat di:

```text
experiments/results/baselines/<engine>/<timestamp>/
```

File penting:

- `report.md`
- `baseline_manifest.json`
- `combined/scenario_summary.csv`
- `combined/stability_summary.csv`
- `combined/candidates.csv`
- folder `scenarios/`
- folder `stability/`
- `scenarios/scenario_step_results.json`
- `stability/stability_step_result.json`

Baseline runner juga menulis pointer hasil terbaru:

```text
experiments/results/latest/baseline.txt
```

Catatan: pada beberapa scenario constraint, engine tertentu bisa berjalan lama. Jika banyak step timeout, naikkan timeout atau turunkan limit terlebih dahulu.

## 7. Jalankan Comparison Experiment

Untuk membandingkan DiCE dan OCEAN dengan desain eksperimen yang sama:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_comparison.py --scenario-limit 5 --stability-limit 5 --repeat-count 3 --scenario-timeout-seconds 120 --stability-timeout-seconds 120
```

Comparison runner menjalankan baseline setiap engine dengan:

- scenario config yang sama,
- limit case yang sama,
- repeat count stability yang sama,
- timeout yang sama.

Output akan dibuat di:

```text
experiments/results/comparisons/<timestamp>/
```

File penting:

- `comparison_report.md`
- `comparison_manifest.json`
- `combined/scenario_summary.csv`
- `combined/stability_summary.csv`
- `combined/candidates.csv`
- `baselines/<engine>/<timestamp>/report.md`

Comparison runner juga menulis pointer hasil terbaru:

```text
experiments/results/latest/comparison.txt
```

Audit hasil comparison terbaru:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\audit_comparison.py
```

Audit ini memeriksa file wajib, engine wajib, failed step, violation rate, dan stability evaluability. Timeout dan infeasible case dicatat sebagai warning agar tetap terlihat sebagai temuan eksperimen, bukan crash pipeline.

Gunakan command ini sebagai jalur utama ketika tujuan eksperimen adalah membandingkan beberapa library.

## 8. Jalankan Checkpoint Terkunci

Untuk menjalankan comparison dan langsung mengaudit hasilnya dalam satu command:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_checkpoint.py --scenario-limit 5 --stability-limit 5 --repeat-count 3 --scenario-timeout-seconds 120 --stability-timeout-seconds 120
```

Checkpoint runner melakukan:

- menjalankan `run_comparison.py`,
- mengumpulkan output comparison,
- menjalankan `audit_comparison.py`,
- menulis `checkpoint_report.md`,
- menulis `checkpoint_manifest.json`,
- keluar dengan exit code non-zero jika audit gagal.

Gunakan command ini saat ingin mengunci satu hasil eksperimen sebagai checkpoint teknis yang siap dibandingkan.

## 9. Diagnosa OCEAN

Setelah comparison atau checkpoint selesai, jalankan diagnosa OCEAN:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\diagnose_ocean.py
```

Secara default script ini membaca comparison terbaru dari:

```text
experiments/results/latest/comparison.txt
```

Output akan ditulis di folder comparison yang sama:

```text
ocean_diagnostics.md
ocean_diagnostics.json
```

Diagnosa ini membantu membaca:

- scenario mana yang feasibility OCEAN-nya rendah,
- scenario mana yang OCEAN-nya lebih buruk dari DiCE,
- reason counts OCEAN per scenario,
- overlap case antara OCEAN dan DiCE,
- fitur yang paling sering diubah oleh OCEAN.

Untuk membandingkan varian tuning OCEAN, gunakan beberapa engine config dalam satu checkpoint:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\run_checkpoint.py --engine-configs experiments\configs\engines\dice.json experiments\configs\engines\ocean.json experiments\configs\engines\ocean_attempt4.json --scenario-limit 5 --stability-limit 5 --repeat-count 3 --scenario-timeout-seconds 180 --stability-timeout-seconds 180 --required-engines dice ocean ocean_attempt4
```

Lalu diagnosa varian tuning tersebut:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\diagnose_ocean.py --target-engine ocean_attempt4 --baseline-engine ocean
```

## 10. Tampilkan Quick Report Baseline

```powershell
.\.venv\Scripts\python.exe experiments\scripts\print_baseline_report.py experiments\results\<timestamp>_<engine>_baseline
```

Report menampilkan:

- feasible rate per scenario,
- target success rate,
- immutable/mutable/bounds/directional violation rate,
- mean runtime,
- mean LOF,
- stability aggregate, termasuk feasible-only stability,
- top changed features.

Script ini juga bisa membaca baseline yang belum selesai sepenuhnya, selama sudah ada `summary.csv` atau `candidates.csv` parsial.

## 11. Gabungkan Semua Hasil Eksperimen

```powershell
.\.venv\Scripts\python.exe experiments\scripts\collect_results.py
```

Output gabungan:

```text
experiments/results/combined/scenario_summary.csv
experiments/results/combined/stability_summary.csv
experiments/results/combined/candidates.csv
```

## 12. Config yang Tersedia

```text
experiments/configs/engines/dice.json
experiments/configs/engines/ocean.json
experiments/configs/engines/ocean_attempt4.json
experiments/configs/scenarios/all_mutable.json
experiments/configs/scenarios/lifestyle_combo.json
experiments/configs/scenarios/bmi_only.json
experiments/configs/scenarios/activity_only.json
experiments/configs/scenarios/tight_bounds.json
experiments/configs/scenarios/no_mutable.json
experiments/configs/scenarios/stability.json
```

Engine config menentukan generator counterfactual yang dipakai. Scenario config menentukan constraint dan kasus uji. Runner akan menggabungkan keduanya sebelum benchmark berjalan.

Engine config juga dibuat eksplisit untuk parameter generation dasar:

- DiCE memakai `dice_genetic`, `total_cfs=3`, `timeout_ms=5000`.
- OCEAN memakai `ocean_cp`, `total_cfs=1`, `timeout_ms=15000`, dan deterministic multi-attempt lewat `engine_options`.
- `ocean_attempt4.json` memakai adapter OCEAN yang sama, tetapi dilabeli `ocean_attempt4`, dengan `attempt_count=4` dan `timeout_ms=30000` untuk tuning eksploratif.

Nilai engine config menimpa nilai scenario config ketika keduanya mendefinisikan field yang sama. Karena itu, perubahan parameter generation per engine sebaiknya dibuat sebagai config engine baru, bukan mengubah scenario.

`engine` adalah nama adapter yang dijalankan. `engine_label` adalah nama hasil eksperimen di folder output dan report. Gunakan `engine_label` saat membandingkan beberapa konfigurasi dari adapter yang sama, misalnya `ocean` vs `ocean_attempt4`.

Scenario config dijaga engine-neutral. Scenario tidak mendefinisikan `generation_method`, `total_cfs`, atau `timeout_ms`; field tersebut tinggal di engine config.

## 13. Output dan Git

Semua hasil eksperimen berada di:

```text
experiments/results/
```

Folder ini di-ignore oleh Git. Artinya hasil run lokal tidak otomatis ikut commit.

Layout output baru dikelompokkan seperti ini:

```text
experiments/results/
  benchmarks/<engine>/<timestamp>/
  scenarios/<engine>/<timestamp>/
  stability/<engine>/<timestamp>/
  baselines/<engine>/<timestamp>/
  comparisons/<timestamp>/
  combined/
  latest/
```

## 14. Notebook

Notebook berada di:

```text
experiments/notebooks/
```

Saat ini pipeline eksperimen utama dijalankan lewat script agar reproducible. Notebook dapat digunakan sebagai tempat eksplorasi atau analisis visual dengan membaca file CSV dari `experiments/results/`.
