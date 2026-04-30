# Experiments

Panduan ini diasumsikan dijalankan dari root repository:

```powershell
cd diabetify-cf
```

Gunakan Python dari virtual environment project:

```powershell
.\.venv\Scripts\python.exe
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

Untuk saat ini, engine yang aktif adalah DiCE. CARLA, OCEAN, dan FOCUS akan muncul sebagai missing sampai dependency-nya dipasang.

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

## 7. Tampilkan Quick Report Baseline

```powershell
.\.venv\Scripts\python.exe experiments\scripts\print_baseline_report.py experiments\results\<timestamp>_<engine>_baseline
```

Report menampilkan:

- feasible rate per scenario,
- target success rate,
- immutable/mutable/bounds/directional violation rate,
- mean runtime,
- mean LOF,
- stability aggregate,
- top changed features.

Script ini juga bisa membaca baseline yang belum selesai sepenuhnya, selama sudah ada `summary.csv` atau `candidates.csv` parsial.

## 8. Gabungkan Semua Hasil Eksperimen

```powershell
.\.venv\Scripts\python.exe experiments\scripts\collect_results.py
```

Output gabungan:

```text
experiments/results/combined/scenario_summary.csv
experiments/results/combined/stability_summary.csv
experiments/results/combined/candidates.csv
```

## 9. Config yang Tersedia

```text
experiments/configs/engines/dice.json
experiments/configs/scenarios/all_mutable.json
experiments/configs/scenarios/bmi_only.json
experiments/configs/scenarios/activity_only.json
experiments/configs/scenarios/tight_bounds.json
experiments/configs/scenarios/no_mutable.json
experiments/configs/scenarios/stability.json
```

Engine config menentukan generator counterfactual yang dipakai. Scenario config menentukan constraint dan kasus uji. Runner akan menggabungkan keduanya sebelum benchmark berjalan.

## 10. Output dan Git

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
  combined/
  latest/
```

## 11. Notebook

Notebook berada di:

```text
experiments/notebooks/
```

Saat ini pipeline eksperimen utama dijalankan lewat script agar reproducible. Notebook dapat digunakan sebagai tempat eksplorasi atau analisis visual dengan membaca file CSV dari `experiments/results/`.
