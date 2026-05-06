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
  --scenario-limit 10 `
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

Keterangan:
- `latest/` berisi pointer `.txt` ke run terbaru,
- `combined/` berisi CSV gabungan,
- folder timestamp menyimpan artefak run aktual.

## Notebook

Notebook di `experiments/notebooks/` dipakai untuk:
- membaca report hasil benchmark,
- merangkum scenario/stability/candidate metrics,
- menyiapkan analisis untuk penulisan TA.

Notebook bukan source of truth eksekusi eksperimen. Eksekusi utama tetap lewat script di `experiments/scripts/`.

## Catatan

- `run_core_checkpoint.py` adalah jalur utama benchmark final.
- `run_benchmark.py`, `run_baseline.py`, dan `run_comparison.py` lebih cocok untuk debug atau eksperimen kecil.
- `collect_results.py` sebaiknya dipakai hanya pada root hasil yang memang satu scope, misalnya satu `comparison_root` atau satu `baseline_root`.
