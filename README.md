# diabetify-cf

Counterfactual microservice untuk Diabetify berbasis spesifikasi:

- Queue request: `ml.cf.request`
- Queue response: `ml.cf.response`

## Current Phase

Phase 4 (prescriptive planning):

- Contract schema (request/response) sudah dibuat.
- RabbitMQ consumer/publisher sudah dibuat.
- `NN` counterfactual engine aktif sebagai default service engine jika artifact tersedia.
- `DiCE` tetap tersedia sebagai provider alternatif untuk debugging atau pembandingan.
- Hardening aktif: target probability enforcement, actionable-feature filtering, timeout-aware reason code, dan plausibility gate via LOF threshold.
- Hardening klinis tambahan: directional constraints per fitur (mis. aktivitas fisik tidak boleh direkomendasikan menurun).
- Prescriptive planner aktif dengan mode `template` default.
- Integrasi OpenAI planner tersedia dengan fallback otomatis ke template jika API key tidak ada/gagal, dan kini menerima konteks counterfactual before/after yang lebih kaya.

## Folder Structure

```text
diabetify-cf/
  artifacts/
    reference/
      reference_data.parquet
    models/
    results/
  configs/
    feature_registry.json
  experiments/
    notebooks/
    scripts/
    configs/
    results/
  src/diabetify_cf/
    engine/
    messaging/
    planner/
    app.py
    config.py
    reason_codes.py
    schemas.py
  tests/
  pyproject.toml
  .env.example
```

## Local Setup

1. Buat virtual environment.
2. Install dependencies.
3. Salin `.env.example` menjadi `.env`.
4. Jalankan service.

```powershell
cd diabetify-cf
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
python -m diabetify_cf.app
```

Default local runtime memakai `CF_ENGINE_PROVIDER=nn`, jadi dependency runtime
tetap ringan dan tidak meng-install `dice-ml`.

Jika ingin menjalankan provider DiCE pada service, install extra `dice`:

```powershell
pip install -e ".[dev,dice]"
```

For experiment engines beyond DiCE, install the experiment extra:

```powershell
pip install -e ".[dev,experiments]"
```

## Quality Commands

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m ruff check --no-cache src tests
python -m black --check src tests experiments --exclude notebooks
python -m mypy src
python -m pytest -q tests
```

## Docker

```powershell
cd diabetify-cf
docker build -t diabetify-cf:dev .
docker run --rm --env-file .env diabetify-cf:dev
```

For the normal local full stack, use the provided compose file so the worker
reuses RabbitMQ from `diabetify-be` on host port `5672`:

```powershell
cd diabetify-cf
docker compose up --build -d
```

## Research and Experiment Layout

- `src/` berisi kode service yang dipakai modul Diabetify.
- `experiments/notebooks/` berisi notebook eksplorasi counterfactual engine.
- `experiments/scripts/` disiapkan untuk benchmark dan evaluasi metrik.
- `experiments/results/` disiapkan untuk output eksperimen lokal dan tidak ditujukan untuk Git.
- Model XGBoost yang dipakai service dan benchmark dibekukan sebagai artefak lokal di `artifacts/models/`.

## Experiment Commands

```powershell
python experiments/scripts/check_engine_availability.py
python experiments/scripts/run_benchmark.py --engine-config experiments/configs/engines/dice.json --scenario-config experiments/configs/scenarios/all_mutable.json
python experiments/scripts/run_benchmark.py --engine-config experiments/configs/engines/ocean.json --scenario-config experiments/configs/scenarios/all_mutable.json
python experiments/scripts/summarize_results.py experiments/results/<run-folder>
python experiments/scripts/run_scenarios.py --limit 1 --timeout-seconds 60
python experiments/scripts/evaluate_stability.py --engine-config experiments/configs/engines/dice.json --scenario-config experiments/configs/scenarios/stability.json --limit 1 --repeat-count 2
python experiments/scripts/collect_results.py
python experiments/scripts/run_baseline.py --engine-config experiments/configs/engines/dice.json --scenario-limit 1 --stability-limit 1 --repeat-count 2 --scenario-timeout-seconds 60 --stability-timeout-seconds 60
python experiments/scripts/run_comparison.py --scenario-limit 20 --stability-limit 10 --repeat-count 5 --scenario-timeout-seconds 180 --stability-timeout-seconds 180
python experiments/scripts/audit_comparison.py
python experiments/scripts/print_baseline_report.py experiments/results/<baseline-folder>
```

Baseline runs create a readable `report.md` in the baseline folder and update
`experiments/results/latest/baseline.txt`.
Comparison runs create `comparison_report.md` and update
`experiments/results/latest/comparison.txt`.
Comparison audits create `audit_report.json` in the comparison folder.
Stability reports separate all-repeat stability from feasible-only stability, so
an engine that consistently fails is not treated as a stable counterfactual
generator.

## Notes

- Service default menjalankan engine `NN`; set `CF_ENGINE_PROVIDER=dice` untuk provider DiCE.
- Runtime Docker/service default tidak meng-install `dice-ml`; gunakan extra `dice`
  atau `experiments` hanya saat memang butuh jalur DiCE.
- Service membutuhkan `CF_MODEL_PATH` dan `CF_COLUMNS_PATH` valid.
- Default `CF_REFERENCE_DATA_PATH` mengarah ke `artifacts/reference/reference_data.parquet`.
- Input `instance.features` wajib berisi seluruh fitur model pada `x_columns.pkl`.
- `CF_MAX_LOF_SCORE` mengatur batas maksimum skor LOF kandidat (semakin kecil semakin ketat).
- `CF_NN_*` mengatur candidate pool, jumlah neighbor yang diproyeksikan, dan sparsity projection untuk engine `NN`.
- Planner default `CF_PLANNER_PROVIDER=template`; set `openai` + `OPENAI_API_KEY` untuk narasi LLM.
- CARLA is not enabled because `carla-recourse==0.0.5` pins `numpy==1.19.4`, which conflicts with this project's `numpy==2.2.6` stack.
