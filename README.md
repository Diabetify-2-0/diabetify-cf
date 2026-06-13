# diabetify-cf

Counterfactual microservice untuk Diabetify berbasis spesifikasi:

- Queue request: `ml.cf.request`
- Queue response: `ml.cf.response`

## Current Phase

Phase 4 (counterfactual service hardening):

- Contract schema (request/response) sudah dibuat.
- RabbitMQ consumer/publisher sudah dibuat.
- `NN` counterfactual engine aktif sebagai satu-satunya production engine jika artifact tersedia.
- Hardening aktif: target probability enforcement, actionable-feature filtering, timeout-aware reason code, dan plausibility gate via LOF threshold.
- Hardening klinis tambahan: directional constraints per fitur (mis. aktivitas fisik tidak boleh direkomendasikan menurun).

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
    tests/
  src/diabetify_cf/
    engine/
    messaging/
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
tetap ringan dan tidak membawa dependency eksperimen seperti `dice-ml`.

Untuk stack eksperimen yang terpisah dari service produksi, install extra `experiments`:

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

Test di `tests/` dikhususkan untuk counterfactual service yang dipakai
Diabetify 2.0. Test eksperimen dipisahkan di `experiments/tests/` dan
dijalankan terpisah saat memang mengevaluasi pipeline riset:

```powershell
python -m pytest -q experiments/tests
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
- `src/diabetify_cf/verification/` berisi verifier eksternal untuk memvalidasi
  kandidat returned dari service produksi secara independen, serta scenario
  runner untuk mengagregasi metrik evaluasi produksi.
- `experiments/notebooks/` berisi notebook eksplorasi counterfactual engine.
- `experiments/scripts/` disiapkan untuk benchmark dan evaluasi metrik.
- `experiments/results/` disiapkan untuk output eksperimen lokal dan tidak ditujukan untuk Git.
- Model XGBoost yang dipakai service dan benchmark dibekukan sebagai artefak lokal di `artifacts/models/`.

## Experiment Commands

```powershell
python experiments/scripts/check_engine_availability.py
python experiments/scripts/run_benchmark.py --engine-config experiments/configs/engines/dice.json --scenario-config experiments/configs/scenarios/all_mutable.json
python experiments/scripts/summarize_results.py experiments/results/<run-folder>
python experiments/scripts/run_scenarios.py --limit 1 --timeout-seconds 60
python experiments/scripts/evaluate_stability.py --engine-config experiments/configs/engines/dice.json --scenario-config experiments/configs/scenarios/stability.json --limit 1 --repeat-count 2
python experiments/scripts/collect_results.py
python experiments/scripts/run_baseline.py --engine-config experiments/configs/engines/dice.json --scenario-limit 1 --stability-limit 1 --repeat-count 2 --scenario-timeout-seconds 60 --stability-timeout-seconds 60
python experiments/scripts/run_comparison.py --scenario-limit 20 --stability-limit 10 --repeat-count 5 --scenario-timeout-seconds 180 --stability-timeout-seconds 180
python experiments/scripts/audit_comparison.py
python experiments/scripts/print_baseline_report.py experiments/results/<baseline-folder>
```

## Production Verification

Layer verifikasi produksi dibagi menjadi dua:

- `ExternalCounterfactualVerifier` untuk memvalidasi ulang kandidat returned
  secara independen terhadap model, LOF, dan gate constraint produksi.
- `ScenarioRunner` untuk menjalankan kumpulan skenario dan mengagregasi
  metrik seperti immutable violation rate, target satisfaction, infeasible
  handling accuracy, repeatability, dan latency.
- `BackendCounterfactualEngineAdapter` untuk menjalankan fixture yang sama
  melalui alur asynchronous `diabetify-be`, sehingga metrik dihitung dari
  flow backend produksi, bukan hanya pemanggilan engine langsung.

Fixture skenario produksi dapat disimpan di `configs/verification/` lalu
dijalankan melalui entry point berikut:

```powershell
python -m diabetify_cf.verification.run_service_scenarios --scenarios configs/verification
```

Secara default report JSON akan ditulis ke
`artifacts/verification/service_verification_report.json`.

Runner juga dapat difilter berdasarkan tag fixture agar suite feasible,
infeasible, atau repeatability dapat dijalankan terpisah:

```powershell
python -m diabetify_cf.verification.run_service_scenarios --scenarios configs/verification --include-tag repeatability
python -m diabetify_cf.verification.run_service_scenarios --scenarios configs/verification --exclude-tag repeatability
```

Fixture produksi yang saat ini sudah dikalibrasi terhadap engine mencakup:

- `feasible_bmi_activity`
- `feasible_bmi_activity_repeatability`
- `feasible_target_already_satisfied`
- `infeasible_no_mutable`
- `infeasible_target_unreachable_bmi_only`
- `infeasible_medical_rule_only_high_target`

Untuk menjalankan fixture yang sama melalui backend Diabetify yang sudah
terautentikasi:

```powershell
python -m diabetify_cf.verification.run_backend_scenarios --scenarios configs/verification --backend-base-url http://localhost:8080 --backend-bearer-token <token>
```

Secara default report JSON backend akan ditulis ke
`artifacts/verification/backend_verification_report.json`.

Runner backend secara default akan melakukan preflight ke
`/counterfactual/health` dan menunggu sampai backend melaporkan
`running=true` serta `rabbitmq_connected=true`. Preflight ini dapat dilewati
secara eksplisit bila environment integrasi memang tidak mengekspos route
health:

```powershell
python -m diabetify_cf.verification.run_backend_scenarios --scenarios configs/verification --backend-base-url http://localhost:8080 --backend-bearer-token <token> --skip-health-check
```

Untuk menjalankan beberapa suite backend sekaligus dan menghasilkan satu
report JSON per suite plus manifest `index.json`:

```powershell
python -m diabetify_cf.verification.run_backend_suite --scenarios configs/verification --backend-base-url http://localhost:8080 --backend-bearer-token <token>
python -m diabetify_cf.verification.run_backend_suite --scenarios configs/verification --backend-base-url http://localhost:8080 --backend-bearer-token <token> --suite infeasible_core --suite repeatability_core
```

Untuk eksekusi yang repeatable tanpa perlu mengisi argumen panjang atau
menyalin bearer token manual setiap kali, tersedia launcher berbasis config:

```powershell
set DIABETIFY_BACKEND_BASE_URL=http://localhost:8080
set DIABETIFY_TEST_USER_EMAIL=tester@example.com
set DIABETIFY_TEST_USER_PASSWORD=replace-with-test-password
python -m diabetify_cf.verification.run_backend_suite_from_config --config configs/verification/backend_suite_launcher.example.json
```

Launcher config mendukung dua mode autentikasi:

- `auth_mode = "bearer_token"` untuk token JWT yang sudah disiapkan
- `auth_mode = "login"` untuk memperoleh token otomatis dari endpoint
  `POST /users/login` menggunakan kredensial user uji
- `register_if_missing = true` untuk membuat user uji terlebih dahulu lewat
  `POST /users` saat login gagal karena akun belum ada di environment target

Placeholder `${ENV_VAR}` di launcher config akan di-resolve saat runtime,
sehingga secret pengujian tidak perlu disimpan literal di file JSON.

Baseline runs create a readable `report.md` in the baseline folder and update
`experiments/results/latest/baseline.txt`.
Comparison runs create `comparison_report.md` and update
`experiments/results/latest/comparison.txt`.
Comparison audits create `audit_report.json` in the comparison folder.
Stability reports separate all-repeat stability from feasible-only stability, so
an engine that consistently fails is not treated as a stable counterfactual
generator.

## Notes

- Service production hanya menjalankan engine `NN`.
- Runtime Docker/service default tidak meng-install `dice-ml`; gunakan extra
  `experiments` hanya saat memang butuh jalur riset.
- Service membutuhkan `CF_MODEL_PATH` dan `CF_COLUMNS_PATH` valid.
- Default `CF_REFERENCE_DATA_PATH` mengarah ke `artifacts/reference/reference_data.parquet`.
- Input `instance.features` wajib berisi seluruh fitur model pada `x_columns.pkl`.
- `CF_MAX_LOF_SCORE` mengatur batas maksimum skor LOF kandidat (semakin kecil semakin ketat).
- `CF_NN_*` mengatur candidate pool, jumlah neighbor yang diproyeksikan, dan sparsity projection untuk engine `NN`.
- CARLA is not enabled because `carla-recourse==0.0.5` pins `numpy==1.19.4`, which conflicts with this project's `numpy==2.2.6` stack.
