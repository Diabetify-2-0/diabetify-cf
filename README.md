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
  configs/
    feature_registry.json
  evaluation/
    fixtures/
    launcher/
    reports/
  src/diabetify_cf/
    engine/
    messaging/
    verification/
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

## Quality Commands

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m ruff check --no-cache src tests
python -m black --check src tests
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

## Production Layout

- `src/` berisi kode service yang dipakai modul Diabetify.
- `src/diabetify_cf/verification/` berisi verifier independen dan scenario
  runner untuk memvalidasi kandidat returned dari service produksi serta
  mengagregasi metrik evaluasi produksi.
- `evaluation/fixtures/` berisi skenario evaluasi yang dapat dijalankan ulang.
- `evaluation/launcher/` berisi launcher config untuk suite backend.
- `evaluation/reports/` berisi output JSON hasil evaluasi.
- Model XGBoost dan referensi data produksi dibekukan sebagai artefak lokal di
  `artifacts/models/` dan `artifacts/reference/`.

## Production Verification

Layer verifikasi produksi dibagi menjadi dua:

- `ExternalCounterfactualVerifier` untuk memvalidasi ulang kandidat returned
  secara independen terhadap model, LOF, dan gate constraint produksi.
- `ScenarioRunner` untuk menjalankan kumpulan skenario dan mengagregasi
  metrik seperti immutable violation rate, mutable violation rate,
  LOF violation rate, average/min/max LOF score, target satisfaction,
  infeasible handling accuracy, repeatability, dan latency.
- `BackendCounterfactualEngineAdapter` untuk menjalankan fixture yang sama
  melalui alur asynchronous `diabetify-be`, sehingga metrik dihitung dari
  flow backend produksi, bukan hanya pemanggilan engine langsung.

Fixture skenario produksi disimpan di `evaluation/fixtures/` lalu
dijalankan melalui entry point berikut:

```powershell
python -m diabetify_cf.verification.run_service_scenarios --scenarios evaluation/fixtures
```

Secara default report JSON akan ditulis ke
`evaluation/reports/service/service_verification_report.json`.

Runner juga dapat difilter berdasarkan tag fixture agar suite feasible,
infeasible, atau repeatability dapat dijalankan terpisah:

```powershell
python -m diabetify_cf.verification.run_service_scenarios --scenarios evaluation/fixtures --include-tag repeatability
python -m diabetify_cf.verification.run_service_scenarios --scenarios evaluation/fixtures --exclude-tag repeatability
```

Fixture produksi yang saat ini sudah dikalibrasi terhadap engine mencakup
suite khusus `actionability_core`, `plausibility_core`, `repeatability_core`,
dan `latency_core`. Suite actionability utama memakai
`evaluation/fixtures/actionability_profiles.json` berisi 10 profil feasible
dengan konfigurasi mutable berbeda dan 2 profil infeasible.

Untuk menjalankan fixture yang sama melalui backend Diabetify yang sudah
terautentikasi:

```powershell
python -m diabetify_cf.verification.run_backend_scenarios --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token>
```

Secara default report JSON backend akan ditulis ke
`evaluation/reports/backend/backend_verification_report.json`.

Runner backend secara default akan melakukan preflight ke
`/counterfactual/health` dan menunggu sampai backend melaporkan
`running=true` serta `rabbitmq_connected=true`. Preflight ini dapat dilewati
secara eksplisit bila environment integrasi memang tidak mengekspos route
health:

```powershell
python -m diabetify_cf.verification.run_backend_scenarios --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token> --skip-health-check
```

Untuk menjalankan beberapa suite backend sekaligus dan menghasilkan satu
report JSON per suite plus manifest `index.json`:

```powershell
python -m diabetify_cf.verification.run_backend_suite --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token>
python -m diabetify_cf.verification.run_backend_suite --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token> --suite actionability_core
python -m diabetify_cf.verification.run_backend_suite --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token> --suite plausibility_core
python -m diabetify_cf.verification.run_backend_suite --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token> --suite repeatability_core
python -m diabetify_cf.verification.run_backend_suite --scenarios evaluation/fixtures --backend-base-url http://localhost:8080 --backend-bearer-token <token> --suite latency_core
```

Untuk eksekusi yang repeatable tanpa perlu mengisi argumen panjang atau
menyalin bearer token manual setiap kali, tersedia launcher berbasis config:

```powershell
set DIABETIFY_BACKEND_BASE_URL=http://localhost:8080
set DIABETIFY_TEST_USER_EMAIL=tester@example.com
set DIABETIFY_TEST_USER_PASSWORD=replace-with-test-password
python -m diabetify_cf.verification.run_backend_suite_from_config --config evaluation/launcher/backend_suite_launcher.example.json
```

Launcher config mendukung dua mode autentikasi:

- `auth_mode = "bearer_token"` untuk token JWT yang sudah disiapkan
- `auth_mode = "login"` untuk memperoleh token otomatis dari endpoint
  `POST /users/login` menggunakan kredensial user uji
- `register_if_missing = true` untuk membuat user uji terlebih dahulu lewat
  `POST /users` saat login gagal karena akun belum ada di environment target

Launcher contoh backend suite secara default juga memasukkan `actionability_core`
agar metrik `immutable_violation_rate` dan `mutable_violation_rate` dapat
dilaporkan sebagai suite khusus untuk evaluasi constraint actionable. Suite ini
memakai 12 profil dari `artifacts/reference/reference_data.parquet`: 5 profil
non-smoker/former-smoker feasible, 5 profil active smoker feasible, dan 2 profil
infeasible.

Launcher contoh backend suite juga memasukkan `plausibility_core` agar metrik
`lof_violation_rate`, `average_lof_score`, `min_lof_score`, dan
`max_lof_score` dapat dilaporkan sebagai suite khusus untuk evaluasi
plausibility berbasis LOF. Suite ini memakai 25 profil high-risk dari
`artifacts/reference/reference_data.parquet` dengan konfigurasi full-actionable
non-smoker dan smoker.

Launcher contoh backend suite juga memasukkan `repeatability_core` agar metrik
`repeatability_rate` dapat dilaporkan sebagai suite khusus untuk evaluasi
konsistensi hasil untuk input yang identik. Suite ini memakai 12 profil
high-risk dari `artifacts/reference/reference_data.parquet`, terdiri dari 10
skenario feasible dan 2 skenario infeasible, masing-masing diulang 10 kali.
Seluruh run tetap dipakai untuk pengecekan konsistensi, sementara laporan per
profil hanya menampilkan `counterfactual_profile_run_1` sampai
`counterfactual_profile_run_3`.
Suite `latency_core` dapat dijalankan secara eksplisit untuk pengujian waktu
respons RM-4. Suite ini memakai 25 profil high-risk dari
`artifacts/reference/reference_data.parquet`, masing-masing diulang 5 kali,
dan menghasilkan metrik `average_latency_ms`, `min_latency_ms`,
`max_latency_ms`, serta `p95_latency_ms`.

Placeholder `${ENV_VAR}` di launcher config akan di-resolve saat runtime,
sehingga secret pengujian tidak perlu disimpan literal di file JSON.

## Notes

- Service production hanya menjalankan engine `NN`.
- Service membutuhkan `CF_MODEL_PATH` dan `CF_COLUMNS_PATH` valid.
- Default `CF_REFERENCE_DATA_PATH` mengarah ke `artifacts/reference/reference_data.parquet`.
- Input `instance.features` wajib berisi seluruh fitur model pada `x_columns.pkl`.
- `CF_MAX_LOF_SCORE` mengatur batas maksimum skor LOF kandidat (semakin kecil semakin ketat).
- `CF_NN_*` mengatur candidate pool, jumlah neighbor yang diproyeksikan, dan sparsity projection untuk engine `NN`.



