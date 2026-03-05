# diabetify-cf

Counterfactual microservice untuk Diabetify berbasis spesifikasi:
- Queue request: `ml.cf.request`
- Queue response: `ml.cf.response`
- Contract: `docs/CF_SPEC_V1.md`

## Current Phase
Phase 4 (prescriptive planning):
- Contract schema (request/response) sudah dibuat.
- RabbitMQ consumer/publisher sudah dibuat.
- DICE real engine sudah aktif jika artifact tersedia.
- Mode stub tetap tersedia untuk plumbing test.
- Hardening aktif: target probability enforcement, actionable-feature filtering, timeout-aware reason code, dan plausibility gate via LOF threshold.
- Hardening klinis tambahan: directional constraints per fitur (mis. aktivitas fisik tidak boleh direkomendasikan menurun).
- Prescriptive planner aktif dengan mode `template` default.
- Integrasi OpenAI planner tersedia dengan fallback otomatis ke template jika API key tidak ada/gagal.

## Folder Structure
```text
diabetify-cf/
  src/diabetify_cf/
    engine/
    messaging/
    planner/
    app.py
    config.py
    reason_codes.py
    schemas.py
  tests/
  requirements.txt
  requirements-dev.txt
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
pip install -r requirements-dev.txt
pip install -e .
Copy-Item .env.example .env
python -m diabetify_cf.app
```

## Quality Commands
```powershell
ruff check .
black --check .
mypy src
pytest -q
```

## Docker
```powershell
cd diabetify-cf
docker build -t diabetify-cf:dev .
docker run --rm --env-file .env diabetify-cf:dev
```

## Notes
- Default `CF_ALLOW_STUB_FEASIBLE=false` akan menjalankan engine DICE real.
- Mode real membutuhkan `CF_MODEL_PATH` dan `CF_COLUMNS_PATH` valid.
- Jika butuh uji plumbing end-to-end tanpa model, set `CF_ALLOW_STUB_FEASIBLE=true`.
- Input `instance.features` wajib berisi seluruh fitur model pada `x_columns.pkl`.
- `CF_MAX_LOF_SCORE` mengatur batas maksimum skor LOF kandidat (semakin kecil semakin ketat).
- Planner default `CF_PLANNER_PROVIDER=template`; set `openai` + `OPENAI_API_KEY` untuk narasi LLM.
