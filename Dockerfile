FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CF_REFERENCE_DATA_PATH=/app/artifacts/reference/reference_data.parquet
ENV CF_FEATURE_REGISTRY_PATH=/app/configs/feature_registry.json
ENV CF_MODEL_PATH=/app/artifacts/models/xg_model.pkl
ENV CF_COLUMNS_PATH=/app/artifacts/models/x_columns.pkl

WORKDIR /app

COPY pyproject.toml .

RUN python -c "import pathlib, tomllib; deps = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']; pathlib.Path('requirements.txt').write_text('\n'.join(deps) + '\n', encoding='utf-8')" \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY artifacts/models ./artifacts/models
COPY artifacts/reference ./artifacts/reference

RUN pip install --no-cache-dir --no-deps .
RUN useradd --create-home --uid 10001 appuser \
    && rm -f requirements.txt \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "diabetify_cf.app"]
