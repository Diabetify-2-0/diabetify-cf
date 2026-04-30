FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CF_REFERENCE_DATA_PATH=/app/artifacts/reference/reference_data.parquet
ENV CF_FEATURE_REGISTRY_PATH=/app/configs/feature_registry.json
ENV CF_MODEL_PATH=/app/artifacts/models/xg_model.pkl
ENV CF_COLUMNS_PATH=/app/artifacts/models/x_columns.pkl

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
COPY artifacts/reference ./artifacts/reference

RUN pip install --no-cache-dir .
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/artifacts/models \
    && chown -R appuser:appuser /app

USER appuser
CMD ["python", "-m", "diabetify_cf.app"]
