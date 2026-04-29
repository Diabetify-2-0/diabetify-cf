FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY artifacts/reference ./artifacts/reference

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "diabetify_cf.app"]
