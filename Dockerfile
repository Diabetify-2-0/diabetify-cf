FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
COPY artifacts/reference ./artifacts/reference

RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "diabetify_cf.app"]
