FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

WORKDIR /app

COPY pyproject.toml README.md requirements-api.txt ./
COPY mibici_mlops/ ./mibici_mlops/

RUN pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "mibici_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
