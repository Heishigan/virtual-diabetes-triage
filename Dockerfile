FROM python:3.11-slim as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

COPY ./requirements.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt


FROM python:3.11-slim

WORKDIR /usr/src/app

COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .

RUN pip install --no-cache /wheels/*

ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

COPY ./src /usr/src/app/src
COPY ./release_assets/model-${MODEL_VERSION}.joblib /app/model.joblib
COPY ./release_assets/scaler-${MODEL_VERSION}.joblib /app/scaler.joblib

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
