# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY backend/requirements ./backend/requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --no-cache-dir -r ./requirements.txt

COPY backend/app ${APP_HOME}/app
COPY frontend ${APP_HOME}/frontend
COPY brainbay_company.txt ${APP_HOME}/brainbay_company.txt
COPY brainbay_market.txt ${APP_HOME}/brainbay_market.txt
COPY brainbay_geography.txt ${APP_HOME}/brainbay_geography.txt
COPY brainbay_matching.txt ${APP_HOME}/brainbay_matching.txt
COPY infra/docker/entrypoint.sh /usr/local/bin/entrypoint

RUN chmod +x /usr/local/bin/entrypoint \
    && groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser \
    && mkdir -p ${APP_HOME}/.cache/huggingface \
    && chown -R appuser:appuser ${APP_HOME}

USER appuser

ENV TRANSFORMERS_CACHE=${APP_HOME}/.cache/huggingface \
    HF_HOME=${APP_HOME}/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    COMPANY_INFO_PATH=${APP_HOME}/brainbay_company.txt \
    BRAINBAY_MARKET_PATH=${APP_HOME}/brainbay_market.txt \
    BRAINBAY_GEOGRAPHY_PATH=${APP_HOME}/brainbay_geography.txt \
    BRAINBAY_MATCHING_PATH=${APP_HOME}/brainbay_matching.txt \
    MODEL_QUANTIZATION=auto \
    PORT=8000 \
    APP_MODULE=app.main:app \
    APP_HOST=0.0.0.0

EXPOSE 8000

ENTRYPOINT ["entrypoint"]
CMD []
