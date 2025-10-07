FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY frontend /app/frontend
COPY brainbay_company.txt /app/brainbay_company.txt
COPY brainbay_market.txt /app/brainbay_market.txt
COPY brainbay_geography.txt /app/brainbay_geography.txt
COPY brainbay_matching.txt /app/brainbay_matching.txt

ENV TRANSFORMERS_CACHE=/app/.cache/huggingface HF_HOME=/app/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 PORT=8000 COMPANY_INFO_PATH=/app/brainbay_company.txt BRAINBAY_MARKET_PATH=/app/brainbay_market.txt BRAINBAY_GEOGRAPHY_PATH=/app/brainbay_geography.txt BRAINBAY_MATCHING_PATH=/app/brainbay_matching.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
