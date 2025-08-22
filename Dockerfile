FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# install deps first (better caching)
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# download spaCy model & SBERT at build time (avoid first-request lag)
RUN python -m spacy download en_core_web_sm
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PY

# copy code + seed data
COPY src ./src
COPY data ./data

# make package importable
ENV PYTHONPATH=/app/src

EXPOSE 5000

# basic healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:5000/health || exit 1

# Debug: Add verbose logging and explicit uvicorn configuration
CMD ["uvicorn", "ats_nlp.main:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "debug", "--access-log"]