FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# sys deps minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Pre-download spaCy model + SBERT at build-time to avoid first-request lag
RUN python -m spacy download en_core_web_sm
# Pull a light, high-quality sentence transformer
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
PY

# Copy code
COPY . .

EXPOSE 5000

# Basic healthcheck (FastAPI /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","5000"]