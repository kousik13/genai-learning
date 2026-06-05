# Day 10 — GenAI & Agentic AI Foundation
# Dockerfile — Containerise the GenAI & Agentic AI API (Day 9 app)

# ─── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ─── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="your-email@example.com"
LABEL description="GenAI & Agentic AI Foundation API"
LABEL version="1.0.0"

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ─────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ────────────────────────────────────────────────────
# Copy Day 9 API files (adjust path if running from project root)
COPY Day09_API_App/app.py      .
COPY Day09_API_App/routes.py   .
COPY Day09_API_App/services.py .
COPY Day10_Production/logger.py .

# ─── Environment variables ───────────────────────────────────────────────────
# Pass secrets at runtime via docker run -e or docker-compose env_file
ENV GROQ_API_KEY=""
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ─── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Start the app ────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Usage ────────────────────────────────────────────────────────────────────
# Build:  docker build -t genai-agentic-api .
# Run:    docker run -e GROQ_API_KEY=your_key -p 8000:8000 genai-agentic-api
# Docs:   http://localhost:8000/docs
