# Lightweight Python image
FROM python:3.12-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools (for faiss, numpy etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY rag ./rag
COPY src ./src
COPY frontend ./frontend
COPY data ./data
COPY README.md .

# Cloud Run will set PORT → default 8080
ENV PORT=8080

# Start FastAPI with uvicorn
CMD ["sh", "-c", "uvicorn rag.app:app --host 0.0.0.0 --port ${PORT}"]
