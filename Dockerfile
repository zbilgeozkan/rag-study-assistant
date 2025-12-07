# Hafif Python imajı
FROM python:3.12-slim

# Ortam ayarları (loglar düzgün görünsün, pyc yazmasın)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# FAISS / numpy vs. için gerekli temel paketler
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Proje dosyaları
COPY . .

# Cloud Run genelde 8080 portunu kullanır
ENV PORT=8080

# Uygulamayı başlat
CMD ["uvicorn", "rag.app:app", "--host", "0.0.0.0", "--port", "8080"]
