# Dockerfile for Stress and Trauma Detection Backend
# Supports local CPU and NVIDIA CUDA GPU execution

FROM python:3.10-slim

# Install system dependencies:
# - ffmpeg / libsndfile1 for audio decoding (pydub & soundfile)
# - build-essential for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and models placeholder
COPY app/ ./app/
COPY models/ ./models/
COPY .env.example ./.env

EXPOSE 8000

# Default execution using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
