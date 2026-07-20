FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (separate index, not on regular PyPI)
# Increased timeout/retries since this is a large (~190MB) download
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
