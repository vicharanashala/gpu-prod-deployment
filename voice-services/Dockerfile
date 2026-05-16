FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py tts_language_detection.py ./

EXPOSE 8080

ENV HOST=0.0.0.0 \
    PORT=8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
