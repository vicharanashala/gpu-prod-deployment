FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for cv2 if torchvision needs it indirectly, though slim usually ok for basic pilot)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY effnet_api.py .
COPY effnet_model ./effnet_model

# Expose port
EXPOSE 8029

CMD ["uvicorn", "effnet_api:app", "--host", "0.0.0.0", "--port", "8029"]
