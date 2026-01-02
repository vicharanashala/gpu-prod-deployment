FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for cv2 if torchvision needs it indirectly, though slim usually ok for basic pilot)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose port
EXPOSE 8029

# Set default environment variable for the app module
ENV APP_MODULE=effnet_api:app

CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port 8029"]
