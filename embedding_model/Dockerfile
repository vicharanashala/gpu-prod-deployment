FROM python:3.12-slim

WORKDIR /app

# Install build dependencies if strictly needed for specific wheels not present in slim
# For huggingface/torch, sometimes we need basic dev tools.
# However, standard pip install usually works fine on slim for these popular packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# We specify versions to avoid conflicts if possible, but for this quick setup we'll use latest compatible
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    llama-index-core \
    llama-index-embeddings-huggingface \
    torch \
    transformers

# Set environment variable for Hugging Face cache
ENV HF_HOME=/app/huggingface_cache

# Copy download script and download model
COPY download_model.py .
RUN python download_model.py && rm download_model.py

# Copy application code
COPY main.py .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
