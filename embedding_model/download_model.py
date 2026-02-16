from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Set the cache directory if not already set, though env var in Dockerfile is better
# print(f"Downloading model to {os.environ.get('HF_HOME', 'default location')}")

# Initialize the embedding model to trigger download
embedder = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
print("Model downloaded successfully.")
