from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uvicorn

app = FastAPI(title="Embedding Service")

# Initialize the embedding model
# We use BAAI/bge-large-en-v1.5 as requested
# cache_folder can be set if we want to persist models in a volume
embedder = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = embedder

class SingleEmbedRequest(BaseModel):
    text: str = Field(..., description="The text string to embed")

class SingleEmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector for the input text")

@app.post("/embed", response_model=SingleEmbedResponse)
def generate_single_embedding(req: SingleEmbedRequest):
    """
    Generate an embedding for a single text using the Hugging Face model via LlamaIndex.

    **Example Input:**
    {
      "text": "What is sustainable farming?"
    }
    
    **Example Output:**
    {
      "embedding": [0.0123, -0.0542, ...]
    }
    """
    embedding = embedder.get_text_embedding(req.text)
    return SingleEmbedResponse(embedding=embedding)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
