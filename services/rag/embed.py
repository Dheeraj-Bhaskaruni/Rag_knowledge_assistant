import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        self.use_openai = use_openai
        self.model_name = model_name
        
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            self.client = OpenAI(api_key=api_key)
        else:
            print(f"Loading local embedding model: {model_name}")
            # Force CPU to avoid ZeroGPU conflicts during ingestion
            self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
            
        if self.use_openai:
            # Batching might be needed for very large lists, but keeping simple for now
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings, dtype='float32')
        else:
            return self.model.encode(texts, convert_to_numpy=True)

def get_embedder():
    # Factory to get configured embedder
    # Prefer OpenAI if specified, else local
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    model_name = "text-embedding-3-small" if use_openai else "all-MiniLM-L6-v2"
    
    return Embedder(model_name=model_name, use_openai=use_openai)
