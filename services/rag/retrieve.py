import os
import faiss
import pickle
import numpy as np
from typing import List, Dict
from ..observability.langfuse_client import observe
from .embed import get_embedder

class Retriever:
    def __init__(self, index_dir: str):
        self.index_path = os.path.join(index_dir, "vector.index")
        self.doc_store_path = os.path.join(index_dir, "doc_store.pkl")
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.doc_store_path):
            raise FileNotFoundError(f"Index or Doc Store not found in {index_dir}")
            
        print(f"Loading index from {index_dir}...")
        self.index = faiss.read_index(self.index_path)
        
        with open(self.doc_store_path, 'rb') as f:
            self.chunks = pickle.load(f)
            
        self.embedder = get_embedder()
        
    @observe(name="retrieve")
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve chunks relevant to the query.
        """
        # Embed query
        query_embedding = self.embedder.embed([query])
        
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
             # Normalize if using cosine similarity (Inner Product)
             faiss.normalize_L2(query_embedding)
             
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue # invalid
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)
            results.append(chunk)
            
        return results

_shared_retriever = None

def get_retriever(index_dir: str = "data/index"):
    global _shared_retriever
    if _shared_retriever is None:
        try:
             _shared_retriever = Retriever(index_dir)
        except Exception as e:
            print(f"Failed to load retriever: {e}")
            return None
    return _shared_retriever
