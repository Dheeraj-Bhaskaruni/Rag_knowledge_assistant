from typing import List, Dict
from sentence_transformers import CrossEncoder
import os
from ..observability.langfuse_client import observe

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # We can make this optional/lazy load to speed startup if not used
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        
    @observe(name="rerank")
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []
            
        pairs = [[query, c['content']] for c in chunks]
        scores = self.model.predict(pairs)
        
        for i, score in enumerate(scores):
            chunks[i]['rerank_score'] = float(score)
            
        # Resort
        chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        return chunks[:top_k]

_shared_reranker = None

def get_reranker():
    global _shared_reranker
    if _shared_reranker is None:
        # Default to a small fast cross encoder
        _shared_reranker = Reranker()
    return _shared_reranker
