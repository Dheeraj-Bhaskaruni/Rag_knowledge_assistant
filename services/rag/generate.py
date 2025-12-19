import os
from typing import List, Dict
from openai import OpenAI
from ..observability.langfuse_client import observe
import torch

SYSTEM_PROMPT = """You are a grounded knowledge assistant. 
Your goal is to answer the user's question using ONLY the provided context.

Rules:
1. Use the provided context to answer the question.
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Cite your sources for every fact using the format [doc_id:chunk_id].
4. Do not make up information.
5. Be concise and direct.
"""

# Global variable for lazy loading on the worker node
_local_pipeline = None

class GeneratorService:
    def __init__(self):
        # Initialize OpenAI if key exists
        self.openai_client = None
        self.openai_model = "gpt-4o-mini" # User reported gpt-5 errors, safer default? Or keep logic.
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            print("Warning: OPENAI_API_KEY not found. OpenAI backend will not work.")

    def _format_context(self, chunks: List[Dict]) -> str:
        context_str = ""
        for c in chunks:
            location = c['metadata']['chunk_id']
            text = c['content']
            context_str += f"<SOURCE ID='{location}'>\n{text}\n</SOURCE>\n\n"
        return context_str

    @observe(name="generate")
    def generate(self, query: str, context_chunks: List[Dict], backend: str = "openai") -> str:
        """
        backend: 'openai' or 'local'
        NOTE: This method must be running in a @spaces.GPU context if backend='local'.
        """
        context = self._format_context(context_chunks)
        
        # Check explicit backend choice
        if backend == "openai":
            if self.openai_client is None:
                 return "Error: OpenAI backend selected but OPENAI_API_KEY not found. Please switch to Local or set key."
            
            # OpenAI Generation
            try:
                # Basic Chat Completion
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"

        else:
            # Local Generation (Mistral)
            # This block expects to be running on ZeroGPU (enforced by app.py decorator)
            
            global _local_pipeline
            
            # Lazy Load the model here (on the GPU node)
            if _local_pipeline is None:
                print("Loading local Mistral-7B model (Lazy Load)...")
                try:
                    from transformers import pipeline
                    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
                    _local_pipeline = pipeline(
                        "text-generation", 
                        model=model_id, 
                        torch_dtype=torch.float16, 
                        device_map="auto" 
                    )
                except Exception as e:
                    return f"Failed to load local model: {e}"
            
            # Prepare messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            try:
                # Execute pipeline (already on GPU context from app.py)
                outputs = _local_pipeline(
                    messages,
                    max_new_tokens=512,
                    do_sample=True, 
                    temperature=0.1, 
                    top_k=50, 
                    top_p=0.95
                )
                # Parse output
                result = outputs[0]['generated_text']
                if isinstance(result, list):
                     return result[-1]['content']
                return str(result)
            except Exception as e:
                return f"Generation Error: {e}"

_shared_generator = None

def get_generator():
    global _shared_generator
    if _shared_generator is None:
        _shared_generator = GeneratorService()
    return _shared_generator
