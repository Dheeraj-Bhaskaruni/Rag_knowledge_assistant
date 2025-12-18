import os
from typing import List, Dict
from openai import OpenAI
from ..observability.langfuse_client import observe

SYSTEM_PROMPT = """You are a grounded knowledge assistant. 
Your goal is to answer the user's question using ONLY the provided context.

Rules:
1. Use the provided context to answer the question.
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Cite your sources for every fact using the format [doc_id:chunk_id].
4. Do not make up information.
5. Be concise and direct.
"""

class GeneratorService:
    def __init__(self):
        # Initialize OpenAI if key exists
        self.openai_client = None
        self.openai_model = "gpt-5-nano" # Default
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.openai_client = OpenAI(api_key=env_key)

        # Initialize Local Pipeline (Lazy load or load now?)
        # To avoid slow startup if not used, we can lazy load, 
        # but user might want to switch instantly.
        # For now, let's load it ONLY if requested or if openai is missing?
        # Re-using strict logic: Load structure but pipeline is None
        self.local_pipeline = None

    def _ensure_local_loaded(self):
        if self.local_pipeline is None:
            # Mistral-7B for ZeroGPU (Powerful)
            # Note: This might be heavy for local machines without strong GPU/RAM.
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
            print(f"Loading local LLM ({model_id})...")
            from transformers import pipeline
            self.local_pipeline = pipeline(
                "text-generation", 
                model=model_id, 
                torch_dtype="auto", 
                device_map="auto"
            )

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
        """
        context = self._format_context(context_chunks)
        
        # Determine actual backend
        use_openai = (backend == "openai") and (self.openai_client is not None)
        
        if backend == "openai" and self.openai_client is None:
            return "Error: OpenAI backend selected but OPENAI_API_KEY not found. Please switch to Local or set key."

        if use_openai:
            # OpenAI Logic
            input_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            if "gpt-5-" in self.openai_model or "nano" in self.openai_model:
                 try:
                     response = self.openai_client.responses.create(
                        model=self.openai_model,
                        input=input_messages,
                        reasoning={"effort": "medium"},
                        text={"verbosity": "medium"},
                        max_output_tokens=1000
                     )
                     return response.output_text
                 except Exception as e:
                     return f"OpenAI Error: {str(e)}"
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=input_messages,
                    temperature=0.1
                )
                return response.choices[0].message.content
        else:
            # Local Logic (Mistral / ZeroGPU)
            self._ensure_local_loaded()
            
            # Mistral expects standard chat messages usually
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            # ZeroGPU / Spaces Handling
            def run_pipeline(msgs):
                # Modern pipelines handle list of messages automatically applying chat template
                outputs = self.local_pipeline(
                    msgs,
                    max_new_tokens=512, # Increased for Mistral
                    do_sample=True, 
                    temperature=0.1, 
                    top_k=50, 
                    top_p=0.95
                )
                # Output is usually a list of dicts. 
                # For chat pipeline: [{'generated_text': [..., {'role': 'assistant', 'content': '...'}]}] 
                # OR just string if text-generation vs text-generation-chat?
                # Default pipeline "text-generation" with list input usually returns: 
                # [{'generated_text': [{'role': 'user', ...}, {'role': 'assistant', 'content': 'Response'}]}]
                return outputs[0]['generated_text']

            # Try to decorate with spaces.GPU
            try:
                import spaces
                print("ZeroGPU enabled for this generation.")
                run_pipeline = spaces.GPU(run_pipeline)
            except ImportError:
                pass
            except Exception as e:
                print(f"Could not use ZeroGPU: {e}")

            result = run_pipeline(messages)
            
            # Parse result (Transformers pipeline behavior varies by version/call)
            # If result is a string (rare for chat list input), return it.
            # If result is a list of messages (standard for chat), extract last content.
            if isinstance(result, list):
                 # Check if it's the full conversation
                 last_msg = result[-1]
                 if last_msg.get('role') == 'assistant':
                     return last_msg['content']
                 else:
                     # Fallback
                     return str(result)
            return str(result)

_shared_generator = None

def get_generator():
    global _shared_generator
    if _shared_generator is None:
        _shared_generator = GeneratorService()
    return _shared_generator
