import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gradio as gr
import shutil
from services.rag.retrieve import get_retriever
from services.rag.rerank import get_reranker
from services.rag.generate import get_generator
from services.rag.ingest import ingest
from services.rag.index import build_index
from services.observability.langfuse_client import observe

# Constants
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(DATA_DIR, "index")
SAMPLE_DOCS_DIR = "sample_docs"

# Global Singletons
retriever = None
reranker = None
generator = None

def init_services():
    global retriever, reranker, generator
    try:
        if os.path.exists(INDEX_DIR):
             retriever = get_retriever(INDEX_DIR)
        reranker = get_reranker()
        generator = get_generator()
    except Exception as e:
        print(f"Service init warning: {e}")

# GPU-wrapped generation function
@spaces.GPU
def generate_response_gpu(message, history, backend):
    # This function runs on ZeroGPU
    return generator.generate(message, "", backend=backend)

@observe(name="chat_interaction")
def chat_fn(message, history, backend):
    global retriever, reranker, generator
    
    if retriever is None:
        # Try to reload if index was just built
        init_services()
        if retriever is None:
            return "System is not ready. Please go to Admin tab and ingest documents or check logs."
    
    # 0. Contextualize Query (Simple)
    full_query = message

    import time
    start_time = time.time()

    # 1. Retrieve
    retrieved = retriever.retrieve(full_query, top_k=10)
    if not retrieved:
        return "No relevant documents found in index."
        
    # 2. Rerank
    reranked = reranker.rerank(full_query, retrieved, top_k=5)
    
    # 3. Generate
    answer = generator.generate(full_query, reranked, backend=backend)
    
    # 4. Format Output with Evidence
    elapsed = time.time() - start_time
    
    # Build Sources Text
    sources_text = "\n\n### Evidence\n"
    for i, chunk in enumerate(reranked):
        meta = chunk.get('metadata', {})
        score = chunk.get('rerank_score', chunk.get('score', 0))
        sources_text += f"**[{i+1}] {meta.get('doc_id')}** (Score: {score:.2f})\n"
        snippet = chunk['content'][:150].replace('\n', ' ')
        sources_text += f"> ...{snippet}...\n\n"
        
    final_response = f"{answer}\n\n{sources_text}\n*(Backend: {backend} | Time: {elapsed:.2f}s)*"
    return final_response

def admin_ingest(files):
    # files is a list of file paths
    # Copy files to a temp ingest folder or directly process
    if not files:
        return "No files selected."
        
    # Create temp input dir
    temp_in = "temp_ingest"
    if os.path.exists(temp_in):
        shutil.rmtree(temp_in)
    os.makedirs(temp_in)
    
    for file in files:
        shutil.copy(file.name, temp_in)
        
    # Run Ingest
    status = "Starting ingestion...\n"
    yield status
    try:
        ingest(temp_in, PROCESSED_DIR)
        status += "Ingestion complete.\nBuilding Index...\n"
        yield status
        
        build_index(PROCESSED_DIR, INDEX_DIR)
        status += "Index built successfully.\nReloading services...\n"
        yield status
        
        init_services()
        status += "Services reloaded. Ready to chat."
    except Exception as e:
        status += f"Error: {e}"
        
    return status

# Initialize on module load
init_services()

with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# Production RAG Knowledge Assistant")
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            # Configuration Accordion
            with gr.Accordion("Settings", open=True):
                backend_radio = gr.Radio(
                    choices=["openai", "local"], 
                    value="openai", 
                    label="LLM Backend"
                )
            
            chatbot = gr.ChatInterface(
                fn=chat_fn, 
                additional_inputs=[backend_radio],
                title="Ask me anything about your documents"
            )
            
        with gr.Tab("Admin"):
            gr.Markdown("## Document Management")
            file_upload = gr.File(label="Upload Documents (PDF, HTML, TXT)", file_count="multiple")
            ingest_btn = gr.Button("Ingest & Re-Index")
            status_box = gr.Textbox(label="Status", interactive=False)
            
            ingest_btn.click(admin_ingest, inputs=[file_upload], outputs=[status_box])

if __name__ == "__main__":
    demo.queue().launch()
