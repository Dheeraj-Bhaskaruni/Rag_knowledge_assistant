import sys
import os
from dotenv import load_dotenv

load_dotenv()
# Force rebuild trigger

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gradio as gr
import shutil
import spaces
from services.rag.retrieve import get_retriever
from services.rag.rerank import get_reranker
from services.rag.generate import get_generator, run_local_generation
from services.rag.ingest import ingest
from services.rag.index import build_index
from services.observability.langfuse_client import observe
# Constants
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(DATA_DIR, "index")
SAMPLES_DIR = "samples"

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
def generate_response_gpu(message, context_chunks, backend):
    # Call the standalone function directly
    # Note: We must pass data, not the service instance
    return run_local_generation(message, context_chunks)

@observe(name="chat_interaction")
def chat_fn(message, history, backend):
    global retriever, reranker, generator
    
    if retriever is None:
        # Try to reload if index was just built
        init_services()
        if retriever is None:
            return "System is not ready. Please go to '1. Knowledge Base' tab and ingest documents."
    
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
    if backend == "local":
        answer = generate_response_gpu(full_query, reranked, backend)
    else:
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

def admin_ingest(files, use_sample):
    # Create temp input dir
    temp_in = "temp_ingest"
    if os.path.exists(temp_in):
        shutil.rmtree(temp_in)
    os.makedirs(temp_in)
    
    status = "Starting ingestion...\n"
    
    # Handle Source Selection
    if use_sample:
        # Copy from samples dir
        sample_file = os.path.join(SAMPLES_DIR, "sports_legends.txt")
        if os.path.exists(sample_file):
            shutil.copy(sample_file, temp_in)
            status += f"Loaded sample data: {sample_file}\n"
        else:
            return "Error: Sample data not found on server."
    elif files:
        # Copy uploaded files
        for file in files:
            shutil.copy(file.name, temp_in)
        status += f"Loaded {len(files)} uploaded files.\n"
    else:
        return "No files selected and 'Use Sample' not checked."
        
    yield status
    
    # Run Ingest
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
        print(f"Ingestion Failed: {e}") # Print to server logs
        import traceback
        traceback.print_exc()
        status += f"Error: {e}"
        
    return status

# Initialize on module load
init_services()

with gr.Blocks(title="RAG Knowledge Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Knowledge Assistant")
    
    with gr.Row():

        # Left Column: Sidebar (Controls & Guide)
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🎛️ Control Panel")
            
            with gr.Group():
                gr.Markdown("#### 1. Knowledge Base")
                file_upload = gr.File(
                    label="Upload Documents", 
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".html"],
                    height=100
                )
                
                gr.HTML("<div style='text-align: center; color: #666; font-size: 12px; margin: 5px 0;'>— OR —</div>")
                
                use_sample_chk = gr.Checkbox(
                    label="Load Sports Legends (Demo)", 
                    container=False
                )
                
                ingest_btn = gr.Button("🚀 Update Brain", variant="primary")
            
            # Status Log - Hidden by default to save space
            with gr.Accordion("📝 View Logs", open=False):
                status_box = gr.Textbox(
                    show_label=False, 
                    value="System Ready.", 
                    interactive=False, 
                    lines=4, 
                    max_lines=10,
                    text_align="left"
                )
            
            ingest_btn.click(
                admin_ingest, 
                inputs=[file_upload, use_sample_chk], 
                outputs=[status_box]
            )
            
            gr.Markdown("#### 2. Model Settings")
            with gr.Group():
                backend_radio = gr.Radio(
                    choices=["openai", "gemini", "local"], 
                    value="openai", 
                    label="Active Brain",
                    container=False
                )
                gr.HTML("<div style='font-size: 10px; color: #888; margin-top: 5px;'>*Local = ZeroGPU</div>")

        # Right Column: Main App (Chat)
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_fn, 
                additional_inputs=[backend_radio],
                title="💬 Talk to Me",
                description="I will answer based on the documents you taught me!",
                examples=[
                    ["Who is the greatest quarterback?", "openai"], 
                    ["Summary of Lionel Messi", "local"]
                ]
            )

if __name__ == "__main__":
    # specific server_name needed for Docker/Spaces
    demo.queue().launch(server_name="0.0.0.0")
