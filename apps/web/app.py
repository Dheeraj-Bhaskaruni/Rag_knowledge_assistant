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



def clear_knowledge_base():
    # Helper to wipe data
    for d in [PROCESSED_DIR, INDEX_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    
    # Reset helper
    import services.rag.retrieve
    services.rag.retrieve._shared_retriever = None
    init_services()
    
    return "Knowledge Base Cleared. System is empty."

def admin_ingest(files, use_sample):
    # 1. Clean Temp Input ONLY (Keep Processed/Index for additive)
    temp_in = "temp_ingest"
    if os.path.exists(temp_in):
        shutil.rmtree(temp_in)
    os.makedirs(temp_in)
    
    # Ensure processed/index dirs exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    status = "Starting processing...\n"
    
    # Handle Source Selection
    files_found = False
    
    if use_sample:
        # Copy from samples dir
        sample_file = os.path.join(SAMPLES_DIR, "sports_legends.txt")
        if os.path.exists(sample_file):
            shutil.copy(sample_file, temp_in)
            status += f"Loaded: Sports Legends Dataset\n"
            files_found = True
        else:
            return "Error: Sample data not found on server."
            
    if files:
        # Copy uploaded files
        for file in files:
            shutil.copy(file.name, temp_in)
        status += f"Loaded: {len(files)} new files.\n"
        files_found = True
    
    if not files_found:
        return "No new files selected. Select files or sample data."
        
    yield status
    
    # Run Ingest
    try:
        # Ingest new files to PROCESSED_DIR (Additive)
        ingest(temp_in, PROCESSED_DIR)
        status += "Processing new files complete.\nRebuilding Index...\n"
        yield status
        
        # Build Index (scans ALL files in PROCESSED_DIR)
        build_index(PROCESSED_DIR, INDEX_DIR)
        status += "Index rebuilt with all documents.\nReloading services...\n"
        yield status
        
        # FORCE RELOAD: Clear singletons
        import services.rag.retrieve
        services.rag.retrieve._shared_retriever = None
        
        init_services()
        status += "Services reloaded. Knowledge Base Updated successfully!"
    except Exception as e:
        print(f"Ingestion Failed: {e}") # Print to server logs
        import traceback
        traceback.print_exc()
        status += f"Error: {e}"
        
    return status

# Initialize on module load
init_services()

with gr.Blocks(title="RAG Knowledge Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Knowledge Assistant")
    
    with gr.Row():

        # Left Column: Sidebar (Controls & Guide)
        with gr.Column(scale=1, variant="panel"):
            with gr.Group():
                file_upload = gr.File(
                    label="Document Upload", 
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".html"],
                    height=70
                )
                
                gr.HTML("<div style='text-align: center; color: #666; font-size: 11px; margin: 2px 0;'>— OR —</div>")
                
                use_sample_chk = gr.Checkbox(
                    label="Use Sports Legends Sample Dataset", 
                    container=False
                )
                
                ingest_btn = gr.Button("Process Documents", variant="primary", size="sm")
                clear_btn = gr.Button("Clear Knowledge Base", variant="stop", size="sm")
            
            # Status Log - Visible by default
            with gr.Accordion("System Logs", open=True):
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
            
            clear_btn.click(
                clear_knowledge_base,
                outputs=[status_box]
            )
            
            with gr.Group():
                backend_radio = gr.Radio(
                    choices=["openai", "gemini", "local"], 
                    value="openai", 
                    label="LLM Backend",
                    container=False
                )
                gr.HTML("<div style='font-size: 9px; color: #888; margin-top: 2px;'>*Local = Mistral-7B (ZeroGPU)</div>")

        # Right Column: Main App (Chat)
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_fn, 
                additional_inputs=[backend_radio],
                title="Chat Interface",
                description="Ask questions about your uploaded documents or the sample dataset.",
                examples=[
                    ["Who is the greatest quarterback?", "openai"], 
                    ["Summarize the uploaded documents", "local"],
                    ["What makes Lionel Messi a legend?", "gemini"]
                ]
            )

if __name__ == "__main__":
    # specific server_name needed for Docker/Spaces
    demo.queue().launch(server_name="0.0.0.0")
