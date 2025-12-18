import os
import glob
import json
import argparse
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from bs4 import BeautifulSoup
from .chunk import create_chunks

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_html(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

def clean_text(text: str) -> str:
    # Basic cleaning
    return text.replace('\x00', '')

def process_file(filepath: str) -> Dict:
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.pdf':
        raw_text = load_pdf(filepath)
    elif ext in ['.html', '.htm']:
        raw_text = load_html(filepath)
    elif ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        print(f"Skipping unsupported file: {filename}")
        return None

    cleaned_text = clean_text(raw_text)
    
    doc_id = filename.replace(' ', '_')
    metadata = {
        "source": filename,
        "doc_id": doc_id,
        "created_at": str(os.path.getctime(filepath))
    }
    
    chunks = create_chunks(cleaned_text, metadata)
    
    return {
        "metadata": metadata,
        "chunks": [vars(c) for c in chunks] # Serialize Chunk objects
    }

def ingest(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    files = []
    for ext in ['*.pdf', '*.html', '*.txt', '*.md']:
        files.extend(glob.glob(os.path.join(input_dir, ext))) # Simple glob, non-recursive
        
    processed_count = 0
    all_chunks = []
    
    manifest = []

    for f in files:
        print(f"Processing {f}...")
        try:
            result = process_file(f)
            if result:
                # Save individual doc chunks? Or one big file?
                # User req: "Saved processed artifacts to /data/processed/ with a manifest.json"
                # "cleaned markdown per document"
                
                # We'll save the full text as markdown and the chunks structure
                out_name = result['metadata']['doc_id'] + ".json"
                out_path = os.path.join(output_dir, out_name)
                
                with open(out_path, 'w') as f_out:
                    json.dump(result, f_out, indent=2)
                
                manifest.append({
                    "doc_id": result['metadata']['doc_id'],
                    "path": out_path,
                    "chunk_count": len(result['chunks'])
                })
                processed_count += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Save manifest
    with open(os.path.join(output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Ingestion complete. Processed {processed_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    
    ingest(args.input, args.out)
