import os
import json
import faiss
import numpy as np
import argparse
import pickle
from typing import List, Dict
from .embed import get_embedder

def load_processed_data(processed_dir: str) -> List[Dict]:
    chunks = []
    # Always glob for all JSONs to support additive ingestion
    import glob
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    
    print(f"Found {len(json_files)} existing documents to index.")
    
    for f_path in json_files:
        if f_path.endswith("manifest.json"): continue
        try:
            with open(f_path, 'r') as f:
                doc_data = json.load(f)
                if 'chunks' in doc_data:
                    chunks.extend(doc_data['chunks'])
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
            
    return chunks

def build_index(processed_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading chunks...")
    chunks = load_processed_data(processed_dir)
    print(f"Loaded {len(chunks)} chunks.")
    
    if not chunks:
        print("No chunks found. Exiting.")
        return

    texts = [c['content'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    
    print("Generating embeddings...")
    embedder = get_embedder()
    embeddings = embedder.embed(texts)
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    print("Building FAISS index...")
    # Using simple IndexFlatP, or IVF if dataset large. 
    # For "local lightweight", FlatL2 is safest and exact.
    # Normalize for cosine similarity if using IP (Inner Product).
    # SentenceTransformers are usually cosine-sim optimized.
    
    # Normalize embeddings for Cosine Similarity with IndexFlatIP
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors.")
    
    # Save index
    faiss.write_index(index, os.path.join(output_dir, "vector.index"))
    
    # Save metadatas (chunks map)
    # We need to map index ID -> Chunk Metadata + Content
    with open(os.path.join(output_dir, "doc_store.pkl"), 'wb') as f:
        pickle.dump(chunks, f)
        
    print(f"Index saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, help="Path to processed data directory")
    parser.add_argument("--out", required=True, help="Output directory for index")
    args = parser.parse_args()
    
    build_index(args.processed, args.out)
