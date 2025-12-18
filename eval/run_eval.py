import os
import sys
import json
import argparse
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.rag.retrieve import get_retriever
from services.rag.rerank import get_reranker
from services.rag.generate import get_generator
from eval.metrics import calculate_recall, calculate_mrr
from eval.judge import Judge

def load_dataset(path: str) -> List[Dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_eval(data_path: str, report_dir: str):
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset(data_path)
    
    # Init services
    retriever = get_retriever() # Assumes default data/index
    reranker = get_reranker()
    generator = get_generator()
    judge = Judge()
    
    results = []
    
    total_recall = 0
    total_mrr = 0
    total_grounding = 0
    total_correctness = 0
    count = 0
    
    print(f"Running eval on {len(dataset)} examples...")
    
    for item in dataset:
        qid = item.get('id')
        question = item['question']
        gold_sources = item.get('gold_sources', [])
        
        # 1. Retrieve
        retrieved = retriever.retrieve(question, top_k=10)
        retrieved_ids = [c['metadata']['doc_id'] for c in retrieved] # Check chunk_id or doc_id? metrics.py checks substring
        retrieved_ids_full = [c['metadata']['chunk_id'] for c in retrieved]

        # 2. Rerank
        reranked = reranker.rerank(question, retrieved, top_k=5)
        
        # 3. Generate
        answer = generator.generate(question, reranked)
        
        # 4. Compute Metrics
        recall = calculate_recall(retrieved_ids_full, gold_sources)
        mrr = calculate_mrr(retrieved_ids_full, gold_sources)
        
        # 5. Judge
        # Concatenate context for judge
        context_text = "\n".join([c['content'] for c in reranked])
        # Only run judge if we have an API Key, else skip
        if os.getenv("OPENAI_API_KEY"):
            eval_res = judge.evaluate(question, context_text, answer)
        else:
            eval_res = {"grounding": 0, "correctness": 0, "reasoning": "No API Key"}
            
        result_entry = {
            "id": qid,
            "question": question,
            "answer": answer,
            "metrics": {
                "recall@10": recall,
                "mrr": mrr,
                "grounding": eval_res.get('grounding'),
                "correctness": eval_res.get('correctness')
            },
            "judge_reasoning": eval_res.get('reasoning')
        }
        results.append(result_entry)
        
        total_recall += recall
        total_mrr += mrr
        total_grounding += eval_res.get('grounding', 0)
        total_correctness += eval_res.get('correctness', 0)
        count += 1
        
        print(f"Eval {qid}: Recall={recall:.2f}, MRR={mrr:.2f}")

    # Aggregate
    if count > 0:
        avg_results = {
            "avg_recall@10": total_recall / count,
            "avg_mrr": total_mrr / count,
            "avg_grounding": total_grounding / count,
            "avg_correctness": total_correctness / count
        }
    else:
        avg_results = {}
        
    print("\nResults:", avg_results)
    
    # Save Report
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "eval_report.json"), 'w') as f:
        json.dump({"summary": avg_results, "details": results}, f, indent=2)
        
    with open(os.path.join(report_dir, "eval_report.md"), 'w') as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Summary\n")
        for k, v in avg_results.items():
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write("\n## Details\n")
        # Write top 5 failures? 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", default="reports")
    args = parser.parse_args()
    
    run_eval(args.data, args.report)
