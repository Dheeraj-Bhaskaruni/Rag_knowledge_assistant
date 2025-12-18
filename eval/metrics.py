from typing import List, Set

def calculate_recall(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    if not gold_ids:
        return 0.0
    
    hits = sum(1 for rid in retrieved_ids if any(gid in rid for gid in gold_ids))
    # Note: simple substring match for ID since chunk IDs might have suffix
    
    return hits / len(gold_ids)

def calculate_mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    if not gold_ids:
        return 0.0
        
    for i, rid in enumerate(retrieved_ids):
        if any(gid in rid for gid in gold_ids):
            return 1.0 / (i + 1)
            
    return 0.0

def exact_match(prediction: str, expected: str) -> bool:
    return prediction.strip().lower() == expected.strip().lower()
