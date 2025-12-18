import pytest
from services.rag.chunk import split_text, create_chunks
from eval.metrics import calculate_recall, calculate_mrr

def test_split_text():
    text = "Line 1\nLine 2\nLine 3"
    chunks = split_text(text, chunk_size=10, chunk_overlap=0)
    assert len(chunks) > 1

def test_metrics_recall():
    retrieved = ["doc1_chunk1", "doc2_chunk1"]
    gold = ["doc1"]
    assert calculate_recall(retrieved, gold) == 1.0
    
    gold_fail = ["doc3"]
    assert calculate_recall(retrieved, gold_fail) == 0.0

def test_metrics_mrr():
    retrieved = ["doc1_chunk1", "doc2_chunk1"]
    gold = ["doc2"]
    # doc2 is at index 1 (rank 2). MRR = 1/2 = 0.5
    assert calculate_mrr(retrieved, gold) == 0.5
