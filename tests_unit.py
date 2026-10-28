import pytest
from retrieval import rrf, tokenize
from generate import build_context
from eval_test import recall_at_k, reciprocal_rank
from usage import record, get_totals, _totals


class FakeHit:
    """Mimics a Qdrant hit object with .id and .payload"""
    def __init__(self, id, title, text):
        self.id = id
        self.payload = {"title": title, "text": text}


class FakeUsage:
    """Mimics OpenAI response.usage"""
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


# rrf
def test_rrf_top_in_both_wins():
    result = rrf([10, 20, 30], [10, 30, 20], k=60)
    assert result[0] == 10

def test_rrf_returns_all_docs():
    result = rrf([1, 2, 3], [3, 2, 1], k=60)
    assert set(result) == {1, 2, 3}

def test_rrf_single_doc():
    result = rrf([5], [5], k=60)
    assert result == [5]

def test_rrf_symmetric_swap():
    result = rrf([10, 20, 30], [30, 20, 10], k=60)
    assert result[-1] == 20


# tokenize

def test_tokenize_lowercases():
    assert tokenize("Hello World") == ["hello", "world"]

def test_tokenize_strips_punctuation():
    assert tokenize("it's a test!") == ["it", "s", "a", "test"]


# build_context

def test_build_context_numbering():
    hits = [FakeHit(1, "Title A", "Text A"), FakeHit(2, "Title B", "Text B")]
    result = build_context(hits, 1)
    assert "[1] Title: Title A" in result
    assert "[2] Title: Title B" in result

def test_build_context_offset():
    hits = [FakeHit(1, "Title A", "Text A")]
    result = build_context(hits, 6)
    assert "[6] Title: Title A" in result

def test_build_context_empty():
    assert build_context([], 1) == ""


# recall_at_k 

def test_recall_perfect():
    assert recall_at_k({"A", "B"}, ["A", "B", "C"]) == 1.0

def test_recall_partial():
    assert recall_at_k({"A", "B"}, ["A", "C", "D"]) == 0.5

def test_recall_none():
    assert recall_at_k({"A", "B"}, ["C", "D"]) == 0.0


# reciprocal_rank 

def test_rr_first():
    assert reciprocal_rank({"A"}, ["A", "B", "C"]) == 1.0

def test_rr_second():
    assert reciprocal_rank({"A"}, ["B", "A", "C"]) == 0.5

def test_rr_missing():
    assert reciprocal_rank({"A"}, ["B", "C", "D"]) == 0.0


# usage tracking 

def test_record_and_totals():
    _totals["input_tokens"] = 0
    _totals["output_tokens"] = 0
    _totals["query_n"] = 0
    record(FakeUsage(100, 50))
    totals = get_totals()
    assert totals["input_tokens"] == 100
    assert totals["output_tokens"] == 50
    assert totals["query_n"] == 1