import pytest
from retrieval import rrf, tokenize
from generate import build_context
from eval_test import recall_at_k, reciprocal_rank, token_f1, citation_accuracy
from usage import record, get_totals, _totals
from documents import chunk_text


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


# chunk_text — fixed strategy

def test_fixed_splits_long_text():
    """Text longer than CHUNK_SIZE (500) should produce multiple chunks."""
    text = "a" * 1000
    chunks = chunk_text(text, "fixed")
    assert len(chunks) > 1

def test_fixed_short_text_single_chunk():
    """Text shorter than CHUNK_SIZE stays as one chunk."""
    text = "a" * 30
    chunks = chunk_text(text, "fixed")
    assert len(chunks) == 1

def test_fixed_filters_tiny_remainder():
    """A trailing fragment shorter than MIN_CHUNK_LEN (20) is dropped."""
    text = "a" * 510
    chunks = chunk_text(text, "fixed")
    for chunk in chunks:
        assert len(chunk) >= 20

def test_fixed_overlap():
    """Consecutive chunks overlap — the end of chunk 1 matches the start of chunk 2."""
    text = "abcdefghij" * 100  # 1000 chars of repeating pattern
    chunks = chunk_text(text, "fixed")
    assert len(chunks) >= 2
    chunk1_tail = chunks[0][-100:]  # last 100 chars (CHUNK_OVERLAP)
    chunk2_head = chunks[1][:100:]  # first 100 chars
    assert chunk1_tail == chunk2_head


# chunk_text — paragraph strategy

def test_paragraph_splits_on_double_newline():
    """Text with \\n\\n paragraph breaks splits at those boundaries."""
    text = ("First paragraph with enough text." + "\n\n" +
            "Second paragraph with enough text." + "\n\n" +
            "Third paragraph with enough text.")
    chunks = chunk_text(text, "paragraph")
    assert len(chunks) == 3

def test_paragraph_fallback_no_breaks():
    """Without \\n\\n, paragraph strategy falls back to fixed-size chunking."""
    text = "a" * 1000  # no paragraph breaks
    chunks = chunk_text(text, "paragraph")
    assert len(chunks) > 1  # fell back to fixed, split the text

def test_paragraph_fallback_oversized():
    """If any paragraph exceeds CHUNK_SIZE * 2 (1000), falls back to fixed."""
    text = ("Short paragraph here." + "\n\n" +
            "x" * 1200)  # one paragraph way too long
    chunks = chunk_text(text, "paragraph")
    # fallback to fixed means chunks are ~500 chars, not one 1200-char chunk
    assert all(len(c) <= 500 for c in chunks)

def test_paragraph_filters_tiny():
    """Paragraphs shorter than MIN_CHUNK_LEN (20) are dropped."""
    text = ("ok" + "\n\n" +
            "This paragraph is long enough to keep.")
    chunks = chunk_text(text, "paragraph")
    assert len(chunks) == 1
    assert "long enough" in chunks[0]

def test_crlf_normalized():
    """Windows \\r\\n line endings are converted to \\n before chunking."""
    text = ("First paragraph here." + "\r\n\r\n" +
            "Second paragraph here.")
    chunks = chunk_text(text, "paragraph")
    assert len(chunks) == 2


# token_f1

def test_token_f1_perfect():
    """Identical text gives F1 = 1.0."""
    assert token_f1("the cat sat", "the cat sat") == 1.0

def test_token_f1_partial():
    """Partial overlap gives a score between 0 and 1."""
    score = token_f1("the cat sat on mat", "the cat")
    assert 0 < score < 1

def test_token_f1_no_overlap():
    """No shared tokens gives 0."""
    assert token_f1("hello world", "foo bar") == 0.0


# citation_accuracy

def test_citation_accuracy_all_correct():
    """All cited sources are in gold titles."""
    hits = [FakeHit(1, "Doc A", "text"), FakeHit(2, "Doc B", "text")]
    assert citation_accuracy("[1] and [2]", hits, {"Doc A", "Doc B"}) == 1.0

def test_citation_accuracy_half_correct():
    """One of two citations is correct."""
    hits = [FakeHit(1, "Doc A", "text"), FakeHit(2, "Doc B", "text")]
    assert citation_accuracy("[1] and [2]", hits, {"Doc A"}) == 0.5

def test_citation_accuracy_no_citations():
    """No [n] citations in the answer gives 0."""
    hits = [FakeHit(1, "Doc A", "text")]
    assert citation_accuracy("No citations here", hits, {"Doc A"}) == 0.0