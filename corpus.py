"""Build and cache the document corpus.

A "document" is one Wikipedia paragraph (a title plus its sentences). The corpus is the
deduplicated union of all context paragraphs in the HotpotQA distractor/validation split.
"""
import json
import os

from datasets import load_dataset

from config import CORPUS_PATH


def build_corpus():
    """Stream HotpotQA and pool its paragraphs into {title: text}, deduped by title.

    Dedup-by-title is automatic: assigning corpus[title] = text overwrites duplicates. This
    is safe because the only titles that appeared with differing text differed solely by
    non-breaking spaces, which the whitespace normalization below collapses.
    """
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    corpus = {}
    for item in data:
        context = item["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            text = " ".join(sentences)      # merge the sentence fragments into one paragraph
            text = " ".join(text.split())  # normalize whitespace (\xa0 -> space, collapse runs)
            corpus[title] = text
    return corpus


def load_corpus():
    """Return the corpus from disk if cached, else build it once and save it.

    corpus.json is a build-time cache only: the deployed app reads title+text from the
    Qdrant payloads, not from this file.
    """
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    corpus = build_corpus()
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    return corpus
