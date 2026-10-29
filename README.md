# AI Investigation Engine

A RAG (Retrieval-Augmented Generation) system that searches a large corpus of 66,000+ documents, combines facts across multiple documents, cites evidence, detects contradictions, and refuses when information is missing.

**Demo:** https://appliedaitask.streamlit.app

## Architecture

```
User question
    │
    ▼
┌─────────────────────────────────┐
│  Tool-calling loop (max 3 rounds)│
│                                  │
│  1. LLM decides what to search   │
│  2. Embed query (OpenAI)         │
│  3. Dense search (Qdrant)        │
│  4. BM25 re-rank                 │
│  5. RRF fusion (custom ranker)   │
│  6. Return top-5 hits            │
│  7. LLM decides: answer or       │
│     search again?                │
└─────────────────────────────────┘
    │
    ▼
Answer with [citations] or refusal
```

The LLM controls the retrieval loop via tool calling. It decides what to search, reviews the results, and can search again with a more specific query. This solves bridge-entity questions that single-shot retrieval cannot — for example, finding a person mentioned in a film article, then searching for that person separately.

## Features

- **Multi-hop reasoning** — tool-calling loop lets the LLM search iteratively across documents
- **Hybrid search** — dense (semantic) retrieval + BM25 keyword re-ranking, fused with a custom RRF ranker
- **Citations** — every claim is cited by source number (e.g. [1][3])
- **Refusal** — refuses to answer when retrieved documents don't contain the information, even if the LLM knows the answer
- **Contradiction detection (experimental)** — attempts to flag conflicting information when contradictory documents are co-retrieved. Works when both sources appear in the same search window; limited by retrieval co-occurrence (see Known Limitations)
- **Document management** — upload, view, and delete documents (TXT, PDF, DOCX)
- **Two chunking strategies** — paragraph-based and fixed-size (500 chars, 100 overlap), compared in evaluation
- **Content deduplication** — MD5 hash prevents duplicate documents
- **Usage tracking** — per-query and total token counts displayed in the UI
- **Automated evaluation** — 40-question test suite with 5 metrics (Recall@K, MRR, Answer Correctness, Citation Accuracy, LLM-as-judge), comparing 4 configurations

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Dataset | HotpotQA (66,581 paragraphs) | Multi-hop QA with ground-truth supporting facts for evaluation |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap (~$0.16 for entire corpus), 1536-dim, good quality |
| LLM | OpenAI `gpt-6-luna` | Successor to gpt-4o-mini, cheaper, supports tool calling with `reasoning_effort="none"` |
| Vector DB | Qdrant Cloud | Free tier, persistent, server-side search |
| Search | Custom RRF hybrid ranker | Dense retrieval + BM25 re-ranking fused with Reciprocal Rank Fusion |
| UI | Streamlit | Python-only, fast to build, sufficient for the required functionality |
| Hosting | Streamlit Community Cloud | Free, native Streamlit support, built-in secrets management |

## Setup

### Prerequisites
- Python 3.13+
- OpenAI API key (Build tier or higher)
- Qdrant Cloud cluster

### Installation

```bash
git clone https://github.com/sourcerrorr/applied-ai-task.git
cd applied-ai-task
python -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Run locally

```bash
streamlit run app.py
```

### Ingest the corpus (one-time)

The Qdrant Cloud cluster already contains the ingested data. To re-ingest from scratch:

```bash
python ingest.py          # paragraph chunking → hotpotqa_paragraphs collection
python ingest_fixed.py    # fixed-size chunking → hotpotqa_fixed_chunks collection
python synthetic.py       # inject contradictory documents
```

### Run tests

```bash
pytest tests_unit.py
```

### Run evaluation

```bash
python eval_test.py           # paragraph collection (hybrid + dense)
python eval_test.py fixed     # fixed-size collection (hybrid + dense)
```

## Design Decisions

**Paragraph chunking as default** — splits on `\n\n` boundaries, keeping natural text units intact. Falls back to fixed-size chunking when paragraphs are missing or oversized. Better citation accuracy (1.000 vs 0.970) than fixed-size because chunks align with source boundaries.

**Hybrid search (retrieve-then-rerank)** — dense search retrieves a large pool (top-100), BM25 re-ranks them, RRF fuses both rankings. Needs no re-ingestion and is fully transparent. Limitation: BM25 can only re-order dense candidates, not add new ones.

**RRF as the fusion algorithm** — `score = 1/(k + dense_rank) + 1/(k + bm25_rank)` for each document, then sort by score. 

**Tool calling over single-shot retrieval** — the LLM decides what to search and when to stop. This solves bridge-entity multi-hop questions where you need result A to know what to search for in query B.

**Knowledge suppression in system prompt** — the LLM is instructed to refuse even when it knows the answer, if the retrieved documents don't contain the information. 

**Stateless hosting** — the app is a stateless reader. Qdrant Cloud is the durable data layer. The host can restart, sleep, or redeploy without data loss.

**No FastAPI/REST layer** — the Streamlit app calls module functions directly.

## Evaluation Summary

40 questions: 28 multi-hop (HotpotQA), 6 no-answer, 6 contradictory. 5 metrics across 4 configurations (paragraph/fixed chunking × hybrid/dense search).

|  | Paragraph + Hybrid | Paragraph + Dense | Fixed + Hybrid | Fixed + Dense |
|---|---|---|---|---|
| Recall@5 | 0.809 | 0.838 | — | — |
| MRR | 0.818 | 0.780 | — | — |
| Correctness | 0.900 | 0.900 | — | — |
| Citation Acc | 0.850 | 0.829 | — | — |

See [EVALUATION.md](EVALUATION.md) for full results, analysis, and failure cases.

**Key finding:** Recall@5 of 0.768 but Correctness of 1.000 for multi-hop questions proves the tool-calling loop recovers documents that single-shot retrieval misses.

## Known Limitations

- **Contradiction detection (50% accuracy)** — the LLM gets a confident answer in round 1 of tool calling and stops before retrieving the contradictory document.
- **No-answer (83.3%)** — 5/6 correctly refused, but one question still gets answered from the LLM's prior knowledge despite the system prompt.
- **Dense recall > hybrid recall** — BM25 only re-ranks dense candidates. If BM25 promotes an irrelevant document into top-5 and demotes a relevant one, recall drops. 
- **Latency** — multi-step tool calling adds ~4s per round (sequential API calls to OpenAI and Qdrant Cloud). Typical query takes 5-13 seconds.
- **Free-tier sleep** — Streamlit Community Cloud sleeps after ~12h of inactivity.
- **Usage tracking resets on restart** — token counts are in-memory.

## Deployment Limits

| Service | Free Tier Limits |
|---|---|
| Streamlit Community Cloud | 1 GB RAM, app sleeps after 12h inactivity, US-hosted |
| Qdrant Cloud | 1 GB RAM, 0.5 vCPU, 4 GB disk storage, 1 cluster |

These limits do not block core feature demos. 
Although, latency is affected as Streamlit is US-hosted while Qdrant is EU-hosted.
After 12h of inactivity, the first request takes ~30 seconds while the app wakes up.

## Project Structure

```
├── app.py              # Streamlit UI
├── config.py           # Settings and constants
├── corpus.py           # HotpotQA corpus loading and caching
├── embeddings.py       # OpenAI embeddings client
├── vectordb.py         # Qdrant client and collection setup
├── retrieval.py        # Search: dense, BM25, hybrid, RRF fusion
├── generate.py         # RAG answer: tool-calling loop, system prompt
├── documents.py        # Document processing: extract, chunk, CRUD
├── usage.py            # In-memory token usage tracker
├── synthetic.py        # Inject contradictory test documents
├── ingest.py           # One-time corpus ingestion (paragraph chunking)
├── ingest_fixed.py     # One-time corpus ingestion (fixed-size chunking)
├── eval_test.py        # 40-question evaluation harness
├── tests_unit.py       # Unit tests (31 tests)
├── estimate_cost.py    # One-time embedding cost estimator
├── EVALUATION.md       # Evaluation report
├── AI_USAGE.md         # AI usage report
└── requirements.txt    # Python dependencies
```
