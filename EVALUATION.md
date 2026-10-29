# Evaluation Report

This report evaluates the RAG pipeline across different configurations to identify strengths, weaknesses, and areas for improvement.

## Configurations

The evaluation script (eval_test.py) tests the pipeline across 4 configurations (2 retrieval methods x 2 chunking strategies):

- Hybrid search on paragraph chunks
- Hybrid search on fixed-size chunks (500 characters, 100 character overlap)
- Dense-only search on paragraph chunks
- Dense-only search on fixed-size chunks

## Evaluation Set

The evaluation set consists of 40 questions total:

- 28 multi-hop questions from the HotpotQA validation set, which require cross-document reasoning to answer.
- 6 contradictory questions constructed based on synthetic documents injected into both Qdrant collections. These synthetic docs contain information that contradicts the HotpotQA data (e.g., wrong nationalities, incorrect dates), testing whether the system flags conflicting sources instead of silently picking one.
- 6 no-answer questions about topics that do not exist in the knowledge base at all, testing whether the system refuses to answer rather than fabricating information.

Each question is labeled with a type: multi-hop, contradictory, or no-answer.

The main evaluation is run by the run_eval() function, which accepts a retriever type and collection name as parameters.

## Metrics

Retrieval metrics are measured using the HotpotQA gold titles (the supporting documents needed to answer each question):

- Recall@K (recall_at_k): the fraction of gold titles that appear in the top-K retrieved documents. Measures whether the system finds the right documents.
- MRR (reciprocal_rank): the reciprocal of the rank of the first relevant document. Measures how high the right documents are ranked.

Answer quality metrics are computed after the full RAG pipeline generates an answer:

- Correctness: determined by an LLM judge. For multi-hop questions, the generated answer and the gold answer from HotpotQA are passed to the LLM, which judges whether the predicted answer is correct based on meaning, not exact wording. For no-answer questions, a separate refusal_judge checks whether the system refused to answer. For contradictory questions, a contra_judge checks whether the system flagged a conflict between sources. In all cases the judge returns 1.0 or 0.0.
- Token F1 (token_f1): computed only for multi-hop questions (set to None for no-answer and contradictory types). Splits both the predicted and gold answers into word-level tokens, computes precision (common tokens / predicted tokens), recall (common tokens / gold tokens), and combines them as harmonic mean: 2 * precision * recall / (precision + recall).
- Citation Accuracy (citation_accuracy): finds [n]-style citations in the generated answer and checks whether the cited chunks correspond to HotpotQA gold titles. The score is correct citations / total citations. For no-answer questions this is always 0.0 because there are no supporting documents and the system should not cite any sources. Citation accuracy is measured only against HotpotQA gold titles; for user-uploaded documents there is no ground truth to evaluate citations against.

## Results

| Metric | Paragraph + Hybrid | Paragraph + Dense | Fixed + Hybrid | Fixed + Dense |
|---|---|---|---|---|
| Multi-hop Recall@5 | 0.768 | **0.804** | 0.786 | 0.750 |
| Multi-hop MRR | **0.954** | 0.918 | 0.929 | 0.923 |
| Multi-hop Correctness | 1.000 | 1.000 | 1.000 | 1.000 |
| Multi-hop Citation Acc | **1.000** | 0.982 | 0.970 | 0.970 |
| No-answer Correctness | 0.833 | 0.833 | **1.000** | **1.000** |
| Contradictory Correctness | 0.500 | 0.500 | 0.500 | 0.333 |
| Overall Recall@5 | 0.809 | **0.838** | 0.824 | 0.794 |
| Overall MRR | **0.818** | 0.780 | 0.800 | 0.783 |
| Overall Correctness | 0.900 | 0.900 | **0.925** | 0.900 |
| Overall Citation Acc | **0.850** | 0.829 | 0.829 | 0.829 |

## Analysis

Paragraph chunking performs better on citation accuracy (1.000 vs 0.970 for multi-hop). Natural paragraph boundaries keep each chunk semantically coherent, so the model cites more precisely.

Fixed chunking wins on no-answer refusal (1.000 vs 0.833). Possibly because fixed chunks are less "complete-looking," so the model is less tempted to treat partial matches as real answers. With paragraph chunking, 1 out of 6 no-answer questions received a fabricated answer instead of a refusal.

Contradiction detection is the weakest point across all configs (0.33-0.50). Importantly, this is a generation-level failure, not a retrieval one — recall is 1.0 for all contradictory questions, meaning the conflicting documents are being retrieved successfully. The LLM just doesn't flag the conflict. In 3 out of 6 cases (Shirley Temple, Corliss Archer, University of Kansas), the model answered confidently from one source without acknowledging the contradiction from the synthetic document. A possible improvement would be a dedicated contradiction-detection step before answer generation, or a stricter prompt that explicitly instructs the model to compare sources pairwise.

Hybrid vs Dense is close overall. Hybrid helps MRR (ranking quality), while dense sometimes edges ahead on raw recall. The BM25 re-ranking component seems to help with ranking the right documents higher rather than finding more of them.

Multi-hop correctness at 1.000 across all configs is a strong result, but 28 questions is a relatively small sample. This would likely drop with a larger evaluation set. The tool-calling loop helps here — in several cases where initial retrieval missed a gold document (e.g., Q15 about David Beckham/Alex Ferguson with recall=0.0), the model searched again with a more specific query and still produced a correct answer.

## Limitations

- The evaluation set is small (40 questions). A larger set would give more reliable metrics, especially for the contradiction and no-answer categories which only have 6 questions each.
- Citation accuracy is only measurable against HotpotQA gold titles and does not apply to user-uploaded documents.
- Latency was not formally benchmarked. Per-query response time was observed at a few seconds during evaluation but varies depending on the number of tool-calling rounds.
- The LLM judge itself can be inconsistent — using an LLM to evaluate LLM outputs introduces its own error margin.