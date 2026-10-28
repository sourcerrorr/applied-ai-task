from embeddings import embed_texts
from vectordb import client
from config import COLLECTION_NAME
import openai
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException 
from rank_bm25 import BM25Okapi
import re


def tokenize(text):

    return re.findall(r"\w+", text.lower())

def rrf(dense_ids, bm25_ids, k=60):
    dense_rank = {doc_id: i for i, doc_id in enumerate(dense_ids, start=1)}
    bm25_rank  = {doc_id: i for i, doc_id in enumerate(bm25_ids, start=1)}
    scores = {doc_id: 1/(k + dense_rank[doc_id]) + 1/(k + bm25_rank[doc_id]) for doc_id in dense_rank}
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_search(question, pool_k=100, k=5):
    qdrant_result = search_qdrant(question, pool_k)

    if qdrant_result is None:
        return None

    tokenized_docs = [tokenize(h.payload['title'] + " " + h.payload['text']) for h in qdrant_result]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(scores, qdrant_result), key=lambda h: h[0], reverse=True)

    dense_ids = [h.id for h in qdrant_result]
    bm25_ids  = [h.id for score, h in ranked]

    fused_ids = rrf(dense_ids, bm25_ids)

    id_to_hit = {h.id: h for h in qdrant_result}
    
    return [id_to_hit[doc_id] for doc_id in fused_ids[:k]]


def main():

    question = input("Enter question: ")
    hits = hybrid_search(question)
    for hit in hits:
        print(f"Title: {hit.payload['title']}")


def embed_question(question: str):

    try:
        result = embed_texts([question])[0]
    except openai.OpenAIError as e: 
        print(f"[embed] {e}")
        return None
    return result

def search_qdrant(question, k=5):
    query_vector = embed_question(question)
    if query_vector is None:
        return None
    else:
        try:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query = query_vector,
                limit = k,
                with_payload = True
              )
            result = response.points
        except (ApiException, ResponseHandlingException) as e:
            print(f"[qdrant] {e}")
            return None
    return result

    

if __name__ == "__main__":
    main()