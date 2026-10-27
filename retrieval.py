from embeddings import embed_texts
from vectordb import client
from config import COLLECTION_NAME
import openai
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException 

def main():
    
    question = input("Enter question: ")
    result = search_qdrant(question, 5)
    for hit in result:
        print("Score: ", hit.score )
        print("Title: ", hit.payload["title"])



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