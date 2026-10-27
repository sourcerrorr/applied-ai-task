from embeddings import embed_texts
from vectordb import client
from config import COLLECTION_NAME


def main():
    
    question = input("Enter question: ")
    result = search_qdrant(question, 5)
    for hit in result:
        print("Score: ", hit.score )
        print("Title: ", hit.payload["title"])



def embed_question(question: str):
    result = embed_texts([question])[0]
    return result

def search_qdrant(question, k):
    query_vector = embed_question(question)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query = query_vector,
        limit = k,
        with_payload = True
    )
    result = response.points
    return result

    

if __name__ == "__main__":
    main()