from corpus import load_corpus
from embeddings import embed_texts
from vectordb import ensure_collection, upsert_points
from qdrant_client.models import PointStruct
from config import EMBED_BATCH

def main():
    ensure_collection()  # create the collection if it does not already exist
    corpus = load_corpus()
    titles = list(corpus.keys())
    texts = list(corpus.values())

    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        chunk_embed = embed_texts(chunk)
        points = []
        for j, vector in enumerate(chunk_embed):
            points.append(PointStruct(id=start + j, vector=vector, payload={"title": titles[start + j], 
                                                                              "text": texts[start + j]}))
        upsert_points(points) 
        print(f"embedded and stored {start+len(chunk)} / {len(texts)}") 


if __name__ == "__main__":
    main()