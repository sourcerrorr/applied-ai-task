from corpus import load_corpus
from embeddings import embed_texts
from documents import chunk_text
from vectordb import client
from qdrant_client.models import PointStruct
from config import EMBED_BATCH, FIXED_COLLECTION


def main():
    corpus = load_corpus()

    all_titles = []
    all_chunks = []
    for title, text in corpus.items():
        chunks = chunk_text(text, "fixed")
        for chunk in chunks:
            all_titles.append(title)
            all_chunks.append(chunk)

    print(f"Total chunks: {len(all_chunks)}")

    for start in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[start : start + EMBED_BATCH]
        vectors = embed_texts(batch)
        points = []
        for j, vector in enumerate(vectors):
            points.append(PointStruct(
                id=start + j,
                vector=vector,
                payload={
                    "source": "hotpotqa",
                    "title": all_titles[start + j],
                    "text": all_chunks[start + j]
                }
            ))
        client.upsert(collection_name=FIXED_COLLECTION, points=points)
        print(f"Embedded and stored {start + len(batch)} / {len(all_chunks)}")


if __name__ == "__main__":
    main()
