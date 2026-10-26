"""Interaction with the Qdrant vector store: connection, collection setup, upserts."""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBED_DIM

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection():
    """Create the collection if it does not already exist.

    Cosine distance is used because text-embedding-3-small returns unit-length vectors, so
    similarity is the cosine of the angle between them. The vector size must match EMBED_DIM.
    """
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def upsert_points(points):
    """Write a batch of points into the collection.

    Generic on purpose: the caller builds each point's id, vector, and payload, so the
    point-id scheme stays an ingestion decision rather than a storage one.
    """
    client.upsert(collection_name=COLLECTION_NAME, points=points)
