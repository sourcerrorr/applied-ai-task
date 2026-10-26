""" Load environment variables and define project wide settings.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # read .env into the environment (a no-op in deployment, where the host sets real env vars)

# External services 
# OPENAI_API_KEY is read directly from the environment by the OpenAI() client, so it is
# not re-exported here. Qdrant credentials must be passed explicitly, so we expose them.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Models
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # dimensionality returned by EMBED_MODEL; must match the Qdrant collection

# Vector store 
COLLECTION_NAME = "hotpotqa_paragraphs"

# Corpus cache 
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")

# Batch sizes for the one-time ingestion (tune in ingest.py)
EMBED_BATCH = 100   # paragraphs per OpenAI embeddings request
                    # and points per Qdrant upsert request
