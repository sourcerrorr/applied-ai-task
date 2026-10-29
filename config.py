""" Load environment variables and define project wide settings.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # read .env into the environment 

# External services 
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Models
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # dimensionality returned by EMBED_MODEL; must match the Qdrant collection
CHAT_MODEL = "gpt-6-luna"

# Vector store 
COLLECTION_NAME = "hotpotqa_paragraphs"

# Corpus cache 
CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")

# Batch sizes for the one-time ingestion (tune in ingest.py)
EMBED_BATCH = 100   # paragraphs per OpenAI embeddings request
                    # and points per Qdrant upsert request

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 20
MAX_TOOL_ROUNDS = 3
DENSE_POOL_K = 100
RRF_K = 60

PAR_COLLECTION = "hotpotqa_paragraphs"
FIXED_COLLECTION = "hotpotqa_fixed_chunks"
