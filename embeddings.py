"""Turn text into embedding vectors via the OpenAI embeddings API."""
from openai import OpenAI

from config import EMBED_MODEL

client = OpenAI()  


def embed_texts(texts):
    """Embed a list of strings and return their vectors, aligned to the input order.

    texts:   list[str]          -- the paragraphs (or queries) to embed.
    returns: list[list[float]]  -- one 1536-dim vector per input, in the same order.

    This function embeds whatever list it is handed; splitting the corpus into batches is
    the caller's job (see ingest.py), which keeps "call the API" separate from "orchestrate".
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
