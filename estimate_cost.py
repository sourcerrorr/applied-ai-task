"""One-off script: estimate the cost of embedding the whole corpus with the OpenAI text-embedding-3-small model."""
import tiktoken

from config import EMBED_MODEL
from corpus import load_corpus

RATE_PER_1M_TOKENS = 0.02  # USD for text-embedding-3-small


def main():
    corpus = load_corpus()
    encoding = tiktoken.encoding_for_model(EMBED_MODEL) 
    total_tokens = sum(len(encoding.encode(text)) for text in corpus.values())
    cost = total_tokens / 1_000_000 * RATE_PER_1M_TOKENS
    print(f"Paragraphs:     {len(corpus)}")
    print(f"Total tokens:   {total_tokens}")
    print(f"Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
