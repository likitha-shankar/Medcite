"""
Embedding module: converts text chunks into numerical vectors.

HOW EMBEDDINGS WORK (the mental model):
Imagine a massive coordinate system with 384 dimensions (not just x/y/z, but
384 axes). The embedding model reads a piece of text and places it at a specific
point in this space. Texts about similar topics land near each other. Texts about
unrelated topics land far apart.

When a user asks a question, we embed the question into the same space and find
the stored chunks that are closest to it. "Closest" = most semantically similar.

WHY MODEL CHOICE MATTERS:
The model determines WHAT "similarity" means. A model trained on medical literature
will understand that "myocardial infarction" and "heart attack" should be close
together, while a generic model might not place them as near. We're using
all-MiniLM-L6-v2 (general-purpose, fast, 384 dimensions) — a good starting point.
In production healthcare, you'd upgrade to something like BiomedBERT or MedCPT.
We compensate for this limitation with hybrid search (Phase 3) and reranking (Phase 4).
"""

from sentence_transformers import SentenceTransformer

# Module-level cache: load the model once, reuse across calls.
# Model loading is expensive (~2-3 seconds). We don't want to reload
# it every time we embed a chunk or a query.
_model: SentenceTransformer | None = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # This model outputs 384-dimensional vectors


def get_model() -> SentenceTransformer:
    """Load the embedding model (cached after first call).

    Why cache? Loading a transformer model means reading ~80MB of weights
    from disk into memory and setting up GPU/CPU tensors. Doing this once
    at startup and reusing is standard practice.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[Embedder] Model loaded. Dimension: {EMBEDDING_DIMENSION}")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed (chunks or queries)

    Returns:
        List of vectors, each a list of 384 floats.

    Why batch? Embedding models process batches much faster than individual
    texts because of GPU parallelism and memory optimization. Even on CPU,
    batching reduces Python overhead from the model framework.
    """
    model = get_model()
    # show_progress_bar is helpful during bulk ingestion so you see progress
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
    # Convert numpy arrays to plain Python lists for JSON serialization
    # and ChromaDB compatibility
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string.

    We separate this from embed_texts for clarity — queries and documents
    are embedded the same way with this model, but some models (like E5)
    require different prefixes for queries vs. documents. Having a separate
    function makes it easy to add that later.
    """
    return embed_texts([query])[0]
