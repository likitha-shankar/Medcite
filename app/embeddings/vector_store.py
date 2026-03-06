"""
Vector store: ChromaDB wrapper for storing and searching embedded chunks.

ChromaDB gives us three things:
1. STORAGE: Persist embedded vectors and their metadata to disk so they survive
   restarts. Without persistence, you'd re-embed everything every time you start
   the app — expensive and slow.

2. INDEXING: ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs under
   the hood. Think of it as a smart filing system: instead of comparing your query
   against every single stored vector (O(n) — impossibly slow for millions of chunks),
   HNSW builds a graph where similar vectors are connected, letting us find nearest
   neighbors in roughly O(log n) time.

3. METADATA FILTERING: When you search, you can say "only search chunks from
   document X" or "only search chunks from section 'Dosage'". The filter is
   applied BEFORE the vector search, so you're not wasting compute comparing
   against irrelevant chunks. In healthcare, this is critical — if a doctor
   uploads 50 guidelines and asks about one drug, we shouldn't search all 50.
"""

from pathlib import Path

import chromadb

from app.ingestion.models import DocumentChunk
from app.embeddings.embedder import embed_texts, embed_query

# Where ChromaDB persists its data on disk
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")

# Collection name — think of this as a "table" in a relational database.
# All our healthcare chunks go into one collection.
COLLECTION_NAME = "healthcare_chunks"

# Module-level cache for the ChromaDB client and collection
_client = None
_collection = None


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection.

    get_or_create_collection is idempotent — if the collection exists, it
    returns it. If not, it creates it. This means we can call this function
    freely without worrying about duplicates or errors.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            # cosine similarity is standard for sentence embeddings.
            # It measures the angle between vectors, ignoring magnitude.
            # "heart attack treatment" and "treatment for heart attack"
            # point in roughly the same direction → high cosine similarity.
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collection '{COLLECTION_NAME}' ready. "
              f"Contains {_collection.count()} vectors.")
    return _collection


def add_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed chunks and add them to the vector store.

    Each chunk gets stored with:
    - its embedding vector (for similarity search)
    - its text (so we can return it in results without a separate lookup)
    - its metadata (for filtering and citations)

    We process in batches of 100 to avoid memory issues with large documents
    and to show progress.
    """
    collection = get_collection()
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        ids = [chunk.chunk_id for chunk in batch]
        texts = [chunk.text for chunk in batch]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in batch
        ]

        # Embed the batch
        embeddings = embed_texts(texts)

        # Upsert = insert or update. If a chunk_id already exists (from a
        # previous ingestion of the same document), it gets updated rather
        # than duplicated. This is important for re-ingestion workflows.
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[VectorStore] Upserted batch {i // batch_size + 1} "
              f"({len(batch)} chunks)")

    print(f"[VectorStore] Total vectors in store: {collection.count()}")


def search(
    query: str,
    top_k: int = 10,
    where: dict | None = None,
) -> list[dict]:
    """Search for the most relevant chunks given a query.

    Args:
        query: The user's question in natural language
        top_k: How many results to return (we retrieve more than we need
                because Phase 4's reranker will filter down to the best ones)
        where: Optional metadata filter, e.g. {"source_file": "guidelines.pdf"}

    Returns:
        List of dicts with keys: chunk_id, text, metadata, distance
        Sorted by relevance (most relevant first).

    The 'distance' field is the cosine distance (0 = identical, 2 = opposite).
    Lower is better. In practice, anything below ~0.5 is usually relevant.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    # Build search kwargs — only include 'where' if we have a filter
    search_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),  # Can't request more than we have
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        search_kwargs["where"] = where

    results = collection.query(**search_kwargs)

    # ChromaDB returns nested lists (because you can query multiple things at once).
    # We only ever query one thing, so we unpack the first (and only) result set.
    formatted_results = []
    for i in range(len(results["ids"][0])):
        formatted_results.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return formatted_results


def reset_collection() -> None:
    """Delete all data from the collection. Use with caution.

    Useful during development when you want to re-ingest with different
    chunking settings.
    """
    global _collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection might not exist yet
    _collection = None
    print(f"[VectorStore] Collection '{COLLECTION_NAME}' reset.")
