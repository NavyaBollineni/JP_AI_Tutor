
import json
from pathlib import Path

import faiss
import numpy as np

from embedding import get_embedding_model


# Store the vector database inside the backend folder.
VECTOR_DB_DIR = Path(__file__).resolve().parent / "vector_store"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def save_vector_store(embedded_result: dict) -> dict:
    """
    Save document embeddings and their text chunks
    into a persistent FAISS vector database.
    """

    if not isinstance(embedded_result, dict):
        raise ValueError("Input must be a dictionary.")

    if embedded_result.get("status") != "success":
        raise ValueError("Embeddings were not generated successfully.")

    chunks = embedded_result.get("chunks", [])

    if not chunks:
        raise ValueError("No embedded chunks are available.")

    document_id = embedded_result.get("document_id")

    if not document_id:
        raise ValueError("Document ID is missing.")

    # Get the embedding dimension from the first vector.
    first_vector = chunks[0].get("embedding")

    if not first_vector:
        raise ValueError("Chunk embeddings are missing.")

    dimension = len(first_vector)

    vectors = []
    metadata = []

    for chunk in chunks:
        vector = chunk.get("embedding")
        text = chunk.get("text", "")

        if not vector or not text.strip():
            raise ValueError("A chunk has missing text or embeddings.")

        if len(vector) != dimension:
            raise ValueError("Embedding dimensions do not match.")

        vectors.append(vector)

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "filename": embedded_result.get("filename"),
            "document_id": document_id
        })

    # Convert vectors into a NumPy matrix.
    vectors_array = np.asarray(vectors, dtype=np.float32)

    # Our embedding.py normalizes vectors, so inner product
    # can be used to calculate cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_array)

    # Create a separate folder for each uploaded document.
    document_dir = VECTOR_DB_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index.
    faiss.write_index(
        index,
        str(document_dir / "index.faiss")
    )

    # Save chunk text and metadata.
    with open(
        document_dir / "chunks.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {
        "status": "success",
        "document_id": document_id,
        "filename": embedded_result.get("filename"),
        "total_vectors": index.ntotal,
        "message": "Document vectors saved successfully."
    }


def search_similar_chunks(
    question: str,
    top_k: int = 4
) -> list:
    """
    Find the most relevant chunks across saved documents
    using the question's embedding.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    model = get_embedding_model()

    # Convert the user's question into an embedding.
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

    results = []

    # Search every saved document index.
    for document_dir in VECTOR_DB_DIR.iterdir():

        if not document_dir.is_dir():
            continue

        index_path = document_dir / "index.faiss"
        metadata_path = document_dir / "chunks.json"

        if not index_path.exists() or not metadata_path.exists():
            continue

        index = faiss.read_index(str(index_path))

        # The query and stored vectors must have
        # the same embedding dimensions.
        if index.d != query_vector.shape[1]:
            continue

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        # Search for the nearest chunks in this index.
        scores, indices = index.search(
            query_vector,
            min(top_k, index.ntotal)
        )

        for score, position in zip(scores[0], indices[0]):

            if position < 0:
                continue

            result = metadata[position].copy()
            result["score"] = float(score)

            results.append(result)

    # Combine results from all uploaded documents
    # and keep the most relevant chunks.
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]
