
import faiss
import numpy as np

from backend.embedding import get_embedding_model


def retrieve_chunks(
    embedded_result: dict,
    question: str,
    top_k: int = 3
) -> dict:
    """
    Find the document chunks most relevant to a question
    using local embeddings and FAISS.
    """

    if not isinstance(embedded_result, dict):
        raise ValueError("Input must be a dictionary.")

    if embedded_result.get("status") != "success":
        raise ValueError("Embedding generation was not successful.")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks = embedded_result.get("chunks", [])

    if not chunks:
        raise ValueError("No embedded chunks are available.")

    # Extract the document vectors.
    vectors = np.asarray(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32
    )

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Invalid embedding vectors.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors contain invalid values.")

    # Load the same model used for document embeddings.
    model = get_embedding_model()

    # Convert the user's question into an embedding.
    query_vector = model.encode(
        [question.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    if query_vector.shape[1] != vectors.shape[1]:
        raise ValueError(
            "Question and document embedding dimensions do not match."
        )

    # Build an in-memory FAISS index.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Retrieve the closest matching chunks.
    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    retrieved_chunks = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        chunk = chunks[int(index_position)]

        retrieved_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "similarity_score": float(score)
        })

    return {
        "document_id": embedded_result.get("document_id"),
        "filename": embedded_result.get("filename"),
        "question": question,
        "status": "success",
        "retrieved_count": len(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks
    }
