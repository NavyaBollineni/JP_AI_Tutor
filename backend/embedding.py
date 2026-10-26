
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load the model once and reuse it.
_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")

    return _model


def generate_embeddings(chunked_result: dict) -> dict:
    """
    Generate local embeddings for the text chunks.
    No API key is required.
    """

    if not isinstance(chunked_result, dict):
        raise ValueError("Input must be a dictionary.")

    if chunked_result.get("status") != "success":
        raise ValueError("Chunking was not successful.")

    chunks = chunked_result.get("chunks", [])

    if not chunks:
        raise ValueError("No chunks available for embedding.")

    texts = []

    for chunk in chunks:
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {chunk.get('chunk_id')} has no valid text."
            )

        texts.append(text)

    model = get_embedding_model()

    # Convert each text chunk into a numerical vector.
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": vector.tolist()
        })

    return {
        "document_id": chunked_result.get("document_id"),
        "filename": chunked_result.get("filename"),
        "status": "success",
        "embedding_model": MODEL_NAME,
        "total_embeddings": len(embedded_chunks),
        "embedding_dimensions": int(vectors.shape[1]),
        "chunks": embedded_chunks
    }
