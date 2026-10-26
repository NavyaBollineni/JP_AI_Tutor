
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(processed_result: dict) -> dict:
    """
    Split processed document text into smaller,
    overlapping chunks for the RAG pipeline.
    """

    if not isinstance(processed_result, dict):
        raise ValueError("Input must be a dictionary.")

    if processed_result.get("status") != "success":
        raise ValueError("Input data has not been processed successfully.")

    text = processed_result.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("No text available for chunking.")

    # Configure the chunking strategy
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    # Split the document into chunks
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No chunks were created.")

    # Return JSON-compatible data
    return {
        "document_id": processed_result.get("document_id"),
        "filename": processed_result.get("filename"),
        "file_type": processed_result.get("file_type"),
        "status": "success",
        "chunks": [
            {
                "chunk_id": index + 1,
                "text": chunk
            }
            for index, chunk in enumerate(chunks)
        ],
        "chunking": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "total_chunks": len(chunks)
        }
    }
