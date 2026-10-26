
import re


def process_data(ingestion_result: dict) -> dict:
    """
    Clean extracted document text and return
    a JSON-compatible dictionary for the next RAG stage.
    """

    if not isinstance(ingestion_result, dict):
        raise ValueError("Input must be a dictionary.")

    if ingestion_result.get("status") != "success":
        raise ValueError("Ingestion was not successful.")

    text = ingestion_result.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("No text available to process.")

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove null characters
    text = text.replace("\x00", "")

    # 3. Remove unwanted spaces at the beginning
    #    and end of each line
    lines = [line.strip() for line in text.split("\n")]

    # 4. Remove consecutive duplicate non-empty lines
    cleaned_lines = []
    previous_line = None

    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            previous_line = None
            continue

        if line != previous_line:
            cleaned_lines.append(line)

        previous_line = line

    # 5. Join the lines and normalize excessive blank lines
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        raise ValueError("No usable text remains after cleaning.")

    # 6. Return the processed result
    return {
        "document_id": ingestion_result.get("document_id"),
        "filename": ingestion_result.get("filename"),
        "file_type": ingestion_result.get("file_type"),
        "status": "success",
        "text": cleaned_text,
        "processing": {
            "cleaning_applied": True,
            "character_count_before": len(text),
            "character_count_after": len(cleaned_text)
        }
    }
