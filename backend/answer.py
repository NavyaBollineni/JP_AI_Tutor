
from datetime import datetime, timezone


def format_answer(llm_result: dict) -> dict:
    """
    Format the LLM result into a consistent response.

    Input:
        Result from backend/llm.py

    Output:
        Clean JSON-compatible answer.
    """

    if not isinstance(llm_result, dict):
        raise ValueError("LLM result must be a dictionary.")

    if llm_result.get("status") != "success":
        raise ValueError("Cannot format an unsuccessful LLM result.")

    answer_text = llm_result.get("answer", "")

    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("The generated answer is empty.")

    return {
        "status": "success",
        "question": llm_result.get("question", ""),
        "answer": answer_text.strip(),
        "language": llm_result.get("language", "English"),
        "model": llm_result.get("model", "unknown"),
        "source_document": llm_result.get("source_document"),
        "used_context": llm_result.get("used_context", False),
        "context_chunks": llm_result.get("context_chunks", []),
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }
