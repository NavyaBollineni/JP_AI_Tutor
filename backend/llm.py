import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


def get_llm():
    """Create a Groq client using an environment variable."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to the root .env file."
        )

    return ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        timeout=45,
        max_retries=1,
    )


def generate_answer(
    retrieval_result: dict,
    language: str = "English",
    chat_history: list | None = None,
) -> dict:
    """Generate an answer using document context and recent chat history."""

    if not isinstance(retrieval_result, dict):
        raise ValueError(
            "Retrieval result must be a dictionary."
        )

    if retrieval_result.get("status") != "success":
        raise ValueError(
            "Retrieval was unsuccessful."
        )

    question = str(
        retrieval_result.get("question", "")
    ).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    retrieved_chunks = retrieval_result.get(
        "retrieved_chunks", []
    )

    context_parts = []

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get("chunk_id", "unknown")
        chunk_text = str(
            chunk.get("text", "")
        ).strip()

        if chunk_text:
            context_parts.append(
                f"Chunk {chunk_id}:\n{chunk_text}"
            )

    context = "\n\n".join(context_parts)

    if not context:
        context = (
            "No relevant document text was retrieved."
        )

    # Build recent conversation history.
    history_parts = []

    for message in (chat_history or [])[-5:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            continue

        if role == "user":
            speaker = "User"
        elif role == "assistant":
            speaker = "JP AI Tutor"
        else:
            continue

        history_parts.append(
            f"{speaker}: {content}"
        )

    history_text = (
        "\n".join(history_parts)
        if history_parts
        else "This is the beginning of the conversation."
    )

    prompt = f"""
You are JP AI Tutor, a helpful Japanese language learning assistant.

Instructions:
1. Answer in {language}.
2. Keep answers short, clear, and easy to read.
3. Usually answer in 2–4 short sentences.
4. Use simple words and avoid unnecessary explanations.
5. Give only two examples or facts when the user asks for two.
6. For Japanese translations, include Japanese, hiragana,
   romaji, and English meaning when appropriate.
7. Use uploaded document information when relevant.
8. Never invent facts or claim the document says something
   it does not say.
9. If the user asks for detailed information, provide it.
10. For conversation practice, ask one question at a time.
11. Use short bullet points when they improve readability.
12. Use recent conversation history to understand follow-up questions.
13. Do not treat previous assistant answers as guaranteed facts.
14. If the user refers to something discussed earlier, use the
    recent conversation when the information is available.

RECENT CONVERSATION:
{history_text}

DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

ANSWER:
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, str):
            answer_text = content.strip()

        elif isinstance(content, list):
            answer_text = "\n".join(
                str(block.get("text", ""))
                if isinstance(block, dict)
                else str(block)
                for block in content
            ).strip()

        else:
            answer_text = str(content).strip()

        if not answer_text:
            raise ValueError(
                "Groq returned an empty answer."
            )

        return {
            "status": "success",
            "question": question,
            "language": language,
            "model": MODEL_NAME,
            "answer": answer_text,
            "source_document": retrieval_result.get(
                "filename"
            ),
            "used_context": bool(context_parts),
            "context_chunks": [
                chunk.get("chunk_id", "unknown")
                for chunk in retrieved_chunks
                if isinstance(chunk, dict)
                and str(chunk.get("text", "")).strip()
            ],
        }

    except Exception as error:
        raise RuntimeError(
            f"Groq answer generation failed: {error}"
        ) from error