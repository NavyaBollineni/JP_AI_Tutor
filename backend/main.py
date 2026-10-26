from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.data_ingestion import ingest_document
from backend.data_processing import process_data
from backend.text_chunking import chunk_text
from backend.embedding import generate_embeddings
from backend.retrieval import retrieve_chunks
from backend.llm import generate_answer
from backend.answer import format_answer


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="JP AI Tutor",
    description="Your personal Japanese learning assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One active uploaded document at a time.
CURRENT_DOCUMENT = None

# Temporary conversation memory in backend RAM.
# Each conversation stores its latest 5 individual messages.
CONVERSATION_MEMORY = {}
MAX_MESSAGES = 5



class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    language: str = "English"
    conversation_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )



@app.get("/")
def home():
    """Serve the frontend homepage."""

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend index.html was not found.",
        )

    return FileResponse(index_file)


@app.get("/health")
def health_check():
    """Check whether the backend is running."""

    return {
        "status": "healthy",
        "service": "JP AI Tutor",
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """Process an uploaded document and prepare it for retrieval."""

    global CURRENT_DOCUMENT

    try:
        # 1. Extract text from the uploaded file.
        ingestion_result = await ingest_document(file)

        # 2. Clean the extracted data.
        processing_result = process_data(
            ingestion_result
        )

        # 3. Split the processed text into chunks.
        chunking_result = chunk_text(
            processing_result
        )

        # 4. Generate document embeddings.
        embedding_result = generate_embeddings(
            chunking_result
        )

        if embedding_result.get("status") != "success":
            raise ValueError(
                "Could not create document embeddings."
            )

        # Keep the current document available for questions.
        CURRENT_DOCUMENT = embedding_result

        return {
            "status": "success",
            "message": "Your document is ready for questions!",
            "filename": embedding_result.get("filename"),
            "document_id": embedding_result.get("document_id"),
            "total_chunks": embedding_result.get(
                "total_embeddings", 0
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}",
        ) from error

    finally:
        await file.close()


@app.post("/chat")
async def chat(request: ChatRequest):
    """Answer a question using recent conversation history."""

    question = request.question.strip()
    conversation_id = request.conversation_id

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        # 1. Retrieve the latest messages for this conversation.
        chat_history = CONVERSATION_MEMORY.get(
            conversation_id, []
        )[-MAX_MESSAGES:]

        # 2. Retrieve document context, if a document is uploaded.
        if CURRENT_DOCUMENT is not None:
            retrieval_result = retrieve_chunks(
                embedded_result=CURRENT_DOCUMENT,
                question=question,
                top_k=3,
            )

        else:
            # General Japanese tutoring without a document.
            retrieval_result = {
                "status": "success",
                "document_id": None,
                "filename": None,
                "question": question,
                "retrieved_count": 0,
                "retrieved_chunks": [],
            }

        # 3. Generate an answer using recent history.
        llm_result = generate_answer(
            retrieval_result=retrieval_result,
            language=request.language,
            chat_history=chat_history,
        )

        # 4. Format the response for the frontend.
        formatted_result = format_answer(llm_result)

        # 5. Save the new question and answer in RAM.
        updated_history = chat_history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": llm_result["answer"],
            },
        ]

        # Keep only the latest 5 individual messages.
        CONVERSATION_MEMORY[conversation_id] = (
            updated_history[-MAX_MESSAGES:]
        )

        return formatted_result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {error}",
        ) from error


# Serve CSS, JavaScript, and other frontend assets.
# Keep this mount after the API routes.
app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True,
    ),
    name="frontend",
)