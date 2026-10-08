import os
import shutil
import uuid
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import rag_pipeline

load_dotenv()

app = FastAPI(
    title="Enterprise Hybrid RAG API",
    description="Multi-document Retrieval-Augmented Generation API with Hybrid RRF Search and Multi-LLM Support.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = rag_pipeline.UPLOAD_DIR
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv", ".docx"}


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="The query to ask the document knowledge base.")
    doc_id: Optional[str] = Field("all", description="Document ID or 'all' to search across entire knowledge base.")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve.")
    search_mode: str = Field("hybrid", description="Search strategy: 'hybrid', 'dense', or 'sparse'.")
    provider: Optional[str] = Field(None, description="LLM provider: 'auto', 'gemini', 'groq', 'claude', 'openai', 'offline'.")
    model: Optional[str] = Field(None, description="Specific model name for the selected provider.")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    stats = rag_pipeline.get_system_stats()
    return {
        "status": "healthy",
        "service": "Enterprise Hybrid RAG API",
        "stats": stats,
    }


@app.get("/stats", tags=["System"])
def stats():
    return rag_pipeline.get_system_stats()


@app.post("/upload", tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Uploads and ingests a PDF, TXT, Markdown, or CSV file."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = rag_pipeline.ingest_document(str(temp_path), file.filename)
        return result
    except ValueError as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
    finally:
        file.file.close()


@app.get("/documents", tags=["Knowledge Base"])
def list_documents():
    """Lists all indexed documents across the knowledge base."""
    return rag_pipeline.list_documents()


@app.get("/documents/{doc_id}", tags=["Knowledge Base"])
def get_document(doc_id: str):
    """Retrieves metadata for a specific document."""
    doc = rag_pipeline.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


@app.delete("/documents/{doc_id}", tags=["Knowledge Base"])
def delete_document(doc_id: str):
    """Deletes a document and its embeddings from the vector store."""
    success = rag_pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "success", "deleted_doc_id": doc_id}


@app.post("/ask", tags=["Query Engine"])
def ask(req: AskRequest):
    """Standard query endpoint returning answer, sources, and confidence score."""
    if req.doc_id != "all" and not rag_pipeline.get_document(req.doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found. Upload it first.")

    try:
        result = rag_pipeline.ask_question(
            doc_id=req.doc_id,
            question=req.question,
            top_k=req.top_k,
            search_mode=req.search_mode,
            provider=req.provider,
            model=req.model,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream", tags=["Query Engine"])
def ask_stream(req: AskRequest):
    """Server-Sent Events (SSE) streaming endpoint for real-time token generation."""
    if req.doc_id != "all" and not rag_pipeline.get_document(req.doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_id}' not found. Upload it first.")

    def event_stream():
        for chunk in rag_pipeline.ask_question_stream(
            doc_id=req.doc_id,
            question=req.question,
            top_k=req.top_k,
            search_mode=req.search_mode,
            provider=req.provider,
            model=req.model,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
