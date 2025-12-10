import os
import time
from typing import List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .query_faiss import FAISSQuery
from .llm_wrapper import generate_answer
from .gcs_utils import download_file_from_gcs, file_exists_in_gcs

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Personal Study Assistant RAG API",
    description="RAG pipeline using FAISS + Gemini LLM",
    version="1.0.0",
)

# frontend klasörünü /web altında servis et
app.mount("/web", StaticFiles(directory="frontend", html=True), name="frontend")


# ==========
# GCS & FAISS paths
# ==========
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "rag-documents-bucket-icu")

LOCAL_INDEX_PATH = "data/faiss_index.bin"
LOCAL_METADATA_PATH = "data/faiss_metadata.json"
LOCAL_CHUNKS_PATH = "data/chunks.json"

GCS_INDEX_PATH = "faiss/faiss_index.bin"
GCS_METADATA_PATH = "faiss/faiss_metadata.json"
GCS_CHUNKS_PATH = "faiss/chunks.json"

# FAISSQuery global (startup'ta initialize edilecek)
faiss_query: FAISSQuery | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """
    Container cold start olduğunda 1 kere çalışır:
    1. GCS'den FAISS index + metadata + chunks dosyalarını indirir
    2. FAISSQuery'yi bu dosyalar üzerinden initialize eder
    """
    print("[STARTUP] Downloading FAISS assets from GCS...")

    # 1) GCS'de dosyalar var mı kontrol et ve indir
    if not file_exists_in_gcs(GCS_INDEX_PATH):
        print(f"[ERROR] FAISS index not found in GCS: gs://{BUCKET_NAME}/{GCS_INDEX_PATH}")
        # Burada return dersen FAISSQuery hiç yüklenmez; loglayıp devam ediyoruz
    else:
        download_file_from_gcs(GCS_INDEX_PATH, LOCAL_INDEX_PATH, BUCKET_NAME)

    if file_exists_in_gcs(GCS_METADATA_PATH):
        download_file_from_gcs(GCS_METADATA_PATH, LOCAL_METADATA_PATH, BUCKET_NAME)
    else:
        print(f"[WARN] Metadata not found in GCS: gs://{BUCKET_NAME}/{GCS_METADATA_PATH}")

    if file_exists_in_gcs(GCS_CHUNKS_PATH):
        download_file_from_gcs(GCS_CHUNKS_PATH, LOCAL_CHUNKS_PATH, BUCKET_NAME)
    else:
        print(f"[WARN] Chunks not found in GCS: gs://{BUCKET_NAME}/{GCS_CHUNKS_PATH}")

    # 2) FAISSQuery'yi initialize et
    global faiss_query
    try:
        faiss_query = FAISSQuery(
            index_path=LOCAL_INDEX_PATH,
            metadata_path=LOCAL_METADATA_PATH,
        )
        print("[STARTUP] FAISSQuery initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize FAISSQuery: {e}")
        faiss_query = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Passage(BaseModel):
    text: str
    source: str | None = None
    page: int | None = None
    title: str | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    time: float
    passages: List[Passage]


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """
    Main RAG endpoint:
    1. Retrieves top-k passages from FAISS.
    2. Sends them to Gemini via llm_wrapper.generate_answer.
    3. Returns the answer + used passages.
    """
    global faiss_query

    if faiss_query is None:
        # Startup'ta FAISSQuery oluşturulamadıysa
        raise HTTPException(
            status_code=503,
            detail="FAISS index is not loaded yet. Please try again later.",
        )

    question = payload.question
    top_k = payload.top_k

    start_time = time.time()

    # 1) Retrieve passages from FAISS
    faiss_results: List[dict[str, Any]] = faiss_query.query(question, top_k=top_k)
    passages_text = [r.get("text", "") for r in faiss_results]

    # 2) Generate answer using Gemini (through llm_wrapper)
    answer = generate_answer(question, passages_text)

    elapsed = time.time() - start_time

    # 3) Map raw FAISS dicts into Passage models
    passages_out: List[Passage] = []
    for r in faiss_results:
        passages_out.append(
            Passage(
                text=r.get("text", ""),
                source=r.get("source"),
                page=r.get("page"),
                title=r.get("title"),
                distance=r.get("distance"),
            )
        )

    return AskResponse(
        question=question,
        answer=answer,
        time=elapsed,
        passages=passages_out,
    )
