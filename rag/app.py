# rag/app.py

import time
from typing import List, Any

from fastapi import FastAPI
from pydantic import BaseModel

from .query_faiss import FAISSQuery
from .llm_wrapper import generate_answer


app = FastAPI(
    title="Personal Study Assistant RAG API",
    description="RAG pipeline using FAISS + Gemini LLM",
    version="1.0.0",
)

# FAISS index loader (once, on import)
faiss_query = FAISSQuery()


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