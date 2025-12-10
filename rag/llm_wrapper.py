# rag/llm_wrapper.py
from typing import List
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file if exists (local dev)
load_dotenv()

# Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "ERROR: GEMINI_API_KEY is not set. Add it to your Cloud Run env vars."
    )

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Choose model (default to gemini-2.5-flash)
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)


def build_prompt(question: str, passages: List[str]) -> str:
    """
    Prompt for a university study assistant:
    - Student uploads course documents (lecture slides, PDFs, notes, etc.)
    - You answer based primarily on those documents.
    - Medium-length, clear, student-friendly explanations.
    """
    context_text = "\n\n".join(f"- {p}" for p in passages)

    prompt = f"""
You are a helpful study assistant for a university student.

The student uploads documents (lecture slides, PDFs, notes, assignments, exam reviews, etc.)
for one or more of their courses (e.g., Cloud Computing, Distributed Systems, Machine Learning, Databases, etc.).
The text you see in CONTEXT is made of small chunks taken from those documents.

You MUST always follow these rules:

1. **Document-grounded answers**
   - Treat the CONTEXT as the primary source of truth.
   - Use it to answer the question as much as possible.
   - You may add light general background knowledge **only** to make the explanation clearer or to fill in obvious gaps.
   - Do NOT contradict the documents.
   - If the documents clearly don’t cover the topic, say so briefly
     (e.g. “The provided course materials do not discuss this directly, but in general …”) and keep the “in general” part short.

2. **Relevance and personal / off-topic questions**
   - If the question is personal or unknowable from documents
     (e.g. “What color is my toothbrush?”, “What did I eat yesterday?”, “Who am I?”),
     you MUST respond briefly that you cannot know this and ask the student to ask a question
     related to the course concepts instead.
   - If the question is clearly unrelated to academic/course content,
     say that it is outside the scope of the uploaded course materials and suggest asking about topics
     from lectures, slides, or assignments.

3. **When to use step-by-step format**
   - If the question is about an algorithm, process, workflow, protocol, or multi-step concept
     (e.g. “How does Map-Reduce work step by step?”, “How does 2PC work?”, “Explain how auto-scaling operates”),
     then explain it in ordered steps (Step 1, Step 2, …).
   - Otherwise, do NOT force “Step 1 / Step 2” formatting. Instead, prefer:
     - 1–2 short paragraphs, and/or
     - a few bullet points for key ideas.

4. **Style and length**
   - Write in clear, fluent ENGLISH.
   - Aim for a **medium-length** answer: a short overview plus some detail, but not an extremely long essay.
   - Good structure for most concept questions:
     - A short 2–3 sentence overview.
     - Then more detail (paragraphs or bullets).
     - A simple example if helpful (especially for abstract concepts).
   - You are talking to a student, so keep the tone friendly and explanatory, not overly formal.

---
CONTEXT (snippets from the student’s course documents):
{context_text}

---
QUESTION FROM STUDENT:
{question}

---
ANSWER (follow all the rules above):
"""
    return prompt


def generate_answer(
    question: str,
    passages: List[str],
    max_new_tokens: int = 2500,
) -> str:
    """
    Generates a medium-length, structured English answer using Gemini.
    """
    # Safety: limit passage length so prompt doesn't explode
    PASSAGE_MAX_CHARS = 2000
    passages = [p[:PASSAGE_MAX_CHARS] for p in passages]

    prompt = build_prompt(question, passages)

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": max_new_tokens,
            },
        )

        if not response or not getattr(response, "text", None):
            return "Model did not return a valid response."

        return response.text.strip()

    except Exception as e:
        print("Gemini Error:", repr(e))
        return f"Gemini error: {e}"


# Backwards compatibility alias
def generate_llm_answer(question: str, passages: List[str]) -> str:
    return generate_answer(question, passages)
