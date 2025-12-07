# rag/llm_wrapper.py
from typing import List
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file if exists
load_dotenv()

# Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "ERROR: GEMINI_API_KEY is not set. Add it to your .env file or Cloud Run environment."
    )

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Choose model (flash is fast + cheap, pro is stronger)
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
model = genai.GenerativeModel(MODEL_NAME)


def build_prompt(question: str, passages: List[str]) -> str:
    """
    Build the final prompt used by Gemini.
    Produces **ENGLISH** answers.
    """
    context_text = "\n\n".join(f"- {p}" for p in passages)

    prompt = f"""
You are a helpful AI assistant for Retrieval-Augmented Generation (RAG).
Your task is to answer the user’s question **ONLY** using the information provided in the context.

If the answer is not in the context, say:
"I cannot answer this because the information is not available in the provided context."

Give a clear, structured, step-by-step answer in **ENGLISH**.

---
CONTEXT:
{context_text}

---
QUESTION:
{question}

---
ANSWER (English, clear and step-by-step):
"""

    return prompt


def generate_answer(question: str, passages: List[str], max_new_tokens: int = 500) -> str:
    """
    Generates an English answer using Gemini.
    Replaces the old HuggingFace LLM.
    """
    prompt = build_prompt(question, passages)

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,     # more stable answers than 0.7
                "max_output_tokens": max_new_tokens,
            },
        )
        return (response.text or "").strip()

    except Exception as e:
        # DEBUG: print full error and also return it
        print("Gemini Error:", repr(e))
        return f"Gemini error: {e}"



# Optional alias for compatibility with old imports
def generate_llm_answer(question: str, passages: List[str]) -> str:
    return generate_answer(question, passages)
