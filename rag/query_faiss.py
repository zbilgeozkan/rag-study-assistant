import json
import faiss
import numpy as np
import os
import google.generativeai as genai

# ===============================
# Gemini setup
# ===============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("ERROR: GEMINI_API_KEY missing!")

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/text-embedding-004"


class FAISSQuery:
    def __init__(self, index_path="data/faiss_index.bin", metadata_path="data/faiss_metadata.json"):
        # Load FAISS
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # --------------------------
    # Gemini query embedding
    # --------------------------
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding using Gemini (must match index embeddings)."""

        response = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )

        embedding = np.array(response["embedding"], dtype=np.float32)
        return embedding.reshape(1, -1)

    # --------------------------
    # FAISS retrieval
    # --------------------------
    def query(self, text: str, top_k: int = 5):
        # Embed query
        vec = self.embed_query(text)

        # Search FAISS
        distances, indices = self.index.search(vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            m = self.metadata[idx]
            results.append({
                "text": m["text"],
                "source": m["source"],
                "page": m["page"],
                "title": m["title"],
                "distance": float(dist)
            })

        return results


# Optional test
if __name__ == "__main__":
    q = FAISSQuery()
    out = q.query("What is virtualization?", top_k=5)
    with open("data/query_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved to data/query_test.json")
