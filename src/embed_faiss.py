import os
import json
import faiss
import numpy as np
import google.generativeai as genai

from rag.gcs_utils import upload_file_to_gcs


# =============================================
# 1) Configure Gemini API
# =============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in environment!")

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/text-embedding-004"  # Latest + Best for embeddings


# =============================================
# Helper functions
# =============================================
def load_chunks(chunks_path="data/chunks.json"):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_or_load_embeddings(texts, embeddings_path="data/embeddings.npy"):
    if os.path.exists(embeddings_path):
        print("Embeddings found. Loading from cache...")
        arr = np.load(embeddings_path)
        print(f"Loaded embeddings with shape: {arr.shape}")
        return arr

    print("No cached embeddings found. Computing embeddings with Gemini (ONE BY ONE)...")

    embeddings = []

    for i, text in enumerate(texts):
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
        )

        if "embedding" not in response:
            raise RuntimeError(f"Missing embedding in response: {response}")

        emb = np.array(response["embedding"], dtype="float32")

        # Check dimension consistency
        if i == 0:
            expected_dim = emb.shape[0]
            print(f"Embedding dimension: {expected_dim}")
        else:
            if emb.shape[0] != expected_dim:
                raise RuntimeError(
                    f"Dimension mismatch at index {i}: got {emb.shape[0]}, expected {expected_dim}"
                )

        embeddings.append(emb)

        if i % 20 == 0:
            print(f"Embedded {i}/{len(texts)}")

    embeddings = np.vstack(embeddings)
    np.save(embeddings_path, embeddings)

    print(f"Embeddings saved to {embeddings_path} with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings, index_path="data/faiss_index.bin"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors.")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    return index


def save_metadata(chunks, metadata_path="data/faiss_metadata.json"):
    metadata = [
        {
            "id": c["id"],
            "text": c["text"],
            "source": c["source"],
            "page": c["page"],
            "title": c.get("title", "Unknown"),
        }
        for c in chunks
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata saved to {metadata_path}")


# =============================================
# Main pipeline
# =============================================
def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks.")

    embeddings = build_or_load_embeddings(texts)

    build_faiss_index(embeddings)
    save_metadata(chunks)

    bucket_name = os.getenv("GCS_BUCKET_NAME", "rag-documents-bucket-icu")

    upload_file_to_gcs("data/faiss_index.bin", "faiss/faiss_index.bin", bucket_name)
    upload_file_to_gcs("data/faiss_metadata.json", "faiss/faiss_metadata.json", bucket_name)
    upload_file_to_gcs("data/chunks.json", "faiss/chunks.json", bucket_name)
    upload_file_to_gcs("data/embeddings.npy", "faiss/embeddings.npy", bucket_name)

    print("All FAISS files uploaded to GCS.")


if __name__ == "__main__":
    main()