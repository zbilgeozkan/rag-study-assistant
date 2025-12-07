import os
import time
from datetime import datetime

from query_faiss import FAISSQuery
from llm_wrapper import generate_answer


if __name__ == "__main__":
    question = "How can I remove the mobile device from the headset?"

    # Output directory (under rag/data or project/data if you prefer)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1) Retrieve passages from FAISS
    faiss_query = FAISSQuery()
    results = faiss_query.query(question, top_k=5)
    passages = [r["text"] for r in results]

    # 2) Generate answer with Gemini (via llm_wrapper)
    start_time = time.time()
    answer = generate_answer(question, passages)
    elapsed_time = time.time() - start_time

    # 3) Print to console
    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)
    print(f"\nTime: {elapsed_time:.2f} seconds")

    # 4) Save to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(data_dir, f"output_{timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n\n")
        f.write("Passages used:\n")
        for i, p in enumerate(passages, start=1):
            f.write(f"[{i}] {p}\n\n")
        f.write(f"Answer:\n{answer}\n\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n")

    print(f"\nOutput saved to: {output_file}")
