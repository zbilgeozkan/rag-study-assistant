from rag.llm_wrapper import generate_answer

question = "How does Map-Reduce work in detail step by step? Can you explain the steps?"

passages = [
    "Map-Reduce consists of map, shuffle/sort, and reduce phases. Users implement the map and reduce functions, while the framework handles key grouping and sorting.",
    "In the map phase, input key-value pairs (k1, v1) are transformed into intermediate pairs (k2, v2). In the shuffle/sort phase, values are grouped by key. In the reduce phase, grouped values are aggregated.",
    "Lecture 5 explains Map-Reduce in cloud applications, emphasizing scalability, fault tolerance, and using services such as Amazon EMR on top of cloud infrastructure."
]

answer = generate_answer(question, passages)

print("\n=== LLM OUTPUT START ===\n")
print(answer)
print("\n=== LLM OUTPUT END ===\n")
