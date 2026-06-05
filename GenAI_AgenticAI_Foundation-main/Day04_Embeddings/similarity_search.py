# Day 4 — GenAI & Agentic AI Foundation
# similarity_search.py — Cosine similarity between sentence embeddings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

print("=== Day 4: Semantic Similarity Search ===\n")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------
# 1. Compare one query against a document set
# ------------------------------------------
print("--- 1. Query vs Document similarity ---\n")

query = "How do AI agents make decisions?"

documents = [
    "Agentic AI systems plan and take actions to achieve goals.",
    "Neural networks are inspired by the human brain.",
    "AI agents use reasoning loops like ReAct to decide what to do next.",
    "Python is great for data science and machine learning.",
    "LangChain supports building multi-step reasoning agents.",
    "The Eiffel Tower is in Paris.",
]

query_vector   = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
doc_vectors    = np.array(embeddings_model.embed_documents(documents))

similarities   = cosine_similarity(query_vector, doc_vectors)[0]
ranked         = sorted(zip(similarities, documents), reverse=True)

print(f"Query: '{query}'\n")
print("Documents ranked by similarity:")
for score, doc in ranked:
    bar = "█" * int(score * 30)
    print(f"  {score:.4f} {bar}  {doc}")

# ------------------------------------------
# 2. All-pairs similarity matrix
# ------------------------------------------
print("\n--- 2. Similarity matrix for a small set ---\n")

sentences = [
    "Agents can use tools",
    "AI tools help agents act",
    "Dogs are great pets",
    "Cats are independent animals",
]

vecs  = np.array(embeddings_model.embed_documents(sentences))
matrix = cosine_similarity(vecs)

header = "         " + "  ".join([f"S{i+1}" for i in range(len(sentences))])
print(header)
for i, row in enumerate(matrix):
    scores = "  ".join([f"{v:.2f}" for v in row])
    print(f"  S{i+1} {sentences[i][:25]:<26} {scores}")

print("\nS1 & S2 are about agents/tools → high similarity")
print("S3 & S4 are about pets → high similarity")
print("S1 & S3 are unrelated → low similarity")
