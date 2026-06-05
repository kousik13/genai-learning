# Day 4 — GenAI & Agentic AI Foundation
# embeddings.py — Convert text into vector embeddings using HuggingFace

from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

print("=== Day 4: Text Embeddings ===\n")

# Load the embedding model (downloads on first run, ~90MB)
print("Loading embedding model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Model loaded!\n")

# ------------------------------------------
# 1. Embed a single query
# ------------------------------------------
print("--- 1. Single sentence embedding ---\n")

sentence = "Agentic AI systems can plan, reason, and take actions autonomously."
vector = embeddings_model.embed_query(sentence)

print(f"Sentence  : {sentence}")
print(f"Vector dim: {len(vector)}")
print(f"First 8   : {[round(v, 4) for v in vector[:8]]}")

# ------------------------------------------
# 2. Embed multiple documents at once
# ------------------------------------------
print("\n--- 2. Batch document embeddings ---\n")

docs = [
    "LangChain is a framework for building LLM applications.",
    "FAISS is a library for fast vector similarity search.",
    "Agents use tools to interact with the real world.",
    "RAG combines retrieval with generation for accurate answers.",
]

vectors = embeddings_model.embed_documents(docs)

print(f"Number of documents : {len(docs)}")
print(f"Vector size per doc : {len(vectors[0])}")

for i, doc in enumerate(docs):
    print(f"  Doc {i+1}: '{doc[:50]}...' → vector[0]={vectors[i][0]:.4f}")

# ------------------------------------------
# 3. Vector magnitude check
# ------------------------------------------
print("\n--- 3. Vector norms (should be ~1.0 for normalized embeddings) ---\n")
for i, v in enumerate(vectors):
    norm = np.linalg.norm(v)
    print(f"  Doc {i+1} norm: {norm:.4f}")
