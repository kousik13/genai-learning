# Day 5 — GenAI & Agentic AI Foundation
# search_vector.py — Advanced vector search: filters, retrievers, and MMR

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

print("=== Day 5: Advanced Vector Search ===\n")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample knowledge base
documents = [
    Document(page_content="LangChain agents use the ReAct pattern: Reason then Act.", metadata={"category": "agents"}),
    Document(page_content="Tools in LangChain let agents interact with APIs, databases, and search engines.", metadata={"category": "agents"}),
    Document(page_content="An AgentExecutor manages the agent's reasoning loop and tool calls.", metadata={"category": "agents"}),
    Document(page_content="RAG retrieves relevant documents before generating an answer.", metadata={"category": "rag"}),
    Document(page_content="A retriever in LangChain wraps a vector store and returns relevant docs.", metadata={"category": "rag"}),
    Document(page_content="FAISS performs approximate nearest-neighbor search using L2 or cosine distance.", metadata={"category": "vectordb"}),
    Document(page_content="Chroma supports metadata filtering natively alongside vector search.", metadata={"category": "vectordb"}),
    Document(page_content="Embeddings must be consistent: use the same model at index and query time.", metadata={"category": "embeddings"}),
]

vectorstore = FAISS.from_documents(documents, embeddings_model)

# ------------------------------------------
# 1. Standard retriever (as_retriever)
# ------------------------------------------
print("--- 1. Standard Retriever ---\n")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("how do agents decide what to do?")

for d in docs:
    print(f"  [{d.metadata['category']}] {d.page_content}")

# ------------------------------------------
# 2. MMR — Maximal Marginal Relevance
#    Returns diverse results, avoids duplicates
# ------------------------------------------
print("\n--- 2. MMR Retriever (diverse results) ---\n")

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 6}
)

docs_mmr = mmr_retriever.invoke("tell me about agents and vector stores")
print("MMR results (more diverse):")
for d in docs_mmr:
    print(f"  [{d.metadata['category']}] {d.page_content}")

# ------------------------------------------
# 3. Similarity score threshold
# ------------------------------------------
print("\n--- 3. Similarity Score Threshold Retriever ---\n")

thresh_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4, "k": 5}
)

docs_thresh = thresh_retriever.invoke("vector database for AI")
print(f"Documents above threshold ({len(docs_thresh)} found):")
for d in docs_thresh:
    print(f"  [{d.metadata['category']}] {d.page_content}")
