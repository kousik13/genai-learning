# Day 5 — GenAI & Agentic AI Foundation
# vector_store.py — Create and persist a FAISS vector store

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

print("=== Day 5: FAISS Vector Store ===\n")

# Embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------
# 1. Create vector store from documents
# ------------------------------------------
print("--- 1. Building vector store from documents ---\n")

documents = [
    Document(page_content="An AI agent is a system that perceives its environment and takes actions.", metadata={"topic": "agents", "day": 5}),
    Document(page_content="LangChain agents can call tools, search the web, and run code.", metadata={"topic": "agents", "day": 5}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It helps LLMs answer accurately.", metadata={"topic": "rag", "day": 5}),
    Document(page_content="FAISS is an efficient library for nearest-neighbor search in high dimensions.", metadata={"topic": "vectordb", "day": 5}),
    Document(page_content="Chroma is another popular vector store used with LangChain.", metadata={"topic": "vectordb", "day": 5}),
    Document(page_content="Embeddings convert text into numerical vectors for semantic search.", metadata={"topic": "embeddings", "day": 5}),
    Document(page_content="The ReAct pattern lets agents alternate between reasoning and acting.", metadata={"topic": "agents", "day": 5}),
    Document(page_content="Prompt engineering is the art of crafting inputs to get better LLM outputs.", metadata={"topic": "prompts", "day": 5}),
]

vectorstore = FAISS.from_documents(documents, embeddings_model)
print(f"Vector store created with {len(documents)} documents.\n")

# ------------------------------------------
# 2. Basic similarity search
# ------------------------------------------
print("--- 2. Semantic similarity search ---\n")

query = "What can AI agents do?"
results = vectorstore.similarity_search(query, k=3)

print(f"Query: '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"  Result {i}: [{doc.metadata.get('topic')}] {doc.page_content}")

# ------------------------------------------
# 3. Search with score
# ------------------------------------------
print("\n--- 3. Search with relevance scores ---\n")

query2 = "vector database for semantic search"
results_with_score = vectorstore.similarity_search_with_score(query2, k=3)

print(f"Query: '{query2}'\n")
for doc, score in results_with_score:
    print(f"  Score: {score:.4f}  [{doc.metadata.get('topic')}]  {doc.page_content}")

# ------------------------------------------
# 4. Save and reload the vector store
# ------------------------------------------
print("\n--- 4. Save and reload the vector store ---\n")

save_path = "faiss_genai_store"
vectorstore.save_local(save_path)
print(f"Saved to '{save_path}/'")

reloaded = FAISS.load_local(
    save_path,
    embeddings_model,
    allow_dangerous_deserialization=True
)

result = reloaded.similarity_search("how do agents reason?", k=1)
print(f"Reloaded search result: {result[0].page_content}")
