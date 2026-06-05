# Day 6 — GenAI & Agentic AI Foundation
# document_loader.py — Load and split documents for RAG pipelines

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

print("=== Day 6: Document Loading & Splitting ===\n")

# ------------------------------------------
# 1. Load from plain text (simulated)
# ------------------------------------------
print("--- 1. Load from raw text (simulated doc) ---\n")

# Simulate a loaded document (in real use: TextLoader, PyPDFLoader, CSVLoader, etc.)
raw_text = """
Agentic AI represents a shift from single-turn question answering to multi-step autonomous task completion.
Unlike traditional AI that responds to prompts, agentic systems plan, execute, and adapt across many steps.

LangChain provides the AgentExecutor class which manages an agent's reasoning loop.
The agent repeatedly chooses a tool, runs it, observes the result, and decides the next action.

Tools are functions the agent can call. Common tools include web search, calculators, database queries,
and code interpreters. The agent selects the right tool based on the task at hand.

RAG (Retrieval-Augmented Generation) is often used alongside agents to give them access to private knowledge.
The retriever fetches relevant documents, which are injected into the prompt as context.

Evaluation is critical in GenAI systems. Metrics like faithfulness, relevance, and groundedness help
measure how well a RAG or agent system is performing on real tasks.
"""

doc = Document(page_content=raw_text, metadata={"source": "genai_notes.txt"})
print(f"Loaded document length: {len(raw_text)} characters\n")

# ------------------------------------------
# 2. Split with RecursiveCharacterTextSplitter
# ------------------------------------------
print("--- 2. RecursiveCharacterTextSplitter ---\n")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""]
)

recursive_chunks = recursive_splitter.split_documents([doc])
print(f"Chunks created: {len(recursive_chunks)}\n")
for i, chunk in enumerate(recursive_chunks, 1):
    print(f"  Chunk {i} ({len(chunk.page_content)} chars): {chunk.page_content[:80].strip()}...")

# ------------------------------------------
# 3. Split with CharacterTextSplitter
# ------------------------------------------
print("\n--- 3. CharacterTextSplitter (splits on newlines) ---\n")

char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=0,
)

char_chunks = char_splitter.split_documents([doc])
print(f"Chunks created: {len(char_chunks)}\n")
for i, chunk in enumerate(char_chunks, 1):
    print(f"  Chunk {i}: {chunk.page_content[:80].strip()}...")

# ------------------------------------------
# 4. Metadata inspection
# ------------------------------------------
print("\n--- 4. Metadata on split chunks ---\n")
for chunk in recursive_chunks[:2]:
    print(f"  Source: {chunk.metadata.get('source')} | Length: {len(chunk.page_content)}")
