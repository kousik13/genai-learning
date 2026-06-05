# Day 6 — GenAI & Agentic AI Foundation
# rag_pipeline.py — Full RAG pipeline: load → embed → store → retrieve → generate

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("=== Day 6: Full RAG Pipeline ===\n")

# ------------------------------------------
# 1. LLM
# ------------------------------------------
chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------------------------
# 2. Knowledge base (simulating loaded docs)
# ------------------------------------------
documents = [
    Document(page_content="Agentic AI systems can plan and execute multi-step tasks without human intervention."),
    Document(page_content="LangChain's AgentExecutor runs the ReAct loop: Thought → Action → Observation → Repeat."),
    Document(page_content="Tools available to agents include web search, calculator, SQL query, and code execution."),
    Document(page_content="RAG pipelines retrieve relevant documents and inject them as context into the LLM prompt."),
    Document(page_content="FAISS (Facebook AI Similarity Search) performs fast approximate nearest-neighbor lookups."),
    Document(page_content="Chunking strategy affects RAG quality. Smaller chunks are precise; larger chunks have more context."),
    Document(page_content="Hallucination in LLMs means generating confident but factually incorrect information."),
    Document(page_content="Evaluating RAG pipelines uses metrics like faithfulness, answer relevance, and context recall."),
]

# ------------------------------------------
# 3. Embed and store
# ------------------------------------------
print("Building vector store...\n")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------------------------
# 4. Prompt
# ------------------------------------------
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful GenAI tutor.
Answer the question using ONLY the provided context.
If the answer is not in the context, say 'I don't have that information.'

Context:
{context}"""),
    ("human", "{question}")
])

# ------------------------------------------
# 5. Helper
# ------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------------------------------
# 6. RAG chain
# ------------------------------------------
rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | chat
    | StrOutputParser()
)

# ------------------------------------------
# 7. Run queries
# ------------------------------------------
questions = [
    "What is the ReAct pattern in agentic AI?",
    "How does RAG reduce hallucinations?",
    "What tools can LangChain agents use?",
    "What is the capital of France?",   # Out-of-context question
]

for q in questions:
    print(f"Q: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}\n")
