# Day 6 — GenAI & Agentic AI Foundation
# retriever.py — Conversational RAG with chat history (contextual re-ranking)

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

print("=== Day 6: Conversational RAG Retriever ===\n")

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content="Agents use tools to interact with external systems like APIs and databases."),
    Document(page_content="The ReAct agent pattern stands for: Reasoning + Acting in a loop."),
    Document(page_content="Tool calling in LangChain uses the @tool decorator on Python functions."),
    Document(page_content="Agents can be stopped by max_iterations or when they produce a Final Answer."),
    Document(page_content="RAG retrieval can be improved by query rewriting before embedding lookup."),
    Document(page_content="Multi-agent systems have multiple specialized agents collaborating on a task."),
]

vectorstore = FAISS.from_documents(documents, embeddings_model)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})

# ------------------------------------------
# Contextualize prompt — rewrites follow-up
# questions into standalone questions
# ------------------------------------------
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest user question,
rewrite the question as a clear standalone question.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

contextualizer = contextualize_prompt | chat | StrOutputParser()

# ------------------------------------------
# Answer prompt
# ------------------------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a GenAI expert assistant.
Answer using only the context below.
If not found, say 'Not in my knowledge base.'

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ------------------------------------------
# Conversational RAG chain
# ------------------------------------------
def build_context(inputs):
    rewritten = contextualizer.invoke(inputs)
    docs = retriever.invoke(rewritten)
    return format_docs(docs)

rag_chain = (
    {
        "input":        itemgetter("input"),
        "chat_history": itemgetter("chat_history"),
        "context":      build_context
    }
    | answer_prompt
    | chat
    | StrOutputParser()
)

# ------------------------------------------
# Multi-turn demo
# ------------------------------------------
chat_history = []

def ask(question):
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

questions = [
    "What is the ReAct pattern?",
    "How do agents know when to stop using it?",
    "Can multiple agents work together using this approach?",
]

for q in questions:
    print(f"You : {q}")
    print(f"Bot : {ask(q)}\n")
