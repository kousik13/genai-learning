# Day 9 — GenAI & Agentic AI Foundation
# services.py — Business logic: chat, RAG, and agent service functions

import os
import math
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

load_dotenv()

# ------------------------------------------
# Shared LLM instance
# ------------------------------------------
_chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------
# 1. Chat service
# ------------------------------------------
_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful GenAI assistant. Be concise and accurate."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

_chat_chain = _chat_prompt | _chat | StrOutputParser()

def chat_service(message: str, history: list[dict]) -> str:
    """Run a single chat turn with optional history."""
    lc_history = []
    for turn in history:
        if turn.get("role") == "user":
            lc_history.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            lc_history.append(AIMessage(content=turn["content"]))

    return _chat_chain.invoke({"input": message, "history": lc_history})

# ------------------------------------------
# 2. RAG service
# ------------------------------------------
_default_docs = [
    "Agentic AI systems autonomously plan and execute multi-step tasks.",
    "RAG retrieves relevant documents before the LLM generates an answer.",
    "LangChain AgentExecutor manages the agent's reasoning loop.",
    "FAISS performs fast approximate nearest-neighbor search on vectors.",
    "Embeddings map text into dense float vectors for semantic comparison.",
]

_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer using ONLY the context below.
If not found, say 'Not in the provided documents.'

Context:
{context}"""),
    ("human", "{question}")
])

def rag_service(question: str, documents: list[str]) -> tuple[str, list[str]]:
    """Run RAG over provided documents or the default knowledge base."""
    docs_to_use = documents if documents else _default_docs
    lc_docs     = [Document(page_content=d) for d in docs_to_use]
    vectorstore = FAISS.from_documents(lc_docs, _embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    retrieved   = retriever.invoke(question)
    context     = "\n\n".join(d.page_content for d in retrieved)
    sources     = [d.page_content[:60] + "..." for d in retrieved]

    chain  = _rag_prompt | _chat | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer, sources

# ------------------------------------------
# 3. Agent service
# ------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Examples: '15 * 4', 'sqrt(81)', '2 ** 8'"""
    try:
        safe_env = {"__builtins__": {}, "math": math, "sqrt": math.sqrt}
        return f"{expression} = {eval(expression, safe_env)}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def knowledge_lookup(query: str) -> str:
    """Look up GenAI topics: agents, rag, embeddings, tools, llm, chains."""
    kb = {
        "agents":     "Agents loop: Think → Use tool → Observe → Repeat until Final Answer.",
        "rag":        "RAG retrieves documents and passes them as context to the LLM.",
        "embeddings": "Embeddings are float vectors representing the meaning of text.",
        "tools":      "Tools are Python functions agents call to take real-world actions.",
        "llm":        "LLMs are large neural networks trained on text to generate language.",
        "chains":     "LCEL chains pipe prompt | llm | parser together.",
    }
    for k, v in kb.items():
        if k in query.lower():
            return v
    return "Not found."

_agent_tools = [calculator, knowledge_lookup]

_agent_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use tools when needed.

Tools: {tools}

Question: {input}
Thought: {agent_scratchpad}
Action: one of [{tool_names}]
Action Input: tool input
Observation: result
...
Thought: I know the answer
Final Answer: clear answer
""")

_agent_chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def agent_service(task: str) -> tuple[str, int]:
    """Run an agent task. Returns (result, step_count)."""
    agent    = create_react_agent(llm=_agent_chat, tools=_agent_tools, prompt=_agent_prompt)
    executor = AgentExecutor(agent=agent, tools=_agent_tools, verbose=False,
                             max_iterations=6, handle_parsing_errors=True,
                             return_intermediate_steps=True)

    result = executor.invoke({"input": task})
    steps  = len(result.get("intermediate_steps", []))
    return result["output"], steps
