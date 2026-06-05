# Day 8 — GenAI & Agentic AI Foundation
# search_tool.py — A mock search tool + agent demo (swap for real search API)

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

load_dotenv()

print("=== Day 8: Search Tool ===\n")

# ------------------------------------------
# Mock search tool
# In production: replace with DuckDuckGo, SerpAPI, Tavily, etc.
# ------------------------------------------

MOCK_SEARCH_DB = {
    "langchain":     "LangChain is a framework for developing applications powered by LLMs, with chains, agents, and tools.",
    "groq":          "Groq is an AI inference company offering extremely fast LLM inference using LPU chips.",
    "faiss":         "FAISS (Facebook AI Similarity Search) is an efficient library for dense vector similarity search.",
    "react agent":   "ReAct is an agent pattern that interleaves Reasoning and Acting steps in a loop.",
    "agentic ai":    "Agentic AI systems autonomously plan multi-step actions, use tools, and adapt to observations.",
    "rag":           "RAG (Retrieval-Augmented Generation) augments LLM generation with retrieved context from a knowledge base.",
    "vector store":  "A vector store indexes document embeddings for fast semantic similarity retrieval.",
    "llama":         "LLaMA is Meta's open-weight large language model series used widely in research and production.",
}

@tool
def search(query: str) -> str:
    """
    Search for information about a topic.
    Good for: LangChain, Groq, FAISS, ReAct agent, Agentic AI, RAG, vector store, LLaMA.
    Returns a short factual summary.
    """
    query_lower = query.strip().lower()
    for key, value in MOCK_SEARCH_DB.items():
        if key in query_lower:
            return f"Search result for '{query}': {value}"
    return f"No specific results found for '{query}'. Try a more specific term."

# ------------------------------------------
# Direct tool tests
# ------------------------------------------
print("--- Direct search tests ---\n")

for q in ["What is LangChain?", "Tell me about FAISS", "What is pizza?", "Explain Agentic AI"]:
    result = search.invoke({"query": q})
    print(f"  Q: {q}")
    print(f"  A: {result}\n")

# ------------------------------------------
# Agent using the search tool
# ------------------------------------------
print("--- Agent using search tool ---\n")

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You are a research assistant. Use the search tool to find information.

Tools: {tools}

Format:
Question: the question
Thought: reasoning
Action: search
Action Input: search query
Observation: result
Thought: I know the answer
Final Answer: clear, complete answer

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

agent    = create_react_agent(llm=chat, tools=[search], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[search], verbose=True,
                         max_iterations=5, handle_parsing_errors=True)

questions = [
    "What is FAISS and how does it relate to vector stores?",
    "Explain the ReAct agent pattern.",
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = executor.invoke({"input": q})
    print(f"Answer: {result['output']}\n")
