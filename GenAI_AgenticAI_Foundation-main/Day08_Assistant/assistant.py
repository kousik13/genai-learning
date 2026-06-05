# Day 8 — GenAI & Agentic AI Foundation
# assistant.py — A fully capable AI assistant with tools, memory, and streaming

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
import math
import json

load_dotenv()

print("=== Day 8: Full AI Assistant ===\n")

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------------------------
# Tools
# ------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression safely.
    Examples: '15 * 4', '100 / 5', '2 ** 8', 'sqrt(144)'
    """
    try:
        expr = expression.replace("sqrt(", "math.sqrt(")
        result = eval(expr, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Math error: {str(e)}"

@tool
def knowledge_base(query: str) -> str:
    """
    Look up GenAI and Agentic AI concepts.
    Ask about: agents, rag, tools, llm, embeddings, chains, memory, streaming.
    """
    kb = {
        "agents":     "Agents are AI systems that loop through: Think → Act with a tool → Observe result → Repeat.",
        "rag":        "RAG retrieves relevant documents from a vector store and injects them into the LLM prompt.",
        "tools":      "Tools are Python functions decorated with @tool that agents call to perform real actions.",
        "llm":        "Large Language Models are neural networks trained on massive text datasets to generate text.",
        "embeddings": "Embeddings map text to high-dimensional float vectors for semantic similarity comparison.",
        "chains":     "Chains (LCEL) pipe together prompt | llm | parser components into a reusable pipeline.",
        "memory":     "Memory stores conversation history so the LLM can give contextually aware responses.",
        "streaming":  "Streaming returns LLM output token by token instead of waiting for the full response.",
    }
    key = query.strip().lower()
    for k, v in kb.items():
        if k in key:
            return v
    return "Topic not found. Try: agents, rag, tools, llm, embeddings, chains, memory, streaming."

@tool
def summarise(text: str) -> str:
    """Summarise a long text into one sentence."""
    words = text.split()
    if len(words) <= 15:
        return text
    return " ".join(words[:12]) + "... [summarised]"

tools = [calculator, knowledge_base, summarise]

# ------------------------------------------
# Prompt
# ------------------------------------------
prompt = PromptTemplate.from_template("""
You are an intelligent GenAI assistant with access to tools.
Use tools when needed. Always be accurate and helpful.

Tools:
{tools}

Format:
Question: the input question
Thought: reasoning step
Action: one of [{tool_names}]
Action Input: input for the tool
Observation: result
... (repeat as needed)
Thought: I know the final answer
Final Answer: your answer

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

agent    = create_react_agent(llm=chat, tools=tools, prompt=prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)

# ------------------------------------------
# Conversation with memory
# ------------------------------------------
conversation_history = []

def ask_assistant(question: str) -> str:
    result = executor.invoke({"input": question})
    answer = result["output"]
    conversation_history.append({"role": "user",      "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    return answer

print("--- Session Start ---\n")

queries = [
    "What is 2 to the power of 10?",
    "Explain what RAG is in the context of GenAI.",
    "What is 1024 divided by 32?",
]

for q in queries:
    print(f"\nYou: {q}")
    answer = ask_assistant(q)
    print(f"Assistant: {answer}")

print(f"\nTotal conversation turns: {len(conversation_history) // 2}")
