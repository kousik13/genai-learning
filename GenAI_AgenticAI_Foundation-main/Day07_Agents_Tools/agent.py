# Day 7 — GenAI & Agentic AI Foundation
# agent.py — Build a ReAct agent that uses tools to answer questions

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

load_dotenv()

print("=== Day 7: ReAct Agent with Tools ===\n")

# Use a larger model for better reasoning
chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------------------------
# Define tools
# ------------------------------------------
@tool
def multiply(expression: str) -> str:
    """Multiply two numbers. Input format: 'a * b' e.g. '12 * 7'"""
    try:
        parts = expression.replace(" ", "").split("*")
        result = float(parts[0]) * float(parts[1])
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    count = len(text.split())
    return f"The text has {count} words."

@tool
def is_palindrome(word: str) -> str:
    """Check if a word or phrase is a palindrome."""
    clean = word.lower().replace(" ", "")
    result = "is" if clean == clean[::-1] else "is NOT"
    return f"'{word}' {result} a palindrome."

tools = [multiply, word_count, is_palindrome]

# ------------------------------------------
# ReAct prompt
# ------------------------------------------
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use available tools to answer questions accurately.

Tools available:
{tools}

Format EXACTLY:
Question: the input question
Thought: what to do next
Action: tool name (one of [{tool_names}])
Action Input: the input to the tool
Observation: the tool result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the complete answer

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# ------------------------------------------
# Build and run agent
# ------------------------------------------
agent = create_react_agent(llm=chat, tools=tools, prompt=prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True
)

print("=== Test 1: Math ===")
r1 = executor.invoke({"input": "What is 47 multiplied by 13?"})
print("Answer:", r1["output"], "\n")

print("=== Test 2: Word count ===")
r2 = executor.invoke({"input": "How many words are in: 'Agentic AI systems plan and act autonomously'?"})
print("Answer:", r2["output"], "\n")

print("=== Test 3: Palindrome ===")
r3 = executor.invoke({"input": "Is 'racecar' a palindrome?"})
print("Answer:", r3["output"])
