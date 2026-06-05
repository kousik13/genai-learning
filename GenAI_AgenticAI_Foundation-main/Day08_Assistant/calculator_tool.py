# Day 8 — GenAI & Agentic AI Foundation
# calculator_tool.py — A rich calculator tool with an agent demo

import os
import math
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

load_dotenv()

print("=== Day 8: Calculator Tool ===\n")

# ------------------------------------------
# Rich calculator tool
# ------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate math expressions safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.
    Examples: '25 * 4', 'sqrt(81)', 'log(100)', 'sin(3.14159)'
    """
    try:
        safe_env = {
            "__builtins__": {},
            "math": math,
            "sqrt": math.sqrt,
            "sin":  math.sin,
            "cos":  math.cos,
            "tan":  math.tan,
            "log":  math.log10,
            "ln":   math.log,
            "pi":   math.pi,
            "e":    math.e,
            "abs":  abs,
            "round": round,
        }
        result = eval(expression, safe_env)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as ex:
        return f"Error evaluating '{expression}': {str(ex)}"

# ------------------------------------------
# Test the tool directly
# ------------------------------------------
print("--- Direct tests ---\n")

test_cases = [
    "25 * 4",
    "sqrt(144)",
    "2 ** 10",
    "log(1000)",
    "pi * 5 ** 2",
    "sin(pi / 2)",
    "100 / 0",
]

for expr in test_cases:
    result = calculator.invoke({"expression": expr})
    print(f"  {result}")

# ------------------------------------------
# Agent with calculator
# ------------------------------------------
print("\n--- Agent using calculator tool ---\n")

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You solve math problems using the calculator tool. Show your work.

Tools: {tools}

Format:
Question: {input}
Thought: {agent_scratchpad}
Action: calculator
Action Input: the expression
Observation: result
Thought: I now know the final answer
Final Answer: state the result clearly

Begin!
Question: {input}
Thought: {agent_scratchpad}
""")

agent    = create_react_agent(llm=chat, tools=[calculator], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[calculator], verbose=True,
                         max_iterations=5, handle_parsing_errors=True)

problems = [
    "What is the area of a circle with radius 7? Use pi * r^2.",
    "What is 2 to the power of 16?",
]

for problem in problems:
    print(f"\nProblem: {problem}")
    result = executor.invoke({"input": problem})
    print(f"Final: {result['output']}\n")
