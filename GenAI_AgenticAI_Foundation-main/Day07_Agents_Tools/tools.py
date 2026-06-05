# Day 7 — GenAI & Agentic AI Foundation
# tools.py — Define, test, and explore LangChain @tool functions

from langchain.tools import tool
import json
import math

print("=== Day 7: LangChain Tools ===\n")

# ------------------------------------------
# 1. Basic tool
# ------------------------------------------
@tool
def square_root(number: str) -> str:
    """Calculate the square root of a positive number."""
    try:
        n = float(number)
        if n < 0:
            return "Cannot calculate square root of a negative number."
        return f"√{n} = {math.sqrt(n):.4f}"
    except ValueError:
        return "Please provide a valid number."

# ------------------------------------------
# 2. Tool that returns structured data
# ------------------------------------------
@tool
def get_genai_concept(concept: str) -> str:
    """
    Get a short explanation of a GenAI concept.
    Supported: 'rag', 'agent', 'embedding', 'prompt', 'llm', 'tool'
    """
    explanations = {
        "rag": "RAG (Retrieval-Augmented Generation) fetches relevant documents to ground LLM answers in facts.",
        "agent": "An agent is an AI system that plans, uses tools, and takes actions to complete tasks.",
        "embedding": "Embeddings are numerical vectors that represent text semantically in high-dimensional space.",
        "prompt": "A prompt is the input given to an LLM to guide the desired output.",
        "llm": "An LLM (Large Language Model) is a model trained on large text datasets to generate human-like text.",
        "tool": "A tool is a function an agent can call to interact with external systems or perform computations.",
    }
    key = concept.strip().lower()
    return explanations.get(key, f"Concept '{concept}' not found. Try: rag, agent, embedding, prompt, llm, tool.")

# ------------------------------------------
# 3. Tool with JSON output
# ------------------------------------------
@tool
def analyse_text(text: str) -> str:
    """
    Analyse a piece of text. Returns word count, character count,
    sentence count, and average word length as a JSON string.
    """
    words     = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")
    avg_len   = sum(len(w) for w in words) / max(len(words), 1)

    result = {
        "word_count":       len(words),
        "character_count":  len(text),
        "sentence_count":   max(sentences, 1),
        "avg_word_length":  round(avg_len, 2)
    }
    return json.dumps(result)

# ------------------------------------------
# 4. Test each tool directly (without agent)
# ------------------------------------------
print("--- Direct tool invocations ---\n")

print("square_root('144'):")
print(" ", square_root.invoke({"number": "144"}))

print("\nget_genai_concept('rag'):")
print(" ", get_genai_concept.invoke({"concept": "rag"}))

print("\nget_genai_concept('agent'):")
print(" ", get_genai_concept.invoke({"concept": "agent"}))

print("\nanalyse_text(...):")
sample = "Agentic AI can plan and act. It uses tools wisely. This is powerful!"
raw    = analyse_text.invoke({"text": sample})
parsed = json.loads(raw)
print(f"  Text   : {sample}")
print(f"  Result : {parsed}")

# ------------------------------------------
# 5. Inspect tool metadata (what agents see)
# ------------------------------------------
print("\n--- Tool metadata (what agents see) ---\n")
for t in [square_root, get_genai_concept, analyse_text]:
    print(f"  Name       : {t.name}")
    print(f"  Description: {t.description}")
    print()
