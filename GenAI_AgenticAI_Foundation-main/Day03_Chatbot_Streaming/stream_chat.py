# Day 3 — GenAI & Agentic AI Foundation
# stream_chat.py — Streaming LLM responses token by token

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("=== Day 3: Streaming LLM Output ===\n")

# ------------------------------------------
# 1. Direct streaming on the LLM
# ------------------------------------------
print("--- Method 1: Direct LLM Streaming ---\n")
print("Asking: 'Explain Agentic AI in 5 lines.'\n")
print("Answer (streaming): ", end="", flush=True)

for chunk in chat.stream("Explain Agentic AI in 5 lines."):
    print(chunk.content, end="", flush=True)

print("\n")

# ------------------------------------------
# 2. Streaming through a full chain
# ------------------------------------------
print("--- Method 2: Streaming via a Prompt | LLM | Parser chain ---\n")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer."),
    ("human", "Describe {concept} in exactly 4 sentences.")
])

chain = prompt | chat | StrOutputParser()

print("Asking about 'RAG (Retrieval-Augmented Generation)':\n")
print("Answer (streaming): ", end="", flush=True)

for chunk in chain.stream({"concept": "RAG (Retrieval-Augmented Generation)"}):
    print(chunk, end="", flush=True)

print("\n")

# ------------------------------------------
# 3. Collecting streamed output into a string
# ------------------------------------------
print("--- Method 3: Collect all streamed chunks into a variable ---\n")

full_response = ""
for chunk in chat.stream("What makes an AI system 'agentic'?"):
    full_response += chunk.content

print("Full collected response:")
print(full_response)
