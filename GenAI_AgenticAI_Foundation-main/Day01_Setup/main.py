# Day 1 — GenAI & Agentic AI Foundation
# main.py — Entry point: verify setup and test your first LLM call

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Initialize the LLM using Groq
chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Your very first GenAI call
print("=== Day 1: GenAI Foundation — First LLM Call ===\n")

response = chat.invoke("What is Generative AI? Explain in 5 simple lines.")
print("Response from LLM:")
print(response.content)

print("\n=== Setup Complete! Move on to Day 2 ===")
