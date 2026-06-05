# Day 3 — GenAI & Agentic AI Foundation
# json_output.py — Force the LLM to return structured JSON output

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("=== Day 3: Structured JSON Output from LLM ===\n")

# ------------------------------------------
# 1. Simple JSON extraction via prompt
# ------------------------------------------
print("--- Example 1: Extract structured info from text ---\n")

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You extract information and return ONLY a valid JSON object.
No explanation, no markdown, no backticks. Pure JSON only."""),
    ("human", """Extract the following fields from this text and return JSON:
- name
- role
- company
- skill

Text: 'Priya is a Machine Learning Engineer at TechCorp. She specializes in NLP.'""")
])

chain = json_prompt | chat | StrOutputParser()
raw = chain.invoke({})

print("Raw LLM output:")
print(raw)

try:
    parsed = json.loads(raw)
    print("\nParsed Python dict:")
    print(parsed)
    print(f"\nName: {parsed.get('name')}, Role: {parsed.get('role')}")
except json.JSONDecodeError as e:
    print(f"\nJSON parse error: {e}")

# ------------------------------------------
# 2. Generate a list of items as JSON
# ------------------------------------------
print("\n--- Example 2: Generate a JSON list of AI use cases ---\n")

list_prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY a JSON array. No extra text, no markdown."),
    ("human", "Give me 4 real-world use cases of Agentic AI as a JSON array of objects with 'use_case' and 'industry' fields.")
])

chain2 = list_prompt | chat | StrOutputParser()
raw2 = chain2.invoke({})

print("Raw output:")
print(raw2)

try:
    parsed2 = json.loads(raw2)
    print("\nParsed list:")
    for i, item in enumerate(parsed2, 1):
        print(f"  {i}. [{item.get('industry')}] {item.get('use_case')}")
except json.JSONDecodeError as e:
    print(f"\nJSON parse error: {e}")
