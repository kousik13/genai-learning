# Day 2 — GenAI & Agentic AI Foundation
# prompts.py — PromptTemplate and ChatPromptTemplate patterns

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

print("=== 1. Basic PromptTemplate (no LLM call, just format) ===\n")

# Simple single-variable template
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in 3 bullet points for a beginner."
)

filled = template.format(topic="neural networks")
print("Filled prompt:")
print(filled)

print("\n=== 2. Multi-variable PromptTemplate ===\n")

multi_template = PromptTemplate(
    input_variables=["role", "task", "language"],
    template="You are a {role}. Your task is: {task}. Respond in {language}."
)

print(multi_template.format(role="data scientist", task="explain overfitting", language="English"))

print("\n=== 3. ChatPromptTemplate with system + human messages ===\n")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} tutor. Keep answers short and clear."),
    ("human", "{question}")
])

chain = chat_prompt | chat

response = chain.invoke({
    "domain": "GenAI",
    "question": "What is a transformer model?"
})

print("Response:")
print(response.content)

print("\n=== 4. Few-shot style prompt ===\n")

few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "You classify text sentiment. Examples:\n'I love this!' → Positive\n'This is terrible.' → Negative\n'It's okay.' → Neutral"),
    ("human", "Classify this: {text}")
])

few_shot_chain = few_shot_template | chat

response = few_shot_chain.invoke({"text": "This product exceeded my expectations!"})
print("Sentiment classification:")
print(response.content)
