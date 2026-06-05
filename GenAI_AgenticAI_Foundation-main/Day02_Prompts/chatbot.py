# Day 2 — GenAI & Agentic AI Foundation
# chatbot.py — A simple multi-turn chatbot with memory using message history

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt with a placeholder that injects full chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful GenAI tutor. Answer clearly and encourage learning."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | chat

# In-memory chat history
chat_history = []

def chat_with_bot(user_input: str) -> str:
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Store the exchange in history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    return response.content

# -----------------------------------------
# Demo conversation
# -----------------------------------------
print("=== Day 2: Multi-turn Chatbot ===\n")

questions = [
    "What is a large language model?",
    "How is it different from a traditional ML model?",
    "Give me a real-world example of what I just asked about."
]

for q in questions:
    print(f"You: {q}")
    answer = chat_with_bot(q)
    print(f"Bot: {answer}\n")

print(f"Total messages in history: {len(chat_history)}")
