import os
import re

from dotenv import load_dotenv
from groq import Groq

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================
# Memory
# =====================================

memory = []

# =====================================
# Knowledge Base
# =====================================

with open("knowledge_base.txt", "r") as file:
    knowledge = file.read()

# =====================================
# Calculator Tool
# =====================================

def calculator_tool(query):

    try:

        expression = re.sub(
            r"[^0-9+\-*/(). ]",
            "",
            query
        )

        result = eval(expression)

        return f"Calculator Result: {result}"

    except:

        return "Invalid calculation."

# =====================================
# Search Tool
# =====================================

def search_tool(query):

    query = query.lower()

    paragraphs = knowledge.split("\n\n")

    for para in paragraphs:

        if query in para.lower():

            return f"Search Result:\n\n{para}"

    return "No matching information found."

# =====================================
# LLM Chat
# =====================================

def llm_chat(question):

    messages = memory.copy()

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=messages
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    memory.append(
        {
            "role": "user",
            "content": question
        }
    )

    memory.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer

# =====================================
# Agent
# =====================================

def agent(user_input):

    calculation_pattern = (
        r"^[0-9+\-*/(). ]+$"
    )

    if re.match(
        calculation_pattern,
        user_input.strip()
    ):

        return calculator_tool(
            user_input
        )

    if user_input.lower().startswith(
        "search "
    ):

        search_query = user_input[
            7:
        ]

        return search_tool(
            search_query
        )

    return llm_chat(
        user_input
    )

# =====================================
# Assistant Loop
# =====================================

print("\n🤖 Personal AI Assistant")
print("Type 'exit' to quit.\n")

while True:

    user_input = input(
        "You: "
    )

    if user_input.lower() == "exit":

        print(
            "\nAssistant: Goodbye!"
        )

        break

    response = agent(
        user_input
    )

    print(
        f"\nAssistant: {response}\n"
    )