from groq import Groq
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Store conversation history
messages = []

print("🤖 AI Chatbot Started")
print("Type 'exit' to quit\n")

while True:

    # Take user input
    user_question = input("You: ")

    # Exit chatbot
    if user_question.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Handle empty input
    if user_question == "":
        print("⚠️ Please enter a question.\n")
        continue

    # Save user message
    messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    try:

        # Send entire conversation history
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=200,
            messages=messages
        )

        # Extract AI response
        ai_response = response.choices[0].message.content

        # Save AI response to memory
        messages.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )

        # Print AI response
        print("\nAI:")
        print(ai_response)
        print()

    except Exception as e:

        print("\n❌ Error occurred:")
        print(e)
        print()