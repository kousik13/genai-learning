from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import os

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Choose Persona Dynamically
# -----------------------------
print("Choose AI Persona")
print("1. ATS Recruiter")
print("2. Career Coach")
print("3. Technical Interviewer")

choice = input("\nEnter choice: ")

if choice == "1":
    system_prompt = """
You are a Senior ATS Recruiter.

Responsibilities:
- Review resumes
- Identify strengths
- Identify weaknesses
- Suggest ATS improvements
"""
elif choice == "2":
    system_prompt = """
You are an experienced Career Coach.

Responsibilities:
- Guide career growth
- Suggest learning paths
- Recommend skills
- Improve employability
"""
elif choice == "3":
    system_prompt = """
You are a Senior Technical Interviewer.

Responsibilities:
- Evaluate technical readiness
- Identify skill gaps
- Suggest interview preparation
- Assess project quality
"""
else:
    system_prompt = "You are a helpful AI assistant."

# -----------------------------
# Read PDF Resume
# -----------------------------
pdf_path = "resume.pdf"

reader = PdfReader(pdf_path)

resume_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        resume_text += text + "\n"

print("\nResume loaded successfully.\n")

# -----------------------------
# Message History
# -----------------------------
messages = [
    {
        "role": "system",
        "content": system_prompt
    }
]

# -----------------------------
# Initial Resume Analysis
# -----------------------------
messages.append(
    {
        "role": "user",
        "content": f"""
Analyze the following resume.

Give:

1. Overall Score
2. Skills Found
3. Strengths
4. Weaknesses
5. Missing Skills
6. Suggestions

Resume:

{resume_text}
"""
    }
)

print("=" * 60)
print("INITIAL ANALYSIS")
print("=" * 60)
print()

# -----------------------------
# Streaming Response
# -----------------------------
stream = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    temperature=0.3,
    stream=True
)

assistant_response = ""

for chunk in stream:

    if chunk.choices[0].delta.content:

        token = chunk.choices[0].delta.content

        print(token, end="", flush=True)

        assistant_response += token

print("\n")

# -----------------------------
# Store Assistant Message
# -----------------------------
messages.append(
    {
        "role": "assistant",
        "content": assistant_response
    }
)

# -----------------------------
# Interactive Chat Loop
# -----------------------------
print("\nYou can now ask questions about the resume.")
print("Type 'change system' to switch persona.")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    # -------------------------
    # Change System Prompt
    # -------------------------
    if user_input.lower() == "change system":

        print("\nChoose New Persona")
        print("1. ATS Recruiter")
        print("2. Career Coach")
        print("3. Technical Interviewer")

        new_choice = input("Choice: ")

        if new_choice == "1":
            messages[0]["content"] = "You are a Senior ATS Recruiter."
        elif new_choice == "2":
            messages[0]["content"] = "You are an experienced Career Coach."
        elif new_choice == "3":
            messages[0]["content"] = "You are a Senior Technical Interviewer."

        print("\nSystem role updated.\n")
        continue

    # -------------------------
    # User Role
    # -------------------------
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # -------------------------
    # Streaming Response
    # -------------------------
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        stream=True
    )

    print("\nAI: ", end="")

    assistant_response = ""

    for chunk in stream:

        if chunk.choices[0].delta.content:

            token = chunk.choices[0].delta.content

            print(token, end="", flush=True)

            assistant_response += token

    print("\n")

    # -------------------------
    # Assistant Role
    # -------------------------
    messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )