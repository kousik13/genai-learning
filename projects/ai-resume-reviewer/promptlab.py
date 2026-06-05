from groq import Groq
from dotenv import load_dotenv
import os
import time

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

print("=" * 60)
print("DAY 2 - PROMPT PLAYGROUND & COMPARISON APP")
print("=" * 60)

topic = input("\nEnter a topic: ")

# ----------------------------------------------------
# ZERO SHOT
# ----------------------------------------------------

zero_shot = f"""
Explain the following topic:

{topic}
"""

# ----------------------------------------------------
# ONE SHOT
# ----------------------------------------------------

one_shot = f"""
Example:

Question:
What is Machine Learning?

Answer:
Machine Learning is a branch of AI that enables systems to learn from data and make predictions without explicit programming.

Now answer:

Question:
{topic}
"""

# ----------------------------------------------------
# FEW SHOT
# ----------------------------------------------------

few_shot = f"""
Example 1:

Question:
What is Machine Learning?

Answer:
Machine Learning is a branch of AI that enables systems to learn from data and make predictions without explicit programming.

------------------------------------------------

Example 2:

Question:
What is Deep Learning?

Answer:
Deep Learning is a subset of Machine Learning that uses multi-layer neural networks to learn complex patterns.

------------------------------------------------

Example 3:

Question:
What is Generative AI?

Answer:
Generative AI creates new content such as text, images, audio, and code using trained models.

------------------------------------------------

Now answer:

Question:
{topic}
"""

# ----------------------------------------------------
# PERSONA PROMPT
# ----------------------------------------------------

persona_prompt = f"""
You are a Senior GenAI Architect.

Explain:

{topic}

Cover:
1. Real-world usage
2. Enterprise applications
3. Benefits
4. Challenges
"""

# ----------------------------------------------------
# CHAIN OF THOUGHT
# ----------------------------------------------------

cot_prompt = f"""
Explain the following topic step by step.

Topic:
{topic}

Follow this reasoning process:

1. Definition
2. How it works
3. Why it is used
4. Real-world examples
5. Advantages
6. Limitations
"""

tests = [
    ("ZERO SHOT", zero_shot),
    ("ONE SHOT", one_shot),
    ("FEW SHOT", few_shot),
    ("PERSONA PROMPTING", persona_prompt),
    ("CHAIN OF THOUGHT", cot_prompt)
]

for title, prompt in tests:

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    start = time.time()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    end = time.time()

    print(response.choices[0].message.content)

    print("\n--- Metrics ---")
    print(f"Response Time: {round(end-start,2)} sec")
    print(f"Input Tokens : {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")
    print(f"Total Tokens : {response.usage.total_tokens}")

print("\nCompleted Prompt Comparison.")