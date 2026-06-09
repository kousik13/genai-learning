import os
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from groq import Groq
from dotenv import load_dotenv

# -------------------------
# Load API Key
# -------------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------
# Read PDF
# -------------------------

reader = PdfReader(
    "company_handbook_5_pages.pdf"
)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# -------------------------
# Chunking
# -------------------------

chunks = text.split("\n\n")

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]

print(f"\nTotal Chunks: {len(chunks)}")

# -------------------------
# Embedding Model
# -------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = model.encode(
    chunks
)

# -------------------------
# Create FAISS Index
# -------------------------

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    np.array(chunk_embeddings)
)

print("FAISS Index Created")

print("\nPDF Chatbot Ready!")
print("Type 'exit' to quit\n")

# -------------------------
# Chat Loop
# -------------------------

while True:

    question = input(
        "Ask Question: "
    )

    if question.lower() == "exit":
        break

    # -------------------------
    # Query Embedding
    # -------------------------

    query_embedding = model.encode(
        [question]
    )

    # -------------------------
    # FAISS Search
    # -------------------------

    distances, indices = index.search(
        np.array(query_embedding),
        k=1
    )

    best_index = indices[0][0]

    context = chunks[best_index]

    # -------------------------
    # Ask LLM
    # -------------------------

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content":
                """
                Answer only using
                the provided context.

                If answer is not found,
                say:

                Information not found
                in document.
                """
            },

            {
                "role": "user",
                "content":
                f"""
                Context:

                {context}

                Question:

                {question}
                """
            }
        ]
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    print("\nAnswer:\n")
    print(answer)

    print("\n" + "-" * 50 + "\n")