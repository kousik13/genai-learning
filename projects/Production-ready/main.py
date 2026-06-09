import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv
from groq import Groq

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================
# FastAPI App
# ==========================

app = FastAPI(
    title="AI Chatbot API",
    description="Production Ready Chatbot API",
    version="1.0"
)

# ==========================
# Request Model
# ==========================

class ChatRequest(BaseModel):
    question: str

# ==========================
# Health Check Endpoint
# ==========================

@app.get("/")
def home():

    return {
        "message": "Chatbot API Running"
    }

# ==========================
# Chat Endpoint
# ==========================

@app.post("/chat")
def chat(request: ChatRequest):

    # ----------------------
    # Validation
    # ----------------------

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": request.question
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )