# Day 9 — GenAI & Agentic AI Foundation
# routes.py — API route definitions (FastAPI router)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import chat_service, rag_service, agent_service

router = APIRouter()

# ------------------------------------------
# Request / Response models
# ------------------------------------------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class RAGRequest(BaseModel):
    question: str
    documents: list[str] = []

class AgentRequest(BaseModel):
    task: str

class ChatResponse(BaseModel):
    answer: str
    tokens_used: int = 0

class RAGResponse(BaseModel):
    answer: str
    sources: list[str] = []

class AgentResponse(BaseModel):
    result: str
    steps: int = 0

# ------------------------------------------
# Chat endpoint
# ------------------------------------------
@router.post("/chat", response_model=ChatResponse, summary="Single-turn or multi-turn chat")
async def chat_endpoint(req: ChatRequest):
    """
    Send a message to the GenAI chatbot.
    Optionally include conversation history for multi-turn context.
    """
    try:
        result = chat_service(req.message, req.history)
        return ChatResponse(answer=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------
# RAG endpoint
# ------------------------------------------
@router.post("/rag", response_model=RAGResponse, summary="RAG-powered Q&A")
async def rag_endpoint(req: RAGRequest):
    """
    Answer a question using Retrieval-Augmented Generation.
    Provide your own documents list, or use the default knowledge base.
    """
    try:
        answer, sources = rag_service(req.question, req.documents)
        return RAGResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------
# Agent endpoint
# ------------------------------------------
@router.post("/agent", response_model=AgentResponse, summary="Agentic task execution")
async def agent_endpoint(req: AgentRequest):
    """
    Run an agentic task. The agent has access to calculator and search tools.
    """
    try:
        result, steps = agent_service(req.task)
        return AgentResponse(result=result, steps=steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------
# List available routes
# ------------------------------------------
@router.get("/routes", summary="List all API routes")
def list_routes():
    return {
        "routes": [
            {"POST": "/api/v1/chat",   "description": "Multi-turn LLM chat"},
            {"POST": "/api/v1/rag",    "description": "RAG Q&A with documents"},
            {"POST": "/api/v1/agent",  "description": "Agentic task execution"},
        ]
    }
