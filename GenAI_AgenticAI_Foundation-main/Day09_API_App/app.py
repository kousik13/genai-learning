# Day 9 — GenAI & Agentic AI Foundation
# app.py — FastAPI application entry point

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

# ------------------------------------------
# App setup
# ------------------------------------------
app = FastAPI(
    title="GenAI & Agentic AI API",
    description="A REST API exposing GenAI chat, RAG, and agent capabilities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")

# ------------------------------------------
# Health check
# ------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "GenAI & Agentic AI API is live.",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

# ------------------------------------------
# Run
# ------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
