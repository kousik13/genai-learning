# GenAI & Agentic AI Foundation — 10-Day Curriculum

A structured, hands-on 10-day learning path for Generative AI and Agentic AI
using LangChain and Groq. Each day builds on the previous one.

---

## Project Structure

```
GenAI_Agentic_AI/
├── Day01_Setup/
│   ├── main.py              ← First LLM call, verify setup
│   └── .env                 ← API keys (never commit this)
│
├── Day02_Prompts/
│   ├── prompts.py           ← PromptTemplate & ChatPromptTemplate
│   └── chatbot.py           ← Multi-turn chatbot with message history
│
├── Day03_Chatbot_Streaming/
│   ├── stream_chat.py       ← Streaming LLM output token by token
│   └── json_output.py       ← Structured JSON extraction from LLM
│
├── Day04_Embeddings/
│   ├── embeddings.py        ← HuggingFace text embeddings
│   └── similarity_search.py ← Cosine similarity & ranking
│
├── Day05_VectorStore/
│   ├── vector_store.py      ← FAISS: create, save, reload
│   └── search_vector.py     ← MMR, threshold, and retriever types
│
├── Day06_RAG/
│   ├── document_loader.py   ← Load & chunk documents
│   ├── rag_pipeline.py      ← Full RAG: load → embed → retrieve → generate
│   └── retriever.py         ← Conversational RAG with chat history
│
├── Day07_Agents_Tools/
│   ├── agent.py             ← ReAct agent with custom tools
│   └── tools.py             ← Define, test & inspect @tool functions
│
├── Day08_Assistant/
│   ├── assistant.py         ← Full assistant with tools + memory
│   ├── calculator_tool.py   ← Rich calculator tool + agent demo
│   └── search_tool.py       ← Search tool + agent demo
│
├── Day09_API_App/
│   ├── app.py               ← FastAPI application entry point
│   ├── routes.py            ← API route definitions
│   └── services.py          ← Business logic: chat, RAG, agent
│
└── Day10_Production/
    ├── logger.py            ← Structured JSON logging + decorator
    ├── Dockerfile           ← Containerise the Day 9 API
    └── requirements.txt     ← All Python dependencies
```

---

## Quick Start

### 1. Clone / unzip the project

### 2. Install dependencies
```bash
pip install -r Day10_Production/requirements.txt
```

### 3. Set up your API key
Copy `Day01_Setup/.env` to the root and fill in your Groq key:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 4. Run Day 1
```bash
python Day01_Setup/main.py
```

---

## Day-by-Day Guide

| Day | Folder                   | What you learn                                  |
|-----|--------------------------|-------------------------------------------------|
| 1   | Day01_Setup              | Setup, .env, first LLM call                     |
| 2   | Day02_Prompts            | Prompt templates, multi-turn chatbot            |
| 3   | Day03_Chatbot_Streaming  | Streaming output, structured JSON from LLM      |
| 4   | Day04_Embeddings         | HuggingFace embeddings, cosine similarity       |
| 5   | Day05_VectorStore        | FAISS vector store, retrievers, MMR             |
| 6   | Day06_RAG                | Document loading, full RAG pipeline, conv. RAG  |
| 7   | Day07_Agents_Tools       | ReAct agent, @tool decorator, tool metadata     |
| 8   | Day08_Assistant          | Full assistant, calculator tool, search tool    |
| 9   | Day09_API_App            | FastAPI REST API for chat, RAG, and agents      |
| 10  | Day10_Production         | Structured logging, Dockerfile, requirements    |

---

## Tech Stack

- **LLM**: Groq (llama-3.1-8b-instant / llama-3.3-70b-versatile)
- **Framework**: LangChain (LCEL, agents, RAG)
- **Embeddings**: HuggingFace sentence-transformers
- **Vector DB**: FAISS
- **API**: FastAPI + Uvicorn
- **Container**: Docker

---

## Notes

- All files use `YOUR_GROQ_API_KEY` placeholder — replace via `.env`
- Day 9 API runs on `http://localhost:8000` — visit `/docs` for Swagger UI
- Day 10 Dockerfile wraps Day 9 — build with `docker build -t genai-api .`
- The search tool in Day 8 is mocked — swap for Tavily or DuckDuckGo in production
