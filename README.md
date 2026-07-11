# Python Documentation Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers Python-related questions using only the official Python HTML documentation. The system runs completely offline using local Ollama models, LangChain, FAISS, FastAPI, and a React frontend.

---

# Project Overview

The Python Documentation Assistant is a domain-specific AI chatbot designed to provide accurate and reliable answers from the official Python documentation.

Instead of relying on internet searches or general LLM knowledge, the application retrieves relevant sections from a locally indexed knowledge base and supplies them as context to a local language model. This Retrieval-Augmented Generation (RAG) approach improves factual accuracy while preventing hallucinations.

The project consists of:

- Local HTML documentation parser
- Document preprocessing and chunking
- Semantic embeddings using Ollama
- FAISS vector database
- Retrieval-Augmented Generation (RAG)
- FastAPI backend
- React frontend
- Domain validation to reject non-Python questions

---

# Architecture Diagram

```text
                          User
                           │
                           ▼
                 React Frontend (Vite)
                           │
                           ▼
                    FastAPI Backend
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
 Python Domain Validation          RAG Pipeline
          │                                 │
 Out-of-domain → Reject          Embed User Query
                                          │
                                          ▼
                          Ollama (nomic-embed-text)
                                          │
                                          ▼
                               FAISS Vector Search
                                          │
                                          ▼
                           Retrieve Relevant Chunks
                                          │
                                          ▼
                           Prompt Construction
                                          │
                                          ▼
                             Ollama (phi3 LLM)
                                          │
                                          ▼
                                   Final Response
```

---

# Folder Structure

```text
Python-Docs-Assistant/
│
├── api.py                         # FastAPI backend entry point
├── chatbot.py                     # Chatbot service
├── rag.py                         # Retrieval-Augmented Generation pipeline
├── prompts.py                     # Prompt templates
├── config.py                      # Project configuration
│
├── frontend/                      # React frontend application
│
├── knowledge_base/
│   ├── raw/                       # Official Python HTML documentation
│   └── processed/
│       └── documents.pkl          # Chunked LangChain documents
│
├── vector_store/
│   ├── index.faiss                # FAISS vector index
│   └── index.pkl                  # FAISS metadata
│
├── models/                        # Reserved for local model assets (optional)
│
├── utils/
│   ├── __init__.py
│   ├── loader.py                  # HTML document loader
│   ├── splitter.py                # Document chunking
│   ├── embeddings.py              # Ollama embedding service
│   └── vector_store.py            # FAISS creation & loading utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_chatbot.py
│   ├── test_embeddings.py
│   ├── test_loader.py
│   ├── test_splitter.py
│   └── test_vector_store.py
│
├── docs/
│   └── project_structure.md
│
├── screenshots/                   # Application screenshots
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Python-Docs-Assistant
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama from:

https://ollama.com

Download the required models:

```bash
ollama pull nomic-embed-text
ollama pull phi3:latest
```

Verify installed models:

```bash
ollama list
```

Start the Ollama server:

```bash
ollama serve
```

---

# Backend Run

## Step 1 — Load Documentation

```bash
python utils/loader.py
```

## Step 2 — Split Documents

```bash
python utils/splitter.py
```

## Step 3 — Create FAISS Vector Store

```bash
python utils/vector_store.py
```

## Step 4 — Start FastAPI

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Frontend Run

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run the React application:

```bash
npm run dev
```

The frontend will be available at:

```
http://localhost:5173
```

---

# API Endpoints

## Health Check

**GET /**

Example Response

```json
{
  "status": "healthy",
  "message": "Python Documentation Assistant API is running successfully."
}
```

---

## Chat Endpoint

**POST /chat**

Request

```json
{
  "message": "What is a list comprehension?"
}
```

Successful Response

```json
{
  "status": "success",
  "answer": "A list comprehension provides a concise way to create lists..."
}
```

Out-of-Domain Response

```json
{
  "status": "error",
  "answer": "I can only answer questions related to Python programming and its official documentation."
}
```

---

# Testing

Run all unit tests:

```bash
pytest
```

Run a specific test:

```bash
pytest tests/test_api.py
```

Generate a coverage report (optional):

```bash
pytest --cov=.
```

---

# Screenshots

Application screenshots can be stored inside the `screenshots/` directory.

Example structure:

```text
screenshots/
├── home_page.png
├── chat_response.png
├── swagger_api.png
└── architecture.png
```

Example README image links:

```markdown
## Home Page

![Home](screenshots/home_page.png)

## Chat Interface

![Chat](screenshots/chat_response.png)

## API Documentation

![Swagger](screenshots/swagger_api.png)
```

---

# Future Improvements

- Implement hybrid search using BM25 and FAISS.
- Add conversational memory for multi-turn interactions.
- Improve document ranking with reranking models.
- Support multiple Python documentation versions.
- Add Docker support for simplified deployment.
- Implement authentication and user session management.
- Add streaming responses for faster user interaction.
- Introduce caching to improve retrieval performance.
- Expand automated unit and integration testing.
- Deploy the application using Docker Compose or Kubernetes.

---

# Technology Stack

| Component | Technology |
|----------|------------|
| Frontend | React + Vite |
| Backend | FastAPI |
| LLM | Ollama (Phi-3) |
| Embeddings | Ollama (nomic-embed-text) |
| Framework | LangChain |
| Vector Database | FAISS |
| Knowledge Base | Official Python HTML Documentation |
| Language | Python 3.10 |
| Testing | Pytest |

---

# License

This project is developed for educational purposes as part of a Python Documentation Assistant assignment.