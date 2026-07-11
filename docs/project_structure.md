# Project Structure

```text
Python-Docs-Assistant/
│
├── api.py                          # FastAPI backend entry point
├── chatbot.py                      # Chatbot service
├── rag.py                          # Retrieval-Augmented Generation pipeline
├── prompts.py                      # Prompt templates
├── config.py                       # Project configuration
│
├── frontend/                       # React frontend application
│
├── knowledge_base/
│   ├── raw/                        # Official Python HTML documentation
│   └── processed/
│       └── documents.pkl           # Chunked LangChain documents
│
├── vector_store/
│   ├── index.faiss                 # FAISS vector index
│   └── index.pkl                   # FAISS metadata
│
├── models/                         # Reserved for local model assets (optional)
│
├── utils/
│   ├── __init__.py
│   ├── loader.py                   # HTML document loader
│   ├── splitter.py                 # Document chunking
│   ├── embeddings.py               # Ollama embedding service
│   └── vector_store.py             # FAISS creation & loading utilities
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
├── screenshots/                    # Application screenshots
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Directory Overview

| Path | Purpose |
|------|---------|
| `api.py` | FastAPI backend exposing REST endpoints for the React frontend |
| `chatbot.py` | Chatbot service that validates user input and communicates with the RAG pipeline |
| `rag.py` | Core Retrieval-Augmented Generation pipeline integrating FAISS, embeddings, and the Phi-3 Mini LLM |
| `config.py` | Centralized project configuration and constants |
| `prompts.py` | System prompts and prompt templates used by the RAG pipeline |
| `frontend/` | React application providing the user interface |
| `knowledge_base/raw/` | Official Python HTML documentation used as the knowledge source |
| `knowledge_base/processed/` | Serialized chunked documents generated during preprocessing |
| `vector_store/` | Persistent FAISS vector index and metadata |
| `models/` | Optional directory reserved for local model assets |
| `utils/loader.py` | Loads and cleans HTML documentation into LangChain documents |
| `utils/splitter.py` | Splits documents into overlapping retrieval chunks |
| `utils/embeddings.py` | Initializes and verifies the local Ollama embedding model |
| `utils/vector_store.py` | Creates, saves, loads, and verifies the FAISS vector store |
| `tests/` | Unit and integration tests covering backend components |
| `docs/` | Project documentation and architecture references |
| `screenshots/` | Screenshots demonstrating the application |
| `README.md` | Project overview, installation, usage, and documentation |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore configuration |
