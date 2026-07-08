# Project Structure

```text
Python-Docs-Assistant/

├── app.py
├── config.py
├── chatbot.py
├── rag.py
├── prompts.py
│
├── knowledge_base/
│   ├── pdfs/
│   └── processed/
│
├── vector_store/
│
├── models/
│
├── utils/
│   ├── loader.py
│   ├── splitter.py
│   └── embeddings.py
│
├── tests/
│
├── screenshots/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Directory Overview

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit application entry point |
| `chatbot.py` | Chatbot workflow and response generation |
| `rag.py` | Retrieval-Augmented Generation pipeline |
| `config.py` | Project configuration and constants |
| `prompts.py` | Prompt templates for the LLM |
| `knowledge_base/pdfs/` | Official Python documentation PDFs |
| `knowledge_base/processed/` | Cleaned and chunked documents |
| `vector_store/` | Persistent Chroma vector database |
| `models/` | Local LLM or embedding model files (if required) |
| `utils/loader.py` | PDF loading utilities |
| `utils/splitter.py` | Document chunking utilities |
| `utils/embeddings.py` | Embedding generation utilities |
| `tests/` | Unit and integration tests |
| `screenshots/` | Project screenshots for documentation |
| `README.md` | Project overview and setup instructions |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |