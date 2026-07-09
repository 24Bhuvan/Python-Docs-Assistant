"""
utils/embeddings.py

Embedding Service for the Python Documentation Assistant.

Responsibilities
----------------
- Initialize the local Ollama embedding model.
- Perform an optional health check.
- Return a reusable embedding model.

This module does NOT:
- Load documents
- Generate document embeddings
- Build FAISS
- Store vectors
"""

from langchain_ollama import OllamaEmbeddings


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "nomic-embed-text"

OLLAMA_BASE_URL = "http://localhost:11434"


# ---------------------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------------------

def get_embedding_model() -> OllamaEmbeddings:
    """
    Initialize and return the embedding model.

    Returns
    -------
    OllamaEmbeddings
        Configured embedding model.
    """

    try:

        embedding_model = OllamaEmbeddings(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
        )

        return embedding_model

    except Exception as exc:

        print(f"\nFailed to initialize embedding model:\n{exc}")

        raise


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------

def verify_embedding_service(
    embedding_model: OllamaEmbeddings,
) -> None:
    """
    Verify that the embedding service is operational.

    Performs a single test embedding to ensure:

    - Ollama is running
    - Model exists
    - Embeddings can be generated
    """

    print("\n" + "=" * 60)
    print("Embedding Service Summary")
    print("=" * 60)

    print(f"Embedding Model : {MODEL_NAME}")

    try:

        # Single lightweight health check
        test_embedding = embedding_model.embed_query("health check")

        print("Initialization Status : SUCCESS")
        print("Ollama Connection     : CONNECTED")
        print("Health Check          : PASSED")
        print(f"Embedding Dimension   : {len(test_embedding)}")

    except Exception as exc:

        print("Initialization Status : FAILED")
        print("Ollama Connection     : FAILED")
        print("Health Check          : FAILED")

        print(f"\nReason:\n{exc}")

        raise

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------
# Standalone Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    embedding_model = get_embedding_model()

    verify_embedding_service(embedding_model)