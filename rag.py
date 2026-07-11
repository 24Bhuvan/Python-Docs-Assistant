"""
RAG Pipeline for the Python Documentation Assistant.

Responsibilities
----------------
- Load FAISS vector store
- Load embedding model
- Initialize LLM (Phi-3 Mini)
- Create retriever
- Build custom RAG pipeline (manual implementation)
- Check retrieval quality before LLM call
- Return answer and retrieved documents

This module does NOT:
- Implement Streamlit UI
- Use ConversationalRetrievalChain
- Implement chatbot interface
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from prompts import (
    SYSTEM_PROMPT,
    CONTEXT_TEMPLATE,
    REFUSAL_RESPONSE,
    INSUFFICIENT_CONTEXT_RESPONSE,
)
from utils.embeddings import get_embedding_model
from utils.vector_store import load_vector_store


# =====================================================================
# Configuration
# =====================================================================

# LLM Configuration
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:latest")
LLM_TEMPERATURE = 0
LLM_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_LLM_TIMEOUT_SECONDS", "90"))

# Retriever Configuration
RETRIEVER_K = 5

# Vector Store Path
PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# RAG Pipeline Class
# =====================================================================

class RAGPipeline:
    """
    Custom RAG Pipeline for Python Documentation Assistant.

    Pipeline Flow:
    1. User Question → Retriever
    2. Retriever → Relevant Documents (k=5)
    3. Check retrieval quality
    4. If meaningful: Prompt Template
    5. Prompt → Phi-3 Mini
    6. Return answer + retrieved documents
    """

    def __init__(self):
        """Initialize the RAG pipeline with FAISS, embeddings, and LLM."""
        logger.info("Initializing RAG Pipeline...")

        try:
            # Step 1: Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_model: OllamaEmbeddings = get_embedding_model()
            logger.info(f"✓ Embedding model loaded: nomic-embed-text")

            # Step 2: Load FAISS vector store
            logger.info("Loading FAISS vector store...")
            self.vector_store: FAISS = load_vector_store(
                embedding_model=self.embedding_model,
                input_dir=VECTOR_STORE_DIR,
            )
            logger.info(f"✓ FAISS vector store loaded from {VECTOR_STORE_DIR}")

            # Step 3: Create retriever
            logger.info(f"Creating retriever with k={RETRIEVER_K}...")
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K},
            )
            logger.info(f"✓ Retriever initialized (k={RETRIEVER_K})")

            # Step 4: Initialize LLM (Phi-3 Mini)
            logger.info(f"Initializing LLM: {LLM_MODEL}...")
            self.llm = ChatOllama(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
            )
            logger.info(f"✓ LLM initialized: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")

            logger.info("✓ RAG Pipeline initialized successfully!")

        except Exception as exc:
            logger.error(f"Failed to initialize RAG Pipeline: {exc}")
            raise

    def retrieve(self, question: str) -> List[Document]:
        """
        Retrieve relevant documents from FAISS.

        Parameters
        ----------
        question : str
            User question

        Returns
        -------
        List[Document]
            List of retrieved documents (up to k=5)
        """
        logger.info(f"Retrieving documents for question: '{question}'")

        try:
            documents = self.retriever.invoke(question)
            logger.info(f"✓ Retrieved {len(documents)} documents")

            # Log retrieved document titles
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source", "Unknown")
                logger.info(f"  [{i}] {source}")

            return documents

        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            raise

    def _check_retrieval_quality(self, documents: List[Document]) -> bool:
        """
        Check if retrieved documents are meaningful.

        Parameters
        ----------
        documents : List[Document]
            Retrieved documents

        Returns
        -------
        bool
            True if retrieval is meaningful, False otherwise
        """
        if not documents:
            logger.warning("⚠ No documents retrieved")
            return False

        # Check if documents have meaningful content
        meaningful = False
        for doc in documents:
            content = doc.page_content.strip()
            if len(content) > 50:  # Arbitrary threshold for meaningful content
                meaningful = True
                break

        if not meaningful:
            logger.warning("⚠ Retrieved documents lack meaningful content")

        return meaningful

    def _build_context(self, documents: List[Document]) -> str:
        """
        Build context string from retrieved documents.

        Parameters
        ----------
        documents : List[Document]
            Retrieved documents

        Returns
        -------
        str
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"[Document {i} - {source}]\n{content}")

        context = "\n\n".join(context_parts)
        return context

    def generate(self, question: str) -> Tuple[str, List[Document]]:
        """
        Generate answer using the RAG pipeline.

        Pipeline:
        1. Retrieve documents
        2. Check retrieval quality
        3. Build prompt if retrieval is good
        4. Call LLM
        5. Return answer + retrieved documents

        Parameters
        ----------
        question : str
            User question

        Returns
        -------
        Tuple[str, List[Document]]
            (answer, retrieved_documents)
        """
        logger.info("=" * 70)
        logger.info("Starting RAG Pipeline Execution")
        logger.info("=" * 70)

        try:
            # Step 1: Retrieve documents
            documents = self.retrieve(question)

            # Step 2: Check retrieval quality
            if not self._check_retrieval_quality(documents):
                logger.warning("Retrieval quality check failed")
                logger.info("Returning insufficient context response")
                logger.info("=" * 70)
                return INSUFFICIENT_CONTEXT_RESPONSE, []

            # Step 3: Build context
            logger.info("Building context from retrieved documents...")
            context = self._build_context(documents)

            # Step 4: Build final prompt
            logger.info("Constructing prompt...")
            full_prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ]

            # Step 5: Call LLM
            logger.info("Calling LLM (Phi-3 Mini)...")
            response = self.llm.invoke(messages)
            answer = response.content

            logger.info("✓ Response generated successfully")
            logger.info("=" * 70)

            return answer, documents

        except Exception as exc:
            logger.error(f"RAG Pipeline execution failed: {exc}")
            raise

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Public interface for querying the RAG pipeline.

        Parameters
        ----------
        question : str
            User question

        Returns
        -------
        Tuple[str, List[Document]]
            (answer, retrieved_documents)
        """
        return self.generate(question)


# =====================================================================
# Initialization
# =====================================================================

def initialize_rag_pipeline() -> RAGPipeline:
    """
    Initialize and return the RAG pipeline.

    Returns
    -------
    RAGPipeline
        Initialized RAG pipeline instance
    """
    return RAGPipeline()

if __name__ == "__main__":
    rag = initialize_rag_pipeline()

    question = "What is a list?"

    answer, docs = rag.query(question)

    print("\nAnswer:")
    print(answer)

    print("\nRetrieved Documents:")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.metadata.get('relative_path', 'Unknown')}")