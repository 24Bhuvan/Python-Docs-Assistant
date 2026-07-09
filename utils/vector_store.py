"""Module for building, persisting, and verifying the FAISS vector store with rate-limited small batching."""

import pickle
import time
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from utils.embeddings import get_embedding_model

# Constants
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH: Path = PROJECT_ROOT / "knowledge_base" / "processed" / "documents.pkl"
VECTOR_STORE_DIR: Path = PROJECT_ROOT / "vector_store"
TOP_K: int = 2
BATCH_SIZE: int = 100
BATCH_DELAY_SECONDS: float = 0.2
TEST_QUERIES: List[str] = [
    "What is a list?",
    "How do I open a file?",
    "Explain decorators."
]
EMBEDDING_MODEL_NAME: str = "nomic-embed-text"


def load_documents(path: Path = DOCUMENTS_PATH) -> List[Document]:
    """Load serialized chunked LangChain documents from a pickle file.
    
    Args:
        path: Path to the documents.pkl file.
        
    Returns:
        A list of LangChain Document objects.
        
    Raises:
        FileNotFoundError: If the documents file does not exist.
        RuntimeError: If document loading fails due to unpickling errors.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Documents serialization file not found at: {path}")
    
    try:
        with open(path, "rb") as f:
            documents = pickle.load(f)
        if not isinstance(documents, list):
            raise TypeError("Deserialized object is not a list.")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load documents from {path}: {str(e)}") from e


def build_vector_store(documents: List[Document], embedding_model: Embeddings) -> FAISS:
    """Construct a FAISS vector store incrementally in throttled small batches to keep Ollama runner stable.
    
    Args:
        documents: List of LangChain Document objects to index.
        embedding_model: Initialized LangChain embedding service.
        
    Returns:
        An initialized and fully populated FAISS vector store object.
        
    Raises:
        ValueError: If the documents list is empty.
        RuntimeError: If vector store creation or incremental batch addition fails.
    """
    if not documents:
        raise ValueError("Cannot build vector store with an empty document list.")

    total_chunks = len(documents)
    total_batches = ((total_chunks - 1) // BATCH_SIZE) + 1
    print(f"Building vector store for {total_chunks} documents using incremental batching (Size: {BATCH_SIZE}, Delay: {BATCH_DELAY_SECONDS}s)...")
    
    try:
        # Task 1: Initialize the vector store using the first small chunk batch
        first_batch = documents[:BATCH_SIZE]
        print(f" -> Initializing index with batch 1/{total_batches} ({len(first_batch)} chunks)")
        vector_store = FAISS.from_documents(documents=first_batch, embedding=embedding_model)
        time.sleep(BATCH_DELAY_SECONDS)
        
        # Task 2: Incrementally add subsequent document groups with pacing throttles
        for i in range(BATCH_SIZE, total_chunks, BATCH_SIZE):
            current_batch = documents[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            print(f" -> Processing batch {batch_num}/{total_batches} ({len(current_batch)} chunks)")
            
            vector_store.add_documents(documents=current_batch)
            time.sleep(BATCH_DELAY_SECONDS)
            
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS vector store incrementally: {str(e)}") from e


def save_vector_store(vector_store: FAISS, output_dir: Path = VECTOR_STORE_DIR) -> None:
    """Serialize and save the FAISS index and chunk mappings locally to disk.
    
    Args:
        vector_store: The FAISS vector store instance to serialize.
        output_dir: Directory where index files will be stored.
        
    Raises:
        RuntimeError: If files fail to generate or save operations fail.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(folder_path=str(output_dir))
        
        index_file = output_dir / "index.faiss"
        pkl_file = output_dir / "index.pkl"
        
        if not index_file.is_file() or not pkl_file.is_file():
            raise RuntimeError("FAISS artifacts are missing after execution of save operation.")
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS vector store to {output_dir}: {str(e)}") from e


def load_vector_store(embedding_model: Embeddings, input_dir: Path = VECTOR_STORE_DIR) -> FAISS:
    """Reload an existing local FAISS vector store from disk.
    
    Args:
        embedding_model: Initialized LangChain embedding service used during index construction.
        input_dir: Directory containing index.faiss and index.pkl.
        
    Returns:
        The deserialized FAISS vector store object.
        
    Raises:
        FileNotFoundError: If index directory or target files do not exist.
        RuntimeError: If index parsing or loading fails.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Vector store directory does not exist at: {input_dir}")
        
    try:
        vector_store = FAISS.load_local(
            folder_path=str(input_dir),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to reload FAISS vector store from {input_dir}: {str(e)}") from e


def verify_vector_store(vector_store: FAISS, queries: List[str] = TEST_QUERIES, top_k: int = TOP_K) -> None:
    """Perform a retrieval validation test using semantic queries without LLM invocation.
    
    Args:
        vector_store: The active reloaded FAISS instance to query against.
        queries: List of test strings to search against the index.
        top_k: Number of relevant context chunks to fetch per query.
        
    Raises:
        RuntimeError: If query execution or retrieval fails.
    """
    try:
        for query in queries:
            results = vector_store.similarity_search(query=query, k=top_k)
            
            print("============================================================")
            print(f"Query: {query}")
            print("============================================================")
            
            for rank, doc in enumerate(results, start=1):
                metadata = doc.metadata
                title = metadata.get("title", "N/A")
                section = metadata.get("section", "N/A")
                source = metadata.get("relative_path", metadata.get("source", "N/A"))
                
                normalized_content = " ".join(doc.page_content.split())
                preview = normalized_content[:200]
                if len(normalized_content) > 200:
                    preview += "..."
                
                print(f"Top Result Rank: {rank}")
                print(f"Title          : {title}")
                print(f"Section        : {section}")
                print(f"Source         : {source}")
                print(f"Preview        : {preview}")
                print("-" * 60)
            print()
    except Exception as e:
        raise RuntimeError(f"Failed retrieval verification: {str(e)}") from e


def run_pipeline() -> None:
    """Execute the sequential workflow tasks for the Phase 4B pipeline stage."""
    status_embedding: str = "FAILED"
    status_persistence: str = "FAILED"
    status_retrieval: str = "FAILED"
    
    chunks_loaded: int = 0
    chunks_indexed: int = 0
    ntotal_vectors: int = 0
    expected_vectors: int = 0

    # Task 1: Load Chunks
    try:
        documents = load_documents(DOCUMENTS_PATH)
        chunks_loaded = len(documents)
    except Exception as e:
        print(f"[ERROR] Stage - Load Documents: {e}")
        raise e

    # Task 2: Embedding Service Initialization & Basic Connectivity Test
    try:
        embedding_model = get_embedding_model()
        embedding_model.embed_query("Connectivity check")
        status_embedding = "SUCCESS"
    except Exception as e:
        print(f"[ERROR] Stage - Initialize Embedding Model: {e}")
        raise e

    # Task 3: Index Construction
    try:
        vector_store = build_vector_store(documents, embedding_model)
        chunks_indexed = len(documents)
        ntotal_vectors = vector_store.index.ntotal
        expected_vectors = chunks_indexed
        
        if ntotal_vectors != expected_vectors:
            print(f"[WARNING] Mismatch: Vector total ({ntotal_vectors}) != Expected ({expected_vectors})")
    except Exception as e:
        print(f"[ERROR] Stage - Build Vector Store: {e}")
        raise e

    # Task 4: Store Serialization
    try:
        save_vector_store(vector_store, VECTOR_STORE_DIR)
        status_persistence = "SUCCESS"
    except Exception as e:
        print(f"[ERROR] Stage - Save Vector Store: {e}")
        raise e

    # Task 5: Store Deserialization Verification
    try:
        reloaded_store = load_vector_store(embedding_model, VECTOR_STORE_DIR)
    except Exception as e:
        print(f"[ERROR] Stage - Load Vector Store Verification: {e}")
        raise e

    # Task 6: Semantic Retrieval Verification
    print("\n--- BEGIN RETRIEVAL VERIFICATION ---")
    try:
        verify_vector_store(reloaded_store, TEST_QUERIES, TOP_K)
        status_retrieval = "SUCCESS"
    except Exception as e:
        print(f"[ERROR] Stage - Verify Vector Store: {e}")
        raise e
    print("--- END RETRIEVAL VERIFICATION ---\n")

    # Output Summary Section
    print("============================================================")
    print("FAISS Vector Store Creation")
    print("============================================================")
    print(f"Chunk Source    : {DOCUMENTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Embedding Model : {EMBEDDING_MODEL_NAME}")
    print(f"Vector Store    : {VECTOR_STORE_DIR.relative_to(PROJECT_ROOT)}/")
    print("============================================================")
    print(f"Chunks Loaded    : {chunks_loaded}")
    print(f"Chunks Indexed   : {chunks_indexed}")
    print(f"Indexed Vectors  : {ntotal_vectors}")
    print(f"Expected Vectors : {expected_vectors}")
    print(f"Status           : SUCCESS")
    print(f"Embedding Model  : {status_embedding}")
    print(f"Persistence      : {status_persistence}")
    print(f"Retrieval        : {status_retrieval}")
    print("============================================================")


if __name__ == "__main__":
    run_pipeline()