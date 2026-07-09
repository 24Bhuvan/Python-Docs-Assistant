"""
utils/splitter.py

Document splitter for the Python Documentation Assistant.

Responsibilities:
- Split LangChain Document objects into retrieval-optimized chunks
- Preserve metadata
- Save chunked documents

This module does NOT:
- Generate embeddings
- Create a FAISS index
- Call an LLM
"""

from pathlib import Path
from typing import List
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "knowledge_base" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "documents.pkl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into retrieval chunks.

    Parameters
    ----------
    documents : List[Document]

    Returns
    -------
    List[Document]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_documents(documents: List[Document]) -> None:
    """
    Serialize chunked documents.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(documents, f)


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def print_summary(
    original_documents: List[Document],
    chunked_documents: List[Document],
) -> None:
    """
    Print verification summary.
    """

    print("\n" + "=" * 60)
    print("Document Splitter Summary")
    print("=" * 60)

    print(f"Original Documents : {len(original_documents)}")
    print(f"Total Chunks       : {len(chunked_documents)}")

    if chunked_documents:

        average_size = sum(
            len(doc.page_content)
            for doc in chunked_documents
        ) / len(chunked_documents)

        print(f"Average Chunk Size : {average_size:.2f}")

        print("\nFirst Chunk")
        print("-" * 60)
        print(chunked_documents[0].page_content)

        print("\nMetadata")
        print("-" * 60)
        print(chunked_documents[0].metadata)

    print(f"\nSaved To : {OUTPUT_FILE}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------------------

def process_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split and save documents.
    """

    chunked_documents = split_documents(documents)

    save_documents(chunked_documents)

    print_summary(
        documents,
        chunked_documents,
    )

    return chunked_documents


# ---------------------------------------------------------------------
# Standalone Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from loader import load_documents

    docs = load_documents()

    process_documents(docs)