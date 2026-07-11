import pytest
import pickle
from pathlib import Path
from langchain_core.documents import Document

# Adjust import mapping based on your directory layout
from utils.splitter import (
    split_documents,
    save_documents,
    process_documents
)


# =====================================================================
# 1. Unit Tests for Splitting Strategy
# =====================================================================

def test_split_documents_generates_chunks():
    """Verify that a large document is broken into smaller chunks with metadata intact."""
    # A string that exceeds the CHUNK_SIZE configuration (1000 characters)
    large_text = "Python statement body sequence text standard data format chunks. " * 30
    metadata = {"title": "Test Doc", "section": "core"}
    original_doc = Document(page_content=large_text, metadata=metadata)
    
    chunks = split_documents([original_doc])
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 1000  # Enforces CHUNK_SIZE
        assert chunk.metadata["title"] == "Test Doc"
        assert chunk.metadata["section"] == "core"


def test_split_documents_preserves_short_documents():
    """Verify short documents within limits pass through without unnecessary splitting."""
    short_text = "Short python snippet code definition statement."
    original_doc = Document(page_content=short_text, metadata={"id": 123})
    
    chunks = split_documents([original_doc])
    
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
    assert chunks[0].metadata["id"] == 123


# =====================================================================
# 2. Unit Tests for Serialization and Storage Layer
# =====================================================================

def test_save_documents_writes_pickle_payload(tmp_path, monkeypatch):
    """Verify file structure generation and clean binary serialization of components."""
    # Build isolated fake path directories
    fake_processed_dir = tmp_path / "processed"
    fake_output_file = fake_processed_dir / "documents.pkl"
    
    # Mock global variables inside splitter module targeting our ephemeral storage
    monkeypatch.setattr("utils.splitter.PROCESSED_DIR", fake_processed_dir)
    monkeypatch.setattr("utils.splitter.OUTPUT_FILE", fake_output_file)
    
    sample_chunks = [
        Document(page_content="Chunk content A", metadata={"idx": 0}),
        Document(page_content="Chunk content B", metadata={"idx": 1})
    ]
    
    save_documents(sample_chunks)
    
    # Assert structural side-effects occurred correctly
    assert fake_output_file.exists()
    
    with open(fake_output_file, "rb") as f:
        loaded_chunks = pickle.load(f)
        
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0].page_content == "Chunk content A"
    assert loaded_chunks[1].metadata["idx"] == 1


# =====================================================================
# 3. Integration Tests for End-to-End Processing
# =====================================================================

def test_process_documents_pipeline(tmp_path, monkeypatch, capsys):
    """Verify the execution orchestration layer saves state and prints summary logging statements."""
    fake_processed_dir = tmp_path / "processed"
    fake_output_file = fake_processed_dir / "documents.pkl"
    
    monkeypatch.setattr("utils.splitter.PROCESSED_DIR", fake_processed_dir)
    monkeypatch.setattr("utils.splitter.OUTPUT_FILE", fake_output_file)
    
    docs = [Document(page_content="Execution integration validation logic loop testing.", metadata={"a": "b"})]
    
    result_chunks = process_documents(docs)
    
    # Verify processing outcomes
    assert len(result_chunks) == 1
    assert fake_output_file.exists()
    
    # Check that print_summary wrote expected output metrics to stdout
    captured = capsys.readouterr()
    assert "Document Splitter Summary" in captured.out
    assert "Original Documents : 1" in captured.out