import pytest
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# Adjust import mapping based on your directory layout
from utils.vector_store import (
    load_documents,
    build_vector_store,
    save_vector_store,
    load_vector_store,
    verify_vector_store,
    run_pipeline
)


# =====================================================================
# 1. Unit Tests for Serialization and File Verification
# =====================================================================

def test_load_documents_success(tmp_path):
    """Verify that chunked documents are correctly unpickled from a file[cite: 3]."""
    test_file = tmp_path / "documents.pkl"
    expected_docs = [Document(page_content="Content", metadata={"title": "Test"})]
    
    with open(test_file, "wb") as f:
        pickle.dump(expected_docs, f)
        
    loaded_docs = load_documents(test_file)
    assert len(loaded_docs) == 1
    assert loaded_docs[0].page_content == "Content"


def test_load_documents_file_missing():
    """Verify FileNotFoundError is raised if the target path does not exist[cite: 3]."""
    with pytest.raises(FileNotFoundError):
        load_documents(Path("non_existent_path/docs.pkl"))


def test_load_documents_invalid_type(tmp_path):
    """Verify TypeError inside RuntimeError handling when unpickled data is not a list[cite: 3]."""
    test_file = tmp_path / "invalid.pkl"
    with open(test_file, "wb") as f:
        pickle.dump("Not a list string", f)
        
    with pytest.raises(RuntimeError) as exc_info:
        load_documents(test_file)
    assert isinstance(exc_info.value.__cause__, TypeError)
    assert str(exc_info.value.__cause__) == "Deserialized object is not a list."


# =====================================================================
# 2. Unit Tests for Index Construction Strategy
# =====================================================================

@patch("utils.vector_store.time.sleep", return_value=None)  # Bypass pacing delays
@patch("langchain_community.vectorstores.FAISS.from_documents")
def test_build_vector_store_success(mock_from_docs, mock_sleep):
    """Verify incremental batch chunk consumption loops execute accurately[cite: 3]."""
    mock_embeddings = MagicMock(spec=Embeddings)
    mock_faiss_instance = MagicMock(spec=FAISS)
    mock_from_docs.return_value = mock_faiss_instance
    
    # Provide 150 documents to trigger initialization + 1 subsequent batch loop
    sample_docs = [Document(page_content=f"Text {i}") for i in range(150)]
    
    with patch.object(mock_faiss_instance, "add_documents") as mock_add_docs:
        vector_store = build_vector_store(sample_docs, mock_embeddings)
        
        # Initial batch should build index
        mock_from_documents_args = mock_from_docs.call_args[1]
        assert len(mock_from_documents_args["documents"]) == 100  # Enforces BATCH_SIZE
        
        # Subsequent batch should invoke add_documents
        mock_add_docs.assert_called_once()
        assert len(mock_add_docs.call_args[1]["documents"]) == 50
        assert vector_store == mock_faiss_instance


def test_build_vector_store_empty_list():
    """Verify empty document allocations raise ValueError[cite: 3]."""
    mock_embeddings = MagicMock(spec=Embeddings)
    with pytest.raises(ValueError, match="Cannot build vector store with an empty document list"):
        build_vector_store([], mock_embeddings)


# =====================================================================
# 3. Unit Tests for Persistence Operations
# =====================================================================

def test_save_vector_store_success(tmp_path):
    """Verify save actions enforce the creation of file system artifacts[cite: 3]."""
    mock_faiss = MagicMock(spec=FAISS)
    output_dir = tmp_path / "vector_store"
    
    # Side effect simulation: creating index artifacts upon execution
    def side_effect(folder_path):
        Path(folder_path, "index.faiss").touch()
        Path(folder_path, "index.pkl").touch()
        
    mock_faiss.save_local.side_effect = side_effect
    
    save_vector_store(mock_faiss, output_dir)
    assert (output_dir / "index.faiss").is_file()
    assert (output_dir / "index.pkl").is_file()


def test_save_vector_store_missing_artifacts(tmp_path):
    """Verify RuntimeError fires if save_local fails to generate output files[cite: 3]."""
    mock_faiss = MagicMock(spec=FAISS)
    output_dir = tmp_path / "vector_store"
    
    with pytest.raises(RuntimeError, match="FAISS artifacts are missing"):
        save_vector_store(mock_faiss, output_dir)


@patch("langchain_community.vectorstores.FAISS.load_local")
def test_load_vector_store_success(mock_load_local, tmp_path):
    """Verify reloaded storage elements map accurately across dangerous deserialization settings[cite: 3]."""
    mock_embeddings = MagicMock(spec=Embeddings)
    input_dir = tmp_path / "vector_store"
    input_dir.mkdir()
    
    load_vector_store(mock_embeddings, input_dir)
    mock_load_local.assert_called_once_with(
        folder_path=str(input_dir),
        embeddings=mock_embeddings,
        allow_dangerous_deserialization=True
    )


# =====================================================================
# 4. Unit Tests for Verification Logic & Orchestration
# =====================================================================

def test_verify_vector_store_query_loop(capsys):
    """Verify similarity searches extract meta properties and print formatted rank blocks[cite: 3]."""
    mock_faiss = MagicMock(spec=FAISS)
    mock_doc = Document(
        page_content="Target response logic output strings verified.",
        metadata={"title": "Doc Title", "section": "lib", "relative_path": "lib/test.html"}
    )
    mock_faiss.similarity_search.return_value = [mock_doc]
    
    verify_vector_store(mock_faiss, queries=["What is a list?"], top_k=1)
    
    mock_faiss.similarity_search.assert_called_once_with(query="What is a list?", k=1)
    
    captured = capsys.readouterr()
    assert "Query: What is a list?" in captured.out
    assert "Title          : Doc Title" in captured.out
    assert "Section        : lib" in captured.out
    assert "Preview        : Target response logic output strings verified." in captured.out


@patch("utils.vector_store.verify_vector_store", return_value=None)
@patch("utils.vector_store.load_vector_store")
@patch("utils.vector_store.save_vector_store", return_value=None)
@patch("utils.vector_store.build_vector_store")
@patch("utils.vector_store.get_embedding_model")
@patch("utils.vector_store.load_documents")
def test_run_pipeline_orchestration(
    mock_load_docs, mock_get_embedding, mock_build, mock_save, mock_load_store, mock_verify, capsys
):
    """Verify full orchestration phases build the matrix and map status output blocks cleanly[cite: 3]."""
    mock_load_docs.return_value = [Document(page_content="Pipeline text chunk entry.")]
    
    mock_embedding_instance = MagicMock()
    mock_get_embedding.return_value = mock_embedding_instance
    
    mock_faiss_instance = MagicMock()
    mock_faiss_instance.index.ntotal = 1
    mock_build.return_value = mock_faiss_instance
    mock_load_store.return_value = mock_faiss_instance
    
    run_pipeline()
    
    captured = capsys.readouterr()
    assert "Chunks Loaded    : 1" in captured.out
    assert "Indexed Vectors  : 1" in captured.out
    assert "Embedding Model  : SUCCESS" in captured.out
    assert "Persistence      : SUCCESS" in captured.out
    assert "Retrieval        : SUCCESS" in captured.out