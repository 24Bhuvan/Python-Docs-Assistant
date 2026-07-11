import pytest
from unittest.mock import MagicMock, patch
from langchain_ollama import OllamaEmbeddings

# Adjust import mapping based on your directory layout
from utils.embeddings import (
    get_embedding_model,
    verify_embedding_service,
    MODEL_NAME,
    OLLAMA_BASE_URL
)


# =====================================================================
# 1. Unit Tests for Model Instantiation
# =====================================================================

def test_get_embedding_model_initialization():
    """Verify that the embedding model is instantiated with the correct parameters."""
    with patch("utils.embeddings.OllamaEmbeddings") as mock_ollama_class:
        model = get_embedding_model()
        
        # Verify the class constructor was called with the configured parameters
        mock_ollama_class.assert_called_once_with(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )
        assert model == mock_ollama_class.return_value


def test_get_embedding_model_bubbles_exceptions():
    """Verify that initialization failures are bubbled up appropriately."""
    with patch("utils.embeddings.OllamaEmbeddings", side_effect=Exception("Ollama instantiation failed")):
        with pytest.raises(Exception) as exc_info:
            get_embedding_model()
        
        assert "Ollama instantiation failed" in str(exc_info.value)


# =====================================================================
# 2. Unit Tests for Health Check Validation
# =====================================================================

def test_verify_embedding_service_success(capsys):
    """Verify a successful query embedding health check outputs correct metrics."""
    mock_model = MagicMock(spec=OllamaEmbeddings)
    # Simulate a standard dense vector response (e.g., length 3 for the test assertion)
    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]
    
    verify_embedding_service(mock_model)
    
    # Assert model method was triggered with correct text string
    mock_model.embed_query.assert_called_once_with("health check")
    
    # Verify the successful console output text
    captured = capsys.readouterr()
    assert "Initialization Status : SUCCESS" in captured.out
    assert "Ollama Connection     : CONNECTED" in captured.out
    assert "Health Check          : PASSED" in captured.out
    assert "Embedding Dimension   : 3" in captured.out


def test_verify_embedding_service_failure(capsys):
    """Verify failing health check steps catch exceptions and print structured logs."""
    mock_model = MagicMock(spec=OllamaEmbeddings)
    mock_model.embed_query.side_effect = Exception("Connection Refused by Host")
    
    with pytest.raises(Exception) as exc_info:
        verify_embedding_service(mock_model)
        
    assert "Connection Refused by Host" in str(exc_info.value)
    
    # Verify the failure logs printed cleanly to stdout
    captured = capsys.readouterr()
    assert "Initialization Status : FAILED" in captured.out
    assert "Ollama Connection     : FAILED" in captured.out
    assert "Reason:" in captured.out
    assert "Connection Refused by Host" in captured.out