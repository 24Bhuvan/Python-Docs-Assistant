import pytest
from unittest.mock import MagicMock, patch

# Adjust import mapping based on your directory layout
import chatbot
from chatbot import get_response


# =====================================================================
# 1. Setup & Teardown Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def reset_chatbot_state():
    """Ensure global pipeline variables are systematically reset before each test execution[cite: 4]."""
    chatbot.rag_pipeline = None
    yield
    chatbot.rag_pipeline = None


# =====================================================================
# 2. Unit Tests for String Parameter Input Validation
# =====================================================================

def test_get_response_invalid_type():
    """Verify system catches non-string arguments gracefully[cite: 4]."""
    # Passing an arbitrary integer instead of expected string input
    response = get_response(12345)
    assert response == "Invalid input. Please enter a text question."


def test_get_response_empty_string():
    """Verify whitespace or empty queries return a localized validation notice[cite: 4]."""
    response = get_response("   ")
    assert response == "Please enter a question about Python."


# =====================================================================
# 3. Unit Tests for RAG Core Pipeline Instantiation Errors
# =====================================================================

@patch("chatbot.initialize_rag_pipeline", side_effect=Exception("FAISS files missing"))
def test_get_response_pipeline_initialization_failure(mock_init):
    """Verify degradation notice returns when initialization throws exceptions[cite: 4]."""
    response = get_response("What is a decorator?")
    
    assert mock_init.call_count == 1
    assert "RAG pipeline could not be initialized" in response


# =====================================================================
# 4. Unit Tests for End-To-End Execution Paths
# =====================================================================

@patch("chatbot.initialize_rag_pipeline")
def test_get_response_successful_pipeline_query(mock_init):
    """Verify clean string answers bubble up successfully under standard conditions[cite: 4]."""
    # Mocking standard LangChain execution output tuples (answer, context_metadata)
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = ("A list is a mutable sequence.", {"source": "docs"})
    mock_init.return_value = mock_pipeline

    response = get_response("What is a list?")
    
    mock_pipeline.query.assert_called_once_with("What is a list?")
    assert response == "A list is a mutable sequence."


@patch("chatbot.initialize_rag_pipeline")
def test_get_response_refuses_non_python_queries(mock_init):
    """Verify out-of-domain questions are refused before the RAG pipeline runs."""
    response = get_response("Who is Narendra Modi?")

    mock_init.assert_not_called()
    assert response == (
        "I'm a Python Documentation Assistant. I can answer questions only about Python programming "
        "and the official Python documentation. Please ask a Python-related question."
    )


@patch("chatbot.initialize_rag_pipeline")
def test_get_response_unexpected_query_exception(mock_init):
    """Verify runtime method queries catch internal framework errors cleanly[cite: 4]."""
    mock_pipeline = MagicMock()
    mock_pipeline.query.side_effect = RuntimeError("Ollama context length limit exhausted")
    mock_init.return_value = mock_pipeline

    response = get_response("Explain advanced metaclasses.")
    
    assert "an unexpected error occurred while processing" in response