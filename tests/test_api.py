import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Standard convention: Import the 'app' instance from the 'api' module
from api import app

client = TestClient(app)


# =====================================================================
# 1. Health Check Endpoint Tests
# =====================================================================

def test_read_root_health_check():
    """Verify that the GET / endpoint returns a 200 OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Python Documentation Assistant API is running smoothly."
    }


# =====================================================================
# 2. Functional Chat Path Tests
# =====================================================================

@patch("api.get_response")
def test_chat_endpoint_success(mock_get_response):
    """Verify that valid inputs yield an appropriate answer payload."""
    mock_get_response.return_value = "A list is a built-in mutable sequence type."
    
    payload = {"message": "What is a list?"}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {
        "answer": "A list is a built-in mutable sequence type.",
        "status": "success"
    }
    mock_get_response.assert_called_once_with("What is a list?")


# =====================================================================
# 3. Request Validation & Input Sanitation Tests
# =====================================================================

def test_chat_endpoint_empty_whitespace_string():
    """Verify that fields containing only whitespace return an HTTP 422 error."""
    payload = {"message": "   \n \t  "}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 422
    assert "cannot be empty or solely whitespace" in response.json()["detail"]


def test_chat_endpoint_missing_message_field():
    """Verify that missing the required 'message' key triggers a 422 validation error."""
    payload = {"wrong_key": "What is a list?"}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "message"]


def test_chat_endpoint_invalid_json_format():
    """Verify that malformed JSON payloads fail parsing with a 422 Unprocessable Entity status."""
    bad_raw_data = "{'message': malformed string missing quotes...}"
    response = client.post("/chat", content=bad_raw_data, headers={"Content-Type": "application/json"})
    
    assert response.status_code == 422


# =====================================================================
# 4. Error Resilience & Recovery Tests
# =====================================================================

@patch("api.get_response", side_effect=RuntimeError("Vector database index read failure"))
def test_chat_endpoint_pipeline_exception_handling(mock_get_response):
    """Verify that unhandled exceptions are captured and converted into clean 500 status codes."""
    payload = {"message": "Query causing unexpected backend failure."}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 500
    assert "An unexpected error occurred internally" in response.json()["detail"]