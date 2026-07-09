"""
Chatbot service for the Python Documentation Assistant.

Responsibilities
----------------
- Validate user input
- Call the RAG pipeline
- Return the generated response
- Handle errors gracefully

This module is intentionally stateless.

It does NOT:
- Implement Streamlit
- Maintain conversation history
- Implement retrieval logic
- Build prompts
- Initialize FAISS manually
"""

from rag import initialize_rag_pipeline

# ---------------------------------------------------------------------
# Initialize RAG pipeline once
# ---------------------------------------------------------------------

try:
    rag_pipeline = initialize_rag_pipeline()
except Exception:
    rag_pipeline = None


# ---------------------------------------------------------------------
# Chatbot Service
# ---------------------------------------------------------------------

def get_response(query: str) -> str:
    """
    Generate a response for a user question.

    Parameters
    ----------
    query : str
        User question.

    Returns
    -------
    str
        Assistant response.
    """

    # -------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------
    if not isinstance(query, str):
        return "Invalid input. Please enter a text question."

    query = query.strip()

    if not query:
        return "Please enter a question about Python."

    # -------------------------------------------------------------
    # Check RAG initialization
    # -------------------------------------------------------------
    if rag_pipeline is None:
        return (
            "The chatbot service is currently unavailable because "
            "the RAG pipeline could not be initialized."
        )

    # -------------------------------------------------------------
    # Generate answer
    # -------------------------------------------------------------
    try:
        answer, _ = rag_pipeline.query(query)
        return answer

    except Exception:
        return (
            "Sorry, an unexpected error occurred while processing "
            "your request. Please try again."
        )


# ---------------------------------------------------------------------
# Simple CLI test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Python Documentation Assistant")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        response = get_response(question)

        print("\nAssistant:")
        print(response)