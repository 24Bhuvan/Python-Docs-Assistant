const API_URL = "http://127.0.0.1:8000";
const TIMEOUT_MS = 60000; // Allow local model generation enough time to complete

export const sendChatMessage = async (messageText) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: messageText }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle standard HTTP error payloads
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: HTTP ${response.status}`);
    }

    const data = await response.json();
    
    // Explicit structural verification for malformed or missing JSON fields
    if (!data || typeof data.answer !== "string") {
      throw new Error("Invalid response format received from API server.");
    }

    return data.answer;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error.name === "AbortError") {
      throw new Error("The request timed out. Please verify your local model backend is responsive.");
    }
    if (error.message.includes("Failed to fetch")) {
      throw new Error("The backend server is unavailable. Is the FastAPI service running on port 8000?");
    }
    
    throw error;
  }
};