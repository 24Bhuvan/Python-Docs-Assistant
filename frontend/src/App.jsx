import React, { useRef, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { ChatInput } from "./components/ChatInput";
import { sendChatMessage } from "./services/api";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const isRequestInFlightRef = useRef(false);

  const handleSendMessage = async (text) => {
    const trimmedText = text.trim();

    if (!trimmedText || isRequestInFlightRef.current) {
      return;
    }

    setError(null);
    const userMessage = { role: "user", content: trimmedText };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    isRequestInFlightRef.current = true;

    try {
      const assistantAnswer = await sendChatMessage(trimmedText);
      setMessages((prev) => [...prev, { role: "assistant", content: assistantAnswer }]);
    } catch {
      const assistantErrorMessage = "Unable to connect to the backend. Please try again.";
      setMessages((prev) => [...prev, { role: "assistant", content: assistantErrorMessage }]);
      setError(assistantErrorMessage);
    } finally {
      setIsLoading(false);
      isRequestInFlightRef.current = false;
      inputRef.current?.focus();
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Python Docs Explorer</h1>
        <div className="header-badges">
          <span className="badge">Offline</span>
          <span className="badge">Local LLM</span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="main-content">
        <ChatWindow messages={messages} isLoading={isLoading} />
      </main>

      <footer className="footer-container">
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} inputRef={inputRef} />
        <p className="footer-credits">Powered by React + FastAPI Backend</p>
      </footer>
    </div>
  );
}

export default App;