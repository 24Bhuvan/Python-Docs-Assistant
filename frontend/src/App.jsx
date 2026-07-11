import React, { useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { ChatInput } from "./components/ChatInput";
import { sendChatMessage } from "./services/api";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSendMessage = async (text) => {
    setError(null);
    const userMessage = { role: "user", content: text };
    
    // Optimistically update conversation layout stream
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const assistantAnswer = await sendChatMessage(text);
      setMessages((prev) => [...prev, { role: "assistant", content: assistantAnswer }]);
    } catch (err) {
      // Direct assignment of explicit semantic error strings thrown by api.js
      setError(err.message || "An unexpected error occurred during request transmission.");
    } finally {
      setIsLoading(false);
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
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        <p className="footer-credits">Powered by React + FastAPI Backend</p>
      </footer>
    </div>
  );
}

export default App;