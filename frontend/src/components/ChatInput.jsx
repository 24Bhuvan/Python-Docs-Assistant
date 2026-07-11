import React, { useState } from "react";

export const ChatInput = ({ onSendMessage, isLoading }) => {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || isLoading) return;
    
    onSendMessage(text.trim());
    setText("");
  };

  return (
    <form className="input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        className="chat-input"
        placeholder="Ask a question about Python..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={isLoading}
      />
      <button type="submit" className="send-button" disabled={!text.trim() || isLoading}>
        {isLoading ? "..." : "Send"}
      </button>
    </form>
  );
};