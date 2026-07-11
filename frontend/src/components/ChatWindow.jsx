import React, { useEffect, useRef } from "react";
import { Message } from "./Message";

export const ChatWindow = ({ messages, isLoading }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-window">
      {messages.length === 0 ? (
        <div className="welcome-container">
          <h2>Python Documentation Assistant</h2>
          <p>Ask me anything about official Python modules, functions, or language syntax.</p>
        </div>
      ) : (
        messages.map((msg, index) => (
          <Message key={index} role={msg.role} content={msg.content} />
        ))
      )}

      {isLoading && (
        <div className="message-row assistant-row">
          <div className="message-avatar assistant-avatar">🐍</div>
          <div className="message-bubble assistant-bubble loading-bubble">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};