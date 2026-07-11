import React from "react";

export const Message = ({ role, content }) => {
  const isUser = role === "user";
  
  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`message-avatar ${isUser ? "user-avatar" : "assistant-avatar"}`}>
        {isUser ? "👤" : "🐍"}
      </div>
      <div className={`message-bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        {content}
      </div>
    </div>
  );
};