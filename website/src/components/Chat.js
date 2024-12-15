// frontend/src/components/Chat.js
import React, { useState } from 'react';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = () => {
    const userMessage = input;
    setMessages([...messages, { role: 'user', content: userMessage }]);
    setInput('');

    const url = `/chat?message=${encodeURIComponent(userMessage)}&user_id=1&agent_id=1`;
    const evtSource = new EventSource(url);

    evtSource.onmessage = function(e) {
      setMessages(prevMessages => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.role === 'assistant') {
          // Append to last assistant message
          const updatedMessage = {
            ...lastMessage,
            content: lastMessage.content + e.data,
          };
          return [...prevMessages.slice(0, -1), updatedMessage];
        } else {
          // Create new assistant message
          return [...prevMessages, { role: 'assistant', content: e.data }];
        }
      });
    };

    evtSource.onerror = function(err) {
      console.error("EventSource failed:", err);
      evtSource.close();
    };
  };

  return (
    <div>
      <div style={{ maxHeight: '400px', overflowY: 'scroll' }}>
        {messages.map((msg, idx) => (
          <div key={idx}>
            <strong>{msg.role}:</strong> {msg.content}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyPress={e => { if (e.key === 'Enter') sendMessage(); }}
        style={{ width: '80%' }}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default Chat;
