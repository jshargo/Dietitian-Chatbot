import React, { useState } from 'react'

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setAnswer('');
    const source = new EventSource(`http://localhost:8000/stream?query=${encodeURIComponent(query)}`);

    source.onmessage = (event) => {
      if (event.data === '[DONE]') {
        source.close();
      } else {
        setAnswer(prev => prev + event.data + "\n");
      }
    };

    source.onerror = () => {
      source.close();
    };
  };

  return (
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "20px" }}>
      <h1>RAG Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          placeholder="Ask a question..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{ width: "100%", padding: "10px", marginBottom: "10px" }}
        />
        <button type="submit" style={{ padding: "10px" }}>Send</button>
      </form>
      <div style={{ whiteSpace: "pre-wrap", marginTop: "20px", background: "#f0f0f0", padding: "10px" }}>
        {answer}
      </div>
    </div>
  );
}

export default App
